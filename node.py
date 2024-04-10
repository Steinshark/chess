#Author: Everett Stenberg
#Description:   Acts as the nodes in the MCTree. Contains information about the position, 
#               move, score, visit count, and other data  



import chess
import torch
import chess_utils
import numpy 
import sys
import math 


#Determine device using availability and command line options
if sys.argv and "--cpu" in sys.argv:

    #Force CPU
    DEVICE      = torch.device('cpu')

elif sys.argv and "--cuda" in "".join(sys.argv):

    #If cuda specfied, use that device ID
    cuda_device = [command.replace('--','') for command in sys.argv if '--cuda' in command ][0]
    DEVICE      = torch.device(cuda_device)

    #attempt device check
    try:
        test    = torch.tensor([1,2,3],device=DEVICE)
    except RuntimeError:
        print(f"CUDA id:{cuda_device[6:]} does not exists on machine with {torch.cuda.device_count()} CUDA devices")
        exit()

else:

    #Default to CUDA device
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Node:

    #Determine exploration tendency
    c           = 3
    
    #For easy game outcome sorting
    RESULTS     = {"1/2-1/2":0,
                   "*":0,
                   "1-0":1,
                   "0-1":-1}



    def __init__(self,move:chess.Move,parent,init_p:float|torch.Tensor,depth,turn:bool):

        #Postition related vars
        self.move               = move 
        self.turn               = 1 if turn else -1 
        self.top_score          = -1_000_000 if turn else 1_000_000
        self.op                 = self.maximize if turn else self.minimize
        self.depth              = depth                         #depth from root node

        #Tree related vars
        self.parent:Node        = parent
        self.children           = []
        self.bubble_up_with     = [self] 

        #Node related vars and such 
        self.n_visits           = 0 
        self.init_p             = init_p
        self.n_wins             = 0 
        self.cumulative_score   = 0

        #precompute val for score computation (good speedup)
        self.precompute         = -1*self.turn*self.c * float(self.init_p)


    #Re-pre-compute (when applying dirichlet after first expansion from root, must do this or 
    # pre compute will be off)
    def pre_compute(self):
        self.precompute         = -1*self.turn*self.c * float(self.init_p)


    #Make mctree code cleaner by wrapping this
    def is_leaf(self):
        return not bool(self.children)
    

    #Used when finding best node. Maximizing if parent node is White els Min
    def maximize(self,x,y):
        return x > y 
    
    #See above comment, I just want a comment above each fn
    def minimize(self,x,y):
        return x < y
    

    #Picks best child from the perspective of the node before it. 
    #   If current node is turn 1, then looking to maximize next node score
    def pick_best_child(self):

        #Set top and best score vars for sort
        top_node    = None
        best_score  = self.top_score

        #Traverse and find best next node
        for package in [(node,node.get_score()) for node in self.children]:
            curnode,score  = package
            if self.op(score,best_score):
                best_score      = score 
                top_node        = curnode 

        return top_node
       
            
    #Score is evaluated in a revised PUCT manner. Uses average result as well as exploration tendency and move counts  
    def get_score(self):
        return (self.cumulative_score / (self.n_visits+1)) + self.precompute * (math.sqrt(self.parent.n_visits) / (self.n_visits+1))

    #Used for debugging to prevent gigantic floats
    def get_score_str(self):
        return f"{self.get_score():.3f}"
    

    #Performs "rollout" i.e. returns the network evaluation of the postiion.
    #Uses a lookup dictionary based on position fen to avoid redoing work 
    def run_rollout(self,board:chess.Board,model:torch.nn.Module,board_key:str,lookup_dict,moves,static_gpu:torch.Tensor,static_cpu_p:torch.Tensor,static_cpu_v:torch.Tensor):
        
        #Check dictionary
        if board_key in lookup_dict:
            return lookup_dict[board_key]
        
        #Or do the work
        else:
            with torch.no_grad():   #no grad! (obviously)

                #Representation as a (bs,19,8,8) tensor
                board_repr              = chess_utils.batched_fen_to_tensor([board_key]).half()

                #Perform copy to static memory in GPU (large speedup if using GPU)
                static_gpu.copy_(board_repr)

                #Get model probability distrubtion and evaluation of the position
                probs,eval              = model.forward(static_gpu)

                #Bring it all over to the cpu (Also large speedup due to copy to static variables -no need to reallocate)
                static_cpu_p.copy_(probs[0],non_blocking=True)
                static_cpu_v.copy_(eval[0])

                #Convert to numpy and renormalize
                revised_numpy_probs     = numpy.take(static_cpu_p.numpy(),[chess_utils.MOVE_TO_I[move] for move in moves])
                revised_numpy_probs     = chess_utils.normalize_numpy(revised_numpy_probs,1)
               
                #Add to dictionary and just return from dictionary
                lookup_dict[board_key]  = (revised_numpy_probs,static_cpu_v.numpy())

                return lookup_dict[board_key]
            
    
    #Expand ocurrs for leaf nodes. Checks if this position has been reached before (and at the same depth). If so, it will 
    #   perform the bubble-up for each node and provide even more compute capability. Observed to only kick into effect 
    #   at around 4k explorations
    def expand(self,board:chess.Board,depth:int,chess_model:torch.nn.Module,max_depth:int,static_gpu,static_cpu_p,static_cpu_v,lookup_dict={},common_nodes={}):

        #Fen used as key
        position_key                    = board.fen()

        #Checks for same node 
        if position_key in common_nodes:
            common_nodes[position_key].append(self)
        else:
            common_nodes[position_key]  = [self]

        #Check end state of node. either return actual outcome or perform computation 
        if board.is_game_over() or board.ply() > max_depth:
            return self.RESULTS[board.result()]
        
        #Run "rollout"
        moves                           = list(board.generate_legal_moves())
        probabilities,rollout_val       = self.run_rollout(board,chess_model,position_key,lookup_dict,moves,static_gpu,static_cpu_p,static_cpu_v)

        #Create and populate child nodes
        self.children               = [Node(move,self,probabilities[i],depth+1,not board.turn) for i,move in enumerate(moves)]
        
        return rollout_val


    #Passes up the value recieved at the leaf and updates the visit count
    def bubble_up(self,outcome):

        self.cumulative_score += outcome
        self.n_visits           += 1

        if not self.parent is None:
            self.parent.bubble_up(outcome)
        

    #???
    def data_repr(self):
        return f"{self.move} vis:{self.n_visits},pvis:{self.parent.n_visits if self.parent else 0},win:{self.n_wins},p:{self.init_p:.2f},scr:{self.get_score_str()}"
    

    #Used once a move is pushed onto the actual game board.
    #   This will be called on MCTree root to traverse to the move played and 
    #   keep the computations
    def traverse_to_child(self,move:chess.Move):
        for child in self.children:
            if child.move == move:
                return child
        return -1 
    

    #A node is represented as the sequence of moves that kets it there
    def __repr__(self):
        if self.parent == None:
            return "root"
        return str(self.parent) + " -> " + str(self.move)






if __name__ == '__main__':

    board   = chess.Board()
    root    = Node(None,None,0,board.turn)


    #ALGORITHM - Get to leaf 
    curnode     = root
    print(f"is leaf = {curnode.is_leaf()}")
    while not curnode.is_leaf():
        curnode     = curnode.pick_best_child()
    
    print(f"root is {curnode}")
    print(f"board is\n{board}")
    root.expand(board)

    print(f"is leaf = {curnode.is_leaf()}")