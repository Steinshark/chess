import chess
import torch
import chess_utils
import numpy 
import sys
import math 

#Determine device using availability and --cpu
if sys.argv and "--cpu" in sys.argv:
    DEVICE      = torch.device('cpu')
elif sys.argv and "--cuda" in "".join(sys.argv):
    cuda_device = [command.replace('--','') for command in sys.argv if '--cuda' in command ][0]
    DEVICE      = torch.device(cuda_device)
    #attempt device check
    try:
        test    = torch.tensor([1,2,3],device=DEVICE)
    except RuntimeError:
        print(f"CUDA id:{cuda_device[6:]} does not exists on machine with {torch.cuda.device_count()} CUDA devices")
        exit()
else:
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Node:

    #Class variables 

    #   c is the factor relating to exploration tendency
    c           = 3
    #   Easy board outcome string to value 
    RESULTS     = {"1/2-1/2":0,
                   "*":0,
                   "1-0":1,
                   "0-1":-1}



    def __init__(self,move:chess.Move,parent,init_p:float|torch.Tensor,depth,turn:bool):

        #Game related vars
        self.move               = move 
        self.turn               = 1 if turn else -1 
        if turn:
            self.op             = self.maximize
            self.top_score      = -1_000_000
        else:
            self.op             = self.minimize
            self.top_score      = 1_000_000
        self.depth              = depth                         #depth from root node

        #Node related vars
        self.parent:Node        = parent
        self.children           = []
        self.bubble_up_with     = [self] 

        #Scores and such 
        self.n_visits           = 0 
        self.init_p             = init_p

        #Game stats from here 
        self.n_wins             = 0 

        self.cumulative_score   = 0

        #precompute val
        self.precompute         = -1*self.turn*self.c * float(self.init_p)


    def pre_compute(self):
        self.precompute         = -1*self.turn*self.c * float(self.init_p)


    def is_leaf(self):
        return not bool(self.children)
    

    def maximize(self,x,y):
        return x > y 
    

    def minimize(self,x,y):
        return x < y
    

    #   Picks best child from the perspective of the node before it. 
    #   If current node is turn 1, then looking to maximize next node score
    def pick_best_child(self):

        #Find top node 
        top_node    = None

        #node,score  = []
        best_score  = self.top_score
        for package in [(node,node.get_score()) for node in self.children]:
            curnode,score  = package

            if self.op(score,best_score):
                best_score      = score 
                top_node        = curnode 

        return top_node
       
            
    #   Score is an absolute evaluated regardless of player perspective 
    #       Black advantage == -1 
    #       White advantage == 1 
    #   Evaluation is done by the node in its own perspective when picking a next move 
    def get_score(self):


        return (self.cumulative_score / (self.n_visits+1)) + self.precompute * (math.sqrt(self.parent.n_visits) / (self.n_visits+1))


    def get_score_str(self):
        return f"{self.get_score():.3f}"
    

    def run_rollout(self,board:chess.Board,model:torch.nn.Module,lookup_dict,moves,static_gpu:torch.Tensor,static_cpu_p:torch.Tensor,static_cpu_v:torch.Tensor):
        board_fen   = board.fen()
        board_key   = " ".join(board_fen.split(" ")[:4])
        if board_key in lookup_dict:
            return lookup_dict[board_key]
        else:
            with torch.no_grad():
                board_repr              = chess_utils.batched_fen_to_tensor([board_fen]).half()
                static_gpu.copy_(board_repr)
                probs,eval              = model.forward(static_gpu)

                #Bring it all over to the cpu
                static_cpu_p.copy_(probs[0],non_blocking=True)
                static_cpu_v.copy_(eval[0])

                #Convert to numpy and renormalize
                revised_numpy_probs     = numpy.take(static_cpu_p.numpy(),[chess_utils.MOVE_TO_I[move] for move in moves])
                revised_numpy_probs     = chess_utils.normalize_numpy(revised_numpy_probs,1)
               
                lookup_dict[board_key]  = (revised_numpy_probs,static_cpu_v.numpy())
                return lookup_dict[board_key]
            
            
    def expand(self,board:chess.Board,depth:int,chess_model:torch.nn.Module,max_depth:int,static_gpu,static_cpu_p,static_cpu_v,lookup_dict={},common_nodes={}):

        position_key                    = board.fen()
        if position_key in common_nodes:
            common_nodes[position_key].append(self)
        else:
            common_nodes[position_key]  = [self]

         #Check end state 
        if board.is_game_over() or board.ply() > max_depth:
            return self.RESULTS[board.result()]

        #Run "rollout"
        moves                           = list(board.generate_legal_moves())
        probabilities,rollout_val       = self.run_rollout(board,chess_model,lookup_dict,moves,static_gpu,static_cpu_p,static_cpu_v)

        #Populate children nodes
        with torch.no_grad():
            
            #Add children
            self.children               = [Node(move,self,probabilities[i],depth+1,not board.turn) for i,move in enumerate(moves)]
                
            return rollout_val



    def bubble_up(self,outcome):

        self.cumulative_score += outcome
        self.n_visits           += 1

        if not self.parent is None:
            self.parent.bubble_up(outcome)
        


    def data_repr(self):
        return f"{self.move} vis:{self.n_visits},pvis:{self.parent.n_visits if self.parent else 0},win:{self.n_wins},p:{self.init_p:.2f},scr:{self.get_score_str()}"
    

    def traverse_to_child(self,move:chess.Move):
        for child in self.children:
            if child.move == move:
                return child
        return -1 
    

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