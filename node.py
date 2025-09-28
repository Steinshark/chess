#Author: Everett Stenberg
#Description:   Acts as the nodes in the MCTree. Contains information about the position,
#               move, score, visit count, and other data



import bulletchess
import torch
import utilities
import numpy
import math
import chess_data 

class Node:

    #Determine exploration tendency
    c           = 4

    #For easy game outcome mapping
    RESULTS     = {"1/2-1/2":0,
                   "*":0,
                   "1-0":1,
                   "0-1":-1}



    def __init__(self,move:bulletchess.Move,parent,prior_p:float|torch.Tensor,depth,turn:bool):

        #Postition related vars
        self.move               = move
        self.turn               = 1 if turn == bulletchess.WHITE else -1
        self.top_score          = -1_000_000 if self.turn == 1 else 1_000_000
        self.op                 = {1:self.maximize,-1:self.minimize}[self.turn]
        #print(f"generated node with turn {self.turn}")
        self.depth              = depth                         #depth from root node

        #Tree related vars
        self.parent:Node        = parent
        self.children           = []
        self.bubble_up_with     = [self]

        #Node related vars and such
        self.n_visits           = 0
        self.prior_p            = float(prior_p)
        self.n_wins             = 0
        self.cumulative_score   = 0
        self.key                = None

        #precompute val for score computation (good speedup)
        self.precompute         = -1*self.turn*self.c * self.prior_p


    #Re-pre-compute (when applying dirichlet after first expansion from root, must do this or
    # pre compute will be off)
    def pre_compute(self):
        self.precompute         = -1*self.turn*self.c*self.prior_p


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
        #print(f"using operation {self}")
        #Set top and best score vars for sort
        top_node    = None
        best_score  = self.top_score
        #print(f"use top score {best_score}")
        #Traverse and find best next node
        for package in [(node,node.get_score()) for node in self.children]:
            curnode,score  = package
            if self.op(score,best_score):
                best_score      = score
                top_node        = curnode


        if top_node is None:
            print(f"I think im {self.turn}, using {self.op}")
            input([(node,node.get_score()) for node in self.children])

        return top_node


    #Score is evaluated in a revised PUCT manner. Uses average result as well as exploration tendency and move counts
    def get_score(self):
        output_score    = (self.cumulative_score / (self.n_visits+1)) + self.precompute * (math.sqrt(self.parent.n_visits) / (self.n_visits+1))

        return output_score


    #Return just the q_score (for training)
    def get_q_score(self):
        return self.cumulative_score / (self.n_visits+1)


    #Used for debugging to prevent gigantic floats
    def get_score_str(self):
        return f"{self.get_score():.3f}"


    #Performs "rollout" i.e. returns the network evaluation of the postiion.
    #Uses a lookup dictionary based on position fen to avoid redoing work
    def run_rollout(self,board:bulletchess.Board,model:torch.nn.Module,board_key:str,lookup_dict:dict,moves:list,static_gpu:torch.Tensor,static_cpu_p:torch.Tensor,static_cpu_v:torch.Tensor):

        #Check dictionary
        if board_key in lookup_dict:
            return lookup_dict[board_key]

        #Or do the work
        else:
            with torch.no_grad():   #no grad! (obviously)

                #Representation as a (bs,19,8,8) tensor
                input_ids               = chess_data.tokenize_fen(board,req_grad=False)

                #Add batch dimension
                input_ids.unsqueeze_(dim=0)

                #Perform copy to static memory in GPU (large speedup if using GPU)
                #static_gpu.copy_(board_repr)

                #Get model probability distrubtion and evaluation of the position
                probs,eval              = model.forward(input_ids)

                #Bring it all over to the cpu (Also large speedup due to copy to static variables -no need to reallocate)
                # static_cpu_p.copy_(probs[0],non_blocking=True)
                # static_cpu_v.copy_(eval[0])
                
                #Convert to numpy and renormalize
                revised_numpy_probs     = numpy.take(probs[0].numpy(),[utilities.MOVE_TO_I[move] for move in moves])
                revised_numpy_probs     = utilities.normalize_numpy(revised_numpy_probs,1)

                #Add to dictionary and just return from dictionary
                #lookup_dict[board_key]  = (revised_numpy_probs,static_cpu_v.item())
                lookup_dict[board_key]  = (revised_numpy_probs,eval[0].item())

                return lookup_dict[board_key]


    def helper(board):
        return
    
    
    #Expand occurs for leaf nodes. Checks if this position has been reached before (and at the same depth). If so, it will
    #   perform the bubble-up for each node and provide even more compute capability. Observed to only kick into effect
    #   at around 4k explorations
    def expand(self,board:bulletchess.Board,depth:int,chess_model:torch.nn.Module,max_depth:int,static_gpu,static_cpu_p,static_cpu_v,lookup_dict={},common_nodes={}):

        #Fen used as key
        position_key                    = board.fen()

        #Checks for same node
        if position_key in common_nodes:
            common_nodes[position_key].add(self)
        else:
            common_nodes[position_key]  = {self}

        #Check end state of node. either return actual outcome or perform computation
        if board in bulletchess.FORCED_DRAW or board.fullmove_number > max_depth:
            return 0.0
        
        #Check for a winning condition - whoever's turn it ISNT has won with this move
        if board in bulletchess.CHECKMATE:  
            if board.turn == bulletchess.WHITE:
                return -1.0 
            elif board.turn == bulletchess.BLACK: 
                return 1.0 
            else:
                input(f"help me! {board.turn}")

        #Run rollout of an unexplored position
        moves                           = board.legal_moves()
        probabilities,rollout_val       = self.run_rollout(board,chess_model,position_key,lookup_dict,moves,static_gpu,static_cpu_p,static_cpu_v)

        #Create and populate child nodes
        newturn                         = {1:bulletchess.BLACK,-1:bulletchess.WHITE}[self.turn]
        self.children                   = [Node(move,self,probabilities[i],depth+1,newturn) for i,move in enumerate(moves)]

        return rollout_val


    #Passes up the value recieved at the leaf and updates the visit count
    def bubble_up(self,outcome):
        self.cumulative_score           += outcome
        self.n_visits                   += 1

        if not self.parent is None:
            self.parent.bubble_up(outcome)


    #???
    def data_repr(self):
        return f"{self.move} vis:{self.n_visits},pvis:{self.parent.n_visits if self.parent else 0},win:{self.n_wins},p:{self.prior_p:.2f},scr:{self.get_score_str()}"


    #Used once a move is pushed onto the actual game board.
    #   This will be called on MCTree root to traverse to the move played and
    #   keep the computations
    def traverse_to_child(self,move:bulletchess.Move):
        for child in self.children:
            if child.move == move:
                return child
        raise ValueError(f"Move {move} was not found")


    #A node is represented as the sequence of moves that kets it there
    def __repr__(self):
        if self.parent is None:
            return "root"
        return str(self.parent) + " -> " + str(self.move) + f": {self.get_score():.2f}"



if __name__ == '__main__':

    board   = bulletchess.Board()
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
