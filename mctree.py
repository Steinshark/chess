#Author: Everett Stenberg
#Description:   The class that acts as the game engine. MCTree can conduct and iterative search 
#               of the current chess position



from node import Node 
import chess 
import time 
import model 
import torch
import value_trainer
import numpy
import sys
import chess_utils
from collections import OrderedDict
import random 


#Creates an instance of a Monte-Carlo style Tree
#   to develop an evaluation of a given position, the tree
#   functions as follows:
#       - start at the root position (the one to be evaluated)
#       - for n_iters:
#       -   traverse the tree to the next best leaf
#       -   expand the leaf and determine leaf's score 
#       -   crawl back up the tree and update each parent node 
#               of the explored leaf with the score  
class MCTree:


    def __init__(self,from_fen="",max_game_ply=160,device=torch.device('cuda' if torch.cuda.is_available() else "cpu"),lookup_dict={}):
        

        #Check if a fen is provided, otherwise use the chess starting position
        if from_fen:
            self.board              = chess.Board(fen=from_fen)
        else:
            self.board              = chess.Board()

        #Define the root node (the one that will be evaluatioed) and set 
        #search variables
        self.root:Node              = Node(None,None,.2,0,self.board.turn) 
        self.curdepth               = 0 
        self.max_game_ply           = max_game_ply 

        #Training vars (control exploration of the engine)
        #   set these to 0 to perform an actual evaluation.
        self.dirichlet_a            = .3
        self.dirichlet_e            = .2 

        #Keep track of prior explored nodes
        self.explored_nodes         = lookup_dict
        self.common_nodes           = {}

        #Check override device 
        self.device                 = device

        #Create template in GPU to copy boardstate into
        #   if using cpu, these are not send to GPU and not pinned
        
        # CPU SPECIFIC 
        # self.static_tensorGPU       = torch.empty(size=(1,chess_utils.TENSOR_CHANNELS,8,8),dtype=torch.float16,requires_grad=False,device=self.device)
        # self.static_tensorCPU_P     = torch.empty(1968,dtype=torch.float16,requires_grad=False,device=torch.device('cpu'))#.pin_memory()
        # self.static_tensorCPU_V     = torch.empty(1,dtype=torch.float16,requires_grad=False,device=torch.device('cpu'))#.pin_memory()
        self.static_tensorGPU       = torch.empty(size=(1,chess_utils.TENSOR_CHANNELS,8,8),dtype=torch.float,requires_grad=False,device=self.device)
        self.static_tensorCPU_P     = torch.empty(1968,dtype=torch.float,requires_grad=False,device=torch.device('cpu'))#.pin_memory()
        self.static_tensorCPU_V     = torch.empty(1,dtype=torch.float,requires_grad=False,device=torch.device('cpu'))#.pin_memory()
        #/CPU SPECIFIC
        #Only pin memory if using a CUDA device 
        if not self.device  == torch.device('cpu'):
            self.static_tensorCPU_P.pin_memory()            
            self.static_tensorCPU_V.pin_memory()


    #Loads in the model to be used for evaluation 
    #   Can either be:  - a state_dict of a torch.nn.Module 
    #                   - a string specifying a file containing a state_dict
    #                   - a full model (subclass of torch.nn.Module)
    def load_dict(self,state_dict):
        self.chess_model            = model.ChessModel().to(self.device)


        if isinstance(state_dict,str):
            if not state_dict == '':
                self.chess_model.load_state_dict(torch.load(state_dict))
            print(f"\tloaded model '{state_dict}'")
        elif isinstance(state_dict,OrderedDict):
            self.chess_model.load_state_dict(state_dict)
        elif isinstance(state_dict,torch.nn.Module):
            self.chess_model    = state_dict
        else:
            print(f"found something strage[{type(state_dict)}]")
            exit()
            

        #As of not, not retracing due to memory issues??
        self.chess_model            = self.chess_model.eval().to(self.device)#.half()

        # CPU SPECIFIC
        torch.backends.cudnn.enabled    = True
        #self.chess_model 			= torch.jit.trace(self.chess_model,[torch.randn((1,chess_utils.TENSOR_CHANNELS,8,8),device=self.device,dtype=torch.float16)])
        self.chess_model 			= torch.jit.trace(self.chess_model,[torch.randn((1,chess_utils.TENSOR_CHANNELS,8,8),device=self.device,dtype=torch.float)])
        self.chess_model 			= torch.jit.freeze(self.chess_model)
        #/CPU SPECIFIC
        

    #Perform one exploration down the tree
    #   If 'initial' is set, then add dirichlet noise to 
    #   children of the root node, which adds noise
    #   when we want additional exploration for training 
    #   purposes
    def perform_iter(self,initial=False):
        
        #If initial and root already has pre-populated values, apply dirichelt before descending
        if initial and self.root.children:
            dirichlet           = numpy.random.dirichlet([self.dirichlet_a for _ in self.root.children]) 
            for i,child in enumerate(self.root.children):
                child.prior_p    = (1-self.dirichlet_e)*child.prior_p + dirichlet[i]*self.dirichlet_e
                child.pre_compute()
            add_after       = False
        elif initial and not self.root.children:
            add_after       = True
        else:
            add_after       = False

        #Get to bottom of tree via traversal algorithm 
        curnode             = self.root 
        while not curnode.is_leaf():
            curnode         = curnode.pick_best_child()
            self.board.push(curnode.move)
            self.curdepth   += 1

        #Expand current working node
        self.working_node   = curnode 
        move_outcome        = self.working_node.expand(self.board,
                                                       self.curdepth,
                                                       self.chess_model,
                                                       self.max_game_ply,
                                                       static_gpu=self.static_tensorGPU,
                                                       static_cpu_p=self.static_tensorCPU_P,
                                                       static_cpu_v=self.static_tensorCPU_V,
                                                       lookup_dict=self.explored_nodes,
                                                       common_nodes=self.common_nodes)

        #Recompute prior probabilities for root on initial iteration (add dirichlet)
        if add_after:
            dirichlet           = numpy.random.dirichlet([self.dirichlet_a for _ in self.root.children]) 
            for i,child in enumerate(self.root.children):
                child.prior_p   = (1-self.dirichlet_e)*child.prior_p + dirichlet[i]*self.dirichlet_e
                child.pre_compute()

        #Update score for all nodes of this position
        for node in self.common_nodes[self.board.fen()]:
            node.bubble_up(move_outcome)
        
        #Undo moves 
        for _ in range(self.curdepth):
            self.board.pop()
        self.curdepth = 0
    

    #Call to begin search down a root. The root may already have 
    #   children. Dirichlet noise is always added to root.  
    def evaluate_root(self,n_iters=1000):

        self.common_nodes   = {}
        
        #First iter will add Dirichlet noise to prior Ps of root children
        self.perform_iter(initial=True)
       
        #All resultant iters will not have dirichlet addition
        for _ in range(n_iters):
            self.perform_iter()
                    
        return {c.move:c.n_visits for c in self.root.children}
    

    #This function will add additional compute to the tree. 
    def add_compute(self,n_iters):

        for _ in range(n_iters):
            self.perform_iter()

        return {c.move:c.n_visits for c in self.root.children}
        

    #Applys the given move to the root 
    #   and descends to corresponding node.
    #   Keeps prior calculations down this line  
    def make_move(self,move:chess.Move):

        #Check if move actually in children
        if self.root.n_visits == 0:
            #print(f"Found 0 visit case looking for {move}")
            #print(f"{self.root} had {[move.move for move in self.root.children]}")
            self.perform_iter(False)
            #print(f"{self.root} had {[move.move for move in self.root.children]}")

        
        #Make move 
        self.board.push(move)

        #check gameover 
        if self.board.is_game_over() or self.board.ply() > self.max_game_ply:
            return Node.RESULTS[self.board.result()]
        
        self.chosen_branch  = self.root.traverse_to_child(move)
        del self.root 
        self.root           =  self.chosen_branch
        self.curdepth       = 0 

        return 


    #Displays all nodes in the tree top to bottom. 
    def __repr__(self):

        rows    = {0:[self.root.data_repr()]}

        def traverse(root):
            for c in root.children:
                if c.depth in rows:
                    rows[c.depth].append(c.data_repr())
                else:
                    rows[c.depth] = [c.data_repr()]
                
                if not c.is_leaf():
                    traverse(c)
        
        traverse(self.root)

        append  =    max([sum([len(m) for m in ll]) for ll in rows.values()])

        rows    = [" | ".join(rows[row]) for row in rows]
        for i in range(len(rows)):
            while len(rows[i]) < append:
                rows[i]     = " " + rows[i] + " "

        return "\n\n".join(rows)


    #Remove the memory that was allocated on the CPU, GPU 
    def cleanup(self):

        del self.static_tensorCPU_P
        del self.static_tensorCPU_V
        del self.static_tensorCPU_V




#DEBUG puporses
if __name__ == '__main__':
    t0  = time.time()
    i       = 0
    gameboard   = chess.Board(fen="rnbq1kr1/1ppp1ppp/4p3/P4n2/2PP4/2N2NP1/1P2PPBP/R1BQK2R w KQ - 5 10")
    while True:

        command = "yes"
        #Apply Kyle move 
        move    = chess.Move.from_uci(input("move uci: "))
        gameboard.push(move)
        

        #Prep engine
        mcTree  = MCTree(from_fen=gameboard.fen())
        mcTree.common_nodes = {}
        print(f"\n\nEngines Turn")
        print(mcTree.board)
        print(f"\n\n")
        while "y" in command or "Y" in command:
            i += 1
            mcTree.perform_iter()


            if i % 50000 == 0:
                
                #sample and make move 
                top_move        = None
                top_visits      = -1 
                nodes           = {c.move:c.n_visits for c in mcTree.root.children}
                for move,n_visits in nodes.items():
                    if n_visits > top_visits:
                        top_move    = move 
                        top_visits  = n_visits

                print(f"top move: {top_move}")
                print(f"{nodes}")
                command     = input(f"cont?: ")

        gameboard.push(top_move)
       
        print(f"\n\n\nBoard now:")
        print(gameboard)
        print(f"\nKyles Turn")

    
        # print(mcTree)
        # print(f"\n\n\n")
    #print(f"evals:")
    print({c.move:c.n_visits for c in mcTree.root.children})
    print(f"time in {(time.time()-t0):.2f}s")
    exit()
    
