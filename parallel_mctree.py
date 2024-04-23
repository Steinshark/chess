#Author: Everett Stenberg
#Description:   The class that acts as the game engine (parallel version). MCTree can conduct and iterative search
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



#TODO 
#   Add dirichlet noise to the root node
#   Re-configure for GPU evaluation

class MCTree:


    def __init__(self,id:int,max_game_ply=160,n_iters=800,lookup_dict={},device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),alpha=.3,epsilon=.25):


        #Create game board
        self.board              = chess.Board()

        #Define the root node (the one that will be evaluatioed) and set
        #search variables
        self.root:Node              = Node(None,None,0,0,self.board.turn)
        self.curdepth               = 0
        self.max_game_ply           = max_game_ply
        self.n_iters                = n_iters

        #Training vars (control exploration of the engine)
        #   set these to 0 to perform an actual evaluation.
        self.dirichlet_a            = alpha
        self.dirichlet_e            = epsilon

        #Keep track of prior explored nodes
        self.lookup_dict            = lookup_dict
        self.common_nodes           = {}
        self.gpu_blocking           = False
        
        #Multithread vars 
        self.pending_fen            = None
        self.reached_iters          = False
        self.game_result            = None
        self.awaiting_eval          = False
        self.game_datapoints        = []
        self.id                     = id

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


    def perform_nongpu_iter(self):

        #Check if time to make a move
        if self.root.n_visits > self.n_iters:
            self.reached_iters          = True 
            return 
            #self.make_move()
            
            
        #Get to bottom of tree via traversal algorithm
        curnode                 = self.root
        while not curnode.is_leaf():
            curnode         = curnode.pick_best_child()
            self.board.push(curnode.move)
            self.curdepth   += 1

        #Node key
        curnode.key                     = " ".join(self.board.fen().split(" ")[:4])

        #Curnode now needs to be expanded
        if curnode.key in self.lookup_dict:
            self.curnode:Node           = curnode
            self.perform_expansion()
            self.perform_nongpu_iter()
        else:
            self.curnode:Node           = curnode
            self.pending_fen            = self.board.fen()
            self.awaiting_eval          = True
            

    #The evaluation will be in the lookup_dict now 
    def perform_gpu_expansion(self):
        self.perform_expansion()

        #Reset gpu-pending vars 
        self.awaiting_eval              = False 
        self.pending_fen                = None


    #Perform an expansion of a leaf node
    def perform_expansion(self):

        node                            = self.curnode
        #Get pre-computed values
        revised_probs,evaluation        = self.lookup_dict[node.key]

        node.children                   = [Node(move,node,revised_probs[i],node.depth+1,not self.board.turn) for i,move in enumerate(self.board.generate_legal_moves())]
        

        #If this is the root, then apply dirichlet 
        if node == self.root:
            self.apply_dirichlet()
    
        #Check in common nodes
        if node.key in self.common_nodes:
            self.common_nodes[node.key].add(node)
        else:
            self.common_nodes[node.key] = set([node])
        #Bubble-up for all common nodes
        for common_node in self.common_nodes[node.key]:
            common_node.bubble_up(evaluation)
        
        #Unpop gameboard 
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


    #Pick the top move
    def get_top_move(self,greedy=True):

        top_move                    = None 
        top_visits                  = 0 

        for move,visit_count in [(child.move,child.n_visits) for child in self.root.children]:
            #print(f"{move}->{visit_count}")
            if visit_count > top_visits:
                top_move            = move 
                top_visits          = visit_count
        
                #print(f"best move is {top_move}")
        #print(f"id {self.id}-> {top_move}")
        #input(f"id {self.id}-> {[f'{node.get_score().item():.3f}' for node in self.root.children]}")
        return top_move
    

    #Applys the given move to the root
    #   and descends to corresponding node.
    #   Keeps prior calculations down this line
    #   Dirichelt noise is added here because the next 
    #   Root will need it for the exploration
    def make_move(self):

        #Save experience
        board_fen                   = self.board.fen()
        post_probs                  = {node.move.uci() for node in self.root.children}
        position_eval               = 0
        datapoint                   = (board_fen,post_probs,position_eval)
        self.game_datapoints.append(datapoint)

        #Get move from probabililtes 
        move                        = self.get_top_move(greedy=True)

        #Push move to board
        self.board.push(move)

        #check gameover
        if self.board.is_game_over() or self.board.ply() > self.max_game_ply:
            self.game_result        = Node.RESULTS[self.board.result()]
            self.game_datapoints    = [(item[0],item[1],self.game_result) for item in self.game_datapoints]
        
        else:
            #update tree
            self.chosen_branch  = self.root.traverse_to_child(move)
            del self.root
            self.root           = self.chosen_branch
            self.root.parent    = None
            self.curdepth       = 0
            self.reached_iters  = False
            self.apply_dirichlet()

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


    #Applys dirichlet noise to a root, presuming all children have been 
    #   explored at least once (i.e. all children are here)
    def apply_dirichlet(self)-> None:

        #Create numpy noise 
        dirichlet                           = numpy.random.dirichlet([self.dirichlet_a for _ in self.root.children])

        #Inplace replacement of child priors
        for i,child in enumerate(self.root.children):
            child.prior_p                   = (dirichlet[i] * self.dirichlet_e) + ((1-self.dirichlet_e) * child.prior_p)
            child.pre_compute()
        
        return


#Creates an instance of a Monte-Carlo style Tree
#   to develop an evaluation of a given position, the tree
#   functions as follows:
#       - start at the root position (the one to be evaluated)
#       - for n_iters:
#       -   traverse the tree to the next best leaf
#       -   expand the leaf and determine leaf's score
#       -   crawl back up the tree and update each parent node
#               of the explored leaf with the score
class MCTree_Handler:

    def __init__(self,n_parallel=8,device=torch.device('cuda' if torch.cuda.is_available() else "cpu"),max_game_ply=160,n_iters=800):

        self.lookup_dict            = {}
        self.active_trees          = [MCTree(max_game_ply=max_game_ply,lookup_dict=self.lookup_dict,n_iters=n_iters,id=tid) for tid in range(n_parallel)]

        self.device                 = device
        self.chess_model            = model.ChessModel(19,16).to(self.device).eval()

        self.dirichlet_a            = .3
        self.dirichlet_e            = .2

        self.dataset                = []

        self.max_game_ply           = max_game_ply
        self.n_iters                = n_iters



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


    def load_dict(self,state_dict):
        self.chess_model            = model.ChessModel(chess_utils.TENSOR_CHANNELS,24).to(self.device)


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


    #Performs an algorithm iteration filling the gpu to its batchsize
    def collect_data(self):

        #Wait until everyone needs a gpu
        while False in [tree.gpu_blocking for tree in self.active_trees]:
            
            #Get all trees to the point that they require 
            #   a GPU eval
            self.pre_process()
            
            #Pass thorugh to model and redistribute to trees
            with torch.no_grad():
                model_batch                             = chess_utils.batched_fen_to_tensor([game.pending_fen for game in self.active_trees]).float().to(self.device) #CPU SPECIFIC
                priors,evals                            = self.chess_model.forward(model_batch)
                
                for prior_probs,evaluation,tree in zip(priors,evals,self.active_trees):
                    
                    #Correct probs for legal moves 
                    revised_numpy_probs                 = numpy.take(prior_probs.numpy(),[chess_utils.MOVE_TO_I[move] for move in tree.board.generate_legal_moves()])
                    revised_numpy_probs                 = chess_utils.normalize_numpy(revised_numpy_probs,1)

                    #Add to lookup dict 
                    self.lookup_dict[tree.curnode.key]  = (revised_numpy_probs,evaluation)

                    #Reset tree fen await 
                    tree.pending_fen                    = None 
                    
                #Perform the expansion relying on GPU 
            [tree.perform_expansion() for tree in self.active_trees]

            #print(f"dataset size is {len(self.dataset)}")
   
   
    def check_gameover(self,tree:MCTree,i:int):

        if not tree.game_result is None:

            print(f"GAMEOVER - replacing tree after {len(tree.game_datapoints)}")
                    
            #Add old tree game experiences to datapoints
            self.dataset            += tree.game_datapoints

            #replace tree inplace 
            new_tree                = MCTree(tree.id,self.max_game_ply,self.n_iters,self.lookup_dict,self.device)

            self.active_trees[i]    =    new_tree
            self.active_trees[i].perform_nongpu_iter()
            input(f"replaced tree")

        #Tree will start another search
        else:
            tree.perform_nongpu_iter()


    #This method will get all boards to a state where they require a GPU evaluation
    def pre_process(self):

        #assume everyone is starting with a 
        while None in [tree.pending_fen for tree in self.active_trees]:

            #Perform tree-search until EITHER:
            #   Node needs a gpu eval 
            #   Node is ready to push moves
            [tree.perform_nongpu_iter() for tree in self.active_trees]

            #Check why we got a None value 
            for i,tree in enumerate(self.active_trees):
                
                #If reached iters, do a move and all that 
                if tree.reached_iters:
                    
                    #Will make a move 
                    tree.make_move()

                    #Handles all outcomes of the move:
                    #   Gameover (Create new)
                    #   New tree (Get to gpu-eval needed)
                    self.check_gameover(tree,i)
        
        #
        # print(f"{[tree.pending_fen for tree in self.active_trees]}")
        #
        # input(f"all ready for GPU COMPUTE")


    #This method gets all trees to a state ready to request a GPU evaluation of the 
    #   position
    def push_moves_on_boards(self):


        for i,tree in enumerate(self.active_trees):

            #Tree is ready to make a move
            if tree.reached_iters:
                tree.make_move()
                self.check_gameover(tree)

            else:
                pass



#DEBUG puporses
if __name__ == '__main__':
    manager                 = MCTree_Handler(4,max_game_ply=200,n_iters=200)
    manager.collect_data()