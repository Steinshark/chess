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
#   Re-configure for GPU evaluation

class MCTree:


    def __init__(self,id:int,max_game_ply=160,n_iters=800,lookup_dict={},device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),alpha=.3,epsilon=.2):


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
        self.start_time             = time.time()
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


    #Performs tree expansions until it finds a node that requires an evaluation 
    #   Recursively calls itself until the tree is pending an evaluation, in which case
    #   class variables are updated to communicate with the handler
    def perform_nongpu_iter(self):
        #Check if time to make a move
        if self.root.n_visits > self.n_iters:
            self.reached_iters          = True 
            self.awaiting_eval          = False
            self.pending_fen            = None 

            for _ in range(self.curdepth):
                self.board.pop()
            self.curdepth               = 0
            return 
            #self.make_move()
            
            
        #Get to bottom of tree via traversal algorithm
        curnode                         = self.root
        mk                              = 0
        while not curnode.is_leaf():
            try:
                curnode         = curnode.pick_best_child()
                self.board.push(curnode.move)
                self.curdepth   += 1
                mk += 1
            except AssertionError:
                print(f"id {self.id} {self.board.fen()} failed to push {curnode.move} after {mk}")
                exit()

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
            

    #Perform a post-GPU call expansion of the nodes. 
    #   The idea is that after a GPU batch is run, the results are placed into the 
    #   lookup dict and a 'perform_expansion' can be done without the next move requiring
    #   an evaluation
    def perform_gpu_expansion(self):

        #Expand, now that we have the value in the dict
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


    #Pick the top move.
    #   argument 'greedy' determines if it will be based on max move count, 
    #   or sampling from the distribution
    def get_top_move(self,greedy=True):

        top_move                    = None 
        top_visits                  = 0 

        for move,visit_count in [(child.move,child.n_visits) for child in self.root.children]:
            if visit_count > top_visits:
                top_move            = move 
                top_visits          = visit_count
        
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
        #print(f"{board_fen}\nroot:{self.root.move}")
        move                        = self.get_top_move(greedy=True)

        #Push move to board
        self.board.push(move)
        print(f"{self.id} -> {move}")

        #check gameover
        if self.board.is_game_over() or self.board.ply() > self.max_game_ply:
            self.game_result        = Node.RESULTS[self.board.result()]
            self.game_datapoints    = [(item[0],item[1],self.game_result) for item in self.game_datapoints]
            self.end_time           = time.time()
            self.run_time           = self.end_time - self.start_time

        
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


    #Load a state dict for the common model
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
    def collect_data(self,n_exps=32_768):

        #Gather n_exps
        while len(self.dataset) < n_exps:
            
            #Get all trees to the point that they require 
            #   a GPU eval
            self.pre_process()
            
            #Pass thorugh to model and redistribute to trees
            with torch.no_grad():
                model_batch                             = chess_utils.batched_fen_to_tensor([game.pending_fen for game in self.active_trees]).float().to(self.device) #CPU SPECIFIC
                priors,evals                            = self.chess_model.forward(model_batch)
                evals                                   = evals.numpy()
                
                for prior_probs,evaluation,tree in zip(priors,evals,self.active_trees):
                    
                    #Correct probs for legal moves 
                    revised_numpy_probs                 = numpy.take(prior_probs.numpy(),[chess_utils.MOVE_TO_I[move] for move in tree.board.generate_legal_moves()])
                    revised_numpy_probs                 = chess_utils.normalize_numpy(revised_numpy_probs,1)
                    
                    #Add to lookup dict 
                    self.lookup_dict[tree.curnode.key]  = (revised_numpy_probs,evaluation[0])

                    #Reset tree fen await 
                    tree.pending_fen                    = None 
                    
                #Perform the expansion relying on GPU 
            [tree.perform_expansion() for tree in self.active_trees]

        
        return self.dataset
   

    #Handle gameover behavior for a given tree
    def check_gameover(self,tree:MCTree,i:int):

        if not tree.game_result is None:

            #print(f"GAMEOVER {tree.id} - replacing tree after {len(tree.game_datapoints)} in {(tree.run_time)/(len(tree.game_datapoints)):.2f}s/move")
                    
            #Add old tree game experiences to datapoints
            self.dataset            += tree.game_datapoints

            #replace tree inplace 
            new_tree                = MCTree(tree.id,self.max_game_ply,self.n_iters,self.lookup_dict,self.device)
            self.active_trees[i]    =    new_tree
            self.active_trees[i].perform_nongpu_iter()

        #Tree will start another search
        else:
            tree.perform_nongpu_iter()


    #This method will get all boards to a state where they require a GPU evaluation
    def pre_process(self):
        
        
        #assume everyone is starting with a 
        while None in [tree.pending_fen for tree in self.active_trees]:
            #Perform tree-search until EITHER:
            #   Node needs a gpu eval (NEEDS TO BE ROLLED BACK)
            #   Node is ready to push moves
            [tree.perform_nongpu_iter() for tree in self.active_trees if tree.pending_fen is None]

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



#DEBUG puporses
if __name__ == '__main__':
    t0 = time.time()
    manager                 = MCTree_Handler(8,max_game_ply=200,n_iters=200)
    data                    = manager.collect_data(n_exps=512)

    print(f"collected {len(data)} - {(time.time()-t0)/len(data):.4f}s/move")


