#Author: Everett Stenberg
#Description:   The class that acts as the game engine. MCTree can conduct and iterative search 
#               of the current chess position



from node import Node 
import bulletchess 
import time 
import model 
import torch
import utilities
import numpy
from collections import OrderedDict
import settings


#Creates an instance of a Monte-Carlo searchable Tree
#   to develop an evaluation of a given position, the tree
#   functions as follows:
#       - start at the root position (the one to be evaluated)
#       - for n_iters:
#       -   traverse the tree to the next best leaf
#       -   expand the leaf and determine leaf's score 
#       -   crawl back up the tree and update each parent node 
#               of the explored leaf with the score  
class MCTree:


    def __init__(self,from_fen="",max_game_ply=settings.MAX_PLY,device=torch.device('cuda' if torch.cuda.is_available() else "cpu"),lookup_dict={}):
        

        #Check if a fen is provided, otherwise use the chess starting position
        if from_fen:
            self.board              = bulletchess.Board.from_fen(from_fen)
        else:
            self.board              = bulletchess.Board()

        #Define the root node (the one that will be evaluatioed) and set 
        #search variables
        self.root:Node              = Node(None,None,0,0,self.board.turn) 
        self.curdepth               = 0 
        self.max_game_ply           = max_game_ply 

        #Training vars (control exploration of the engine)
        #   set these to 0 to perform an actual evaluation.
        self.dirichlet_a            = settings.DIR_A
        self.dirichlet_e            = settings.DIR_E

        #Keep track of prior explored nodes
        self.explored_nodes         = lookup_dict
        self.common_nodes           = {}

        #Check override device 
        self.device                 = device

        #Create static memory locations on GPU and CPU to reduce memory allocations    
        #self.static_tensorGPU       = torch.empty(size=settings.JIT_SHAPE,dtype=torch.float,requires_grad=False,device=self.device)
        self.static_tensorGPU       = torch.empty(size=settings.JIT_SHAPE,dtype=torch.float,requires_grad=False,device=self.device)
        self.static_tensorCPU_P     = torch.empty(settings.N_CHESS_MOVES,dtype=torch.float,requires_grad=False,device=torch.device('cpu'))#.pin_memory()
        self.static_tensorCPU_V     = torch.empty(1,dtype=torch.float,requires_grad=False,device=torch.device('cpu'))#.pin_memory()



    #Loads in the model to be used for evaluation 
    #   Can either be:  - a state_dict of a torch.nn.Module 
    #                   - a string specifying a file containing a state_dict
    #                   - a full model (subclass of torch.nn.Module)
    def load_dict(self,state_dict):


        if isinstance(state_dict,str):
            self.chess_model            = model.ChessModel(**settings.MODEL_KWARGS).to(self.device)
            if not state_dict == '':
                self.chess_model.load_state_dict(torch.load(state_dict))

        elif isinstance(state_dict,OrderedDict):
            self.chess_model            = model.ChessModel(**settings.MODEL_KWARGS).to(self.device)
            self.chess_model.load_state_dict(state_dict)

        elif isinstance(state_dict,torch.nn.Module):
            self.chess_model                = state_dict

        else:
            print(f"{utilities.Color.red}found something strage[{type(state_dict)}]{utilities.Color.end}")
            exit()
            

        #As of not, not retracing due to memory issues??
        self.chess_model                    = self.chess_model.eval().to(self.device).type(settings.DTYPE)

        for parameter in self.chess_model.parameters():
            parameter.requires_grad_(False)

        torch.backends.cudnn.enabled        = True
        example_input                       = torch.randint(0,13+16+2,size=(1,66),device=self.device,dtype=torch.long)
        #self.chess_model                    = torch.jit.trace(self.chess_model,example_input)
        #self.chess_model                    = torch.jit.freeze(self.chess_model)
        #self.chess_model                    = torch.jit.optimize_for_inference(self.chess_model)

    #Perform one exploration down the tree
    #   If 'initial' is set, then add dirichlet noise to 
    #   children of the root node, which adds noise
    #   when we want additional exploration for training 
    #   purposes
    def perform_iter(self,initial=False,debug=False):
        

        if debug:
            input(f"start with {self.root.children}")

        #We no longer apply dirichlet - SIKE
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
            self.board.apply(curnode.move)
            self.curdepth   += 1
            #print(f"turn now {self.board.turn}")

        if debug:
            input(f"Down to {curnode}")

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
            self.board.undo()

        self.curdepth = 0
    

    #Perform an exploration down a given child node
    #   Used in conjunction with the Gumbel noise 
    #   optimization.
    def iter_child(self,node:Node):
        

        #Get to bottom of tree via traversal algorithm 
        curnode             = node 
        self.board.apply(curnode.move)
        self.curdepth       = 1 

        while not curnode.is_leaf():
            curnode         = curnode.pick_best_child()
            self.board.apply(curnode.move)
            self.curdepth   += 1
            print(f"down in depth")
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


        #Update score for all nodes of this position
        for node in self.common_nodes[self.board.fen()]:
            node.bubble_up(move_outcome)
        
        #Undo moves 
        for _ in range(self.curdepth):
            self.board.undo()

        self.curdepth = 0
    

    #Call to begin search down a root. The root may already have 
    #   children. Dirichlet noise is always added to root.  
    def evaluate_root(self,n_iters=1000):

        self.common_nodes:dict[str,Node]   = {}
        
        #First iter will add Dirichlet noise to prior Ps of root children
        self.perform_iter(initial=True)
       
        #All resultant iters will not have dirichlet addition
        for _ in range(n_iters):
            self.perform_iter()
                    
        return {str(c.move):c.n_visits for c in self.root.children}
    

    #Begins the search down the root using Gumbel noise optimization instead of 
    #   adding Dirichlet noise.
    def evaluate_root_with_gumbel(self,n_iters=1000,k=50):
        
        self.common_nodes       = {}

        input(f"board is {self.board}")
        #ROllout once   
        if not self.root.children:
            self.iter_child(self.root)

        print(f"board is now {self.board}")

        #Build the gumbel priors 
        probs                   = numpy.array([c.prior_p for c in self.root.children])
        gumbels                 = -numpy.log(-numpy.log(numpy.random.rand(len(probs))))
        scores                  = (probs + 1e-5) + gumbels 

        explore_nodes           = numpy.argsort(scores)[-k:]
        #Rollout only these half 
        for node_i in explore_nodes:
            child               = self.root.children[node_i]
            self.iter_child(child)
        
        for _ in range(n_iters):
            self.perform_iter()


    #This function will add additional compute to the tree. 
    def add_compute(self,n_iters):

        for _ in range(n_iters):
            self.perform_iter()

        return {c.move:c.n_visits for c in self.root.children}
        

    #Applys the given move to the root 
    #   and descends to corresponding node.
    #   Keeps prior calculations down this line  
    def make_move(self,move:bulletchess.Move):

        #Check if move actually in children
        if self.root.n_visits == 0:
            #print(f"Found 0 visit case looking for {move}")
            #print(f"{self.root} had {[move.move for move in self.root.children]}")
            self.perform_iter(False)
            #print(f"{self.root} had {[move.move for move in self.root.children]}")

        
        #Make move 
        self.board.apply(move)

        #check gameover 
        gameover = self.game_over()
        if gameover is not None:
            return gameover
        
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


    #Checks if the game is over for the given board.
    #   return None if not
    def game_over(self):
        if self.board in bulletchess.FORCED_DRAW or self.board.fullmove_number > self.max_game_ply:
            return 0 
        
        elif self.board in bulletchess.CHECKMATE:
            return 1 if self.board.turn == bulletchess.BLACK else -1
        
        else:
            return None 


#DEBUG purposes
if __name__ == '__main__':
    import random 
    from model import ChessTransformer
    dummy_model = ChessTransformer(emb_dim=settings.N_EMBED,num_layers=12,num_heads=8)
    dummy_model.eval()
    for param in dummy_model.parameters():
        param.requires_grad_(False)

    dummy_model.bfloat16()

    tree    = MCTree()
    #tree = mctree.MCTree(from_fen='rnbqkbnr/ppppp2p/8/5pp1/4P3/P7/1PPP1PPP/RNBQKBNR w KQkq g6 0 3')
    tree.load_dict(dummy_model)

    n_games         = 2
    collections     = []
    t0              = time.time()
    n_moves         = 0 
    for _ in range(n_games):
        game_experiences    = [] 

        while tree.game_over() is None:
            move_counts     = tree.evaluate_root(200)
            
            #input(move_counts)
            move_probs      = numpy.asarray(list(move_counts.values()))
            next_move       = random.choices(list(move_counts.keys()),weights=move_probs,k=1)[0]

            
            game_experiences.append([tree.board.fen(),move_counts,next_move,None])
            tree.make_move(bulletchess.Move.from_uci(next_move))   