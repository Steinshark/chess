from node import Node 
import chess 
import time 
import model 
import torch
import value_trainer
import numpy
import sys
import chess_utils


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

class MCTree:

    def __init__(self,from_fen="",max_game_ply=160,verbose=False):
        if from_fen:
            self.board              = chess.Board(fen=from_fen)
        else:
            self.board              = chess.Board()
        self.root:Node              = Node(None,None,.2,0,self.board.turn) 
        self.curdepth               = 0 
        self.max_game_ply               = max_game_ply 

        #Training vars
        self.dirichlet_a            = .3
        self.dirichlet_e            = .2

        #Load and prep model
        self.chess_model            = model.ChessModel2(chess_utils.TENSOR_CHANNELS,24).to(DEVICE).eval().half()
        self.chess_model 			= torch.jit.trace(self.chess_model,[torch.randn((1,chess_utils.TENSOR_CHANNELS,8,8),device=DEVICE,dtype=torch.float16)])
        self.chess_model 			= torch.jit.freeze(self.chess_model)

        #Keep track of prior explored nodes
        self.explored_nodes         = dict()

        #Create template in GPU to copy boardstate into
        self.static_tensorGPU       = torch.empty(size=(1,chess_utils.TENSOR_CHANNELS,8,8),dtype=torch.float16,requires_grad=False,device=DEVICE)
        self.static_tensorCPU_P     = torch.empty(1968,dtype=torch.float16,requires_grad=False,device=torch.device('cpu')).pin_memory()
        self.static_tensorCPU_V     = torch.empty(1,dtype=torch.float16,requires_grad=False,device=torch.device('cpu')).pin_memory()


    def perform_iter(self,initial=False):
        
        #If initial and root already has pre-populated values, apply dirichelt before descending
        if initial and self.root.children:
            dirichlet           = numpy.random.dirichlet([self.dirichlet_a for _ in self.root.children]) 
            for i,child in enumerate(self.root.children):
                child.init_p    = (1-self.dirichlet_e)*child.init_p + dirichlet[i]*self.dirichlet_e
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
                child.init_p    = (1-self.dirichlet_e)*child.init_p + dirichlet[i]*self.dirichlet_e
                child.pre_compute()

        #Update score for all nodes of this position
        for node in self.common_nodes[self.board.fen()]:
            node.bubble_up(move_outcome)
        
        #Undo moves 
        for _ in range(self.curdepth):
            self.board.pop()
        self.curdepth = 0
    

    def calc_next_move(self,n_iters=1000):

        self.common_nodes   = {}
        
        #First iter will add Dirichlet noise to prior Ps of root 
        self.perform_iter(initial=True)
       
        #All resultant iters will not have dirichlet addition
        for _ in range(n_iters):
            self.perform_iter()
            
        
        return_dict         = {c.move:c.n_visits for c in self.root.children}
        return return_dict
    

    def make_move(self,move:chess.Move):
        
        #Make move 
        self.board.push(move)

        #check gameover 
        if self.board.is_game_over() or self.board.ply() > self.max_game_ply:
            return Node.RESULTS[self.board.result()]
        
        #update tree 
        self.chosen_branch  = self.root.traverse_to_child(move)
        del self.root 
        self.root           =  self.chosen_branch
        self.curdepth       = 0 

        return 


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



if __name__ == '__main__':
    mcTree  = MCTree(from_fen="rnbqkbnr/2ppppp1/pp5p/8/2B5/4PQ2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    #print(f"root: {mcTree.root.is_leaf()}")
    #print(f"{mcTree.board}")
    t0  = time.time()
    for _ in range(1000):
        mcTree.perform_iter()

    
        # print(mcTree)
        # print(f"\n\n\n")
    #print(f"evals:")
    print({c.move:c.n_visits for c in mcTree.root.children})
    print(f"time in {(time.time()-t0):.2f}s")
    exit()
    
