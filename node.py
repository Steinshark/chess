import chess
import torch
import chess_utils
import numpy 
import sys

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

        #Scores and such 
        self.n_visits           = 0 
        self.init_p             = init_p

        #Game stats from here 
        self.n_wins             = 0 

        self.cumulative_score   = 0


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
        return (self.cumulative_score / (self.n_visits+1)) + -1*self.turn*self.c * self.init_p * (numpy.sqrt(self.parent.n_visits) / (self.n_visits+1))


    def get_score_str(self):
        return f"{self.get_score():.3f}"
    

    def run_rollout(self,board:chess.Board,model:torch.nn.Module,lookup_dict,moves):
        board_fen   = board.fen()
        board_key   = " ".join(board_fen.split(" ")[:4])
        if board_key in lookup_dict:
            return lookup_dict[board_key]
        else:
            with torch.no_grad():
                board_repr              = chess_utils.batched_fen_to_tensor([board_fen]).to(DEVICE).half()
                probs,eval              = model.forward(board_repr)
                revised_probs           = torch.index_select(probs[0],dim=0,index=torch.tensor([chess_utils.MOVE_TO_I[move] for move in moves],device=DEVICE))
                revised_probs2          = chess_utils.normalize_cuda(revised_probs)

                lookup_dict[board_key]  = (revised_probs2.to(device=torch.device('cpu'),non_blocking=True).numpy(),eval[0][0].to(device=torch.device('cpu'),non_blocking=True).numpy())
                return lookup_dict[board_key]
            
            
    def expand(self,board:chess.Board,depth:int,chess_model:torch.nn.Module,max_depth:int,lookup_dict={}):

        #Check end state 
        if board.is_game_over() or board.ply() > max_depth:
            return self.RESULTS[board.result()]

        #Run "rollout"
        moves                       = list(board.generate_legal_moves())
        probabilities,rollout_val   = self.run_rollout(board,chess_model,lookup_dict,moves)

        #Populate children nodes
        with torch.no_grad():
            
            #Add children
            torch.cuda.synchronize(torch.device('cuda'))


            self.children           = [Node(move,self,probabilities[i],depth+1,not board.turn) for i,move in enumerate(moves)]
            return rollout_val
            #revized_probs           = [c.init_p for c in self.children]            
            # #Perform softmax on legal moves only
            # if len(revized_probs) == 1:
            #     self.children[0].init_p = 1 

            #     return rollout_val
            # else:
            #     normalized  = chess_utils.normalize(revized_probs,temperature=1)


            #     for prob,node in zip(normalized,self.children):
            #         node.init_p     = prob

            #     return rollout_val


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