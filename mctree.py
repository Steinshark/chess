from node import Node 
import chess 
import time 
import model 
import torch
import value_trainer


class MCTree:

    def __init__(self,from_fen="",max_game=20):
        if from_fen:
            self.board          = chess.Board(fen=from_fen)
        else:
            self.board              = chess.Board()
        self.root:Node          = Node(None,None,.2,0,self.board.turn) 
        self.curdepth           = 0 
        self.max_game           = max_game 

        self.chess_model        = model.ChessModel(15).cuda()
        try:
            self.chess_model.load_state_dict(torch.load("chessmodelparams.pt"))
            print(f"\tloaded model")
        except FileNotFoundError:
            print(f"\tTraining V model")
            value_trainer.train_v_dict(self.chess_model)
        self.chess_model.eval().half()

        self.explored_nodes     = dict()


    def perform_iter(self):

        #Get to bottom of tree via traversal algorithm 
        curnode             = self.root 
        while not curnode.is_leaf():
            curnode         = curnode.pick_best_child()
            self.board.push(curnode.move)
            self.curdepth   += 1

        self.working_node   = curnode 
        move_outcome        = self.working_node.expand(self.board,self.curdepth,self.chess_model,self.max_game,lookup_dict=self.explored_nodes)

        #Update score for tree
        self.working_node.bubble_up(move_outcome)
        
        #Undo moves 
        for _ in range(self.curdepth):
            self.board.pop()
        self.curdepth = 0
    

    def calc_next_move(self,n_iters=1200):
        while self.root.n_visits < n_iters:
            self.perform_iter()
        
        return_dict     = {c.move:c.n_visits for c in self.root.children}
        return return_dict
    

    def make_move(self,move:chess.Move):
        
        #Make move 
        self.board.push(move)

        #check gameover 
        if self.board.is_game_over() or self.board.ply() > self.max_game:
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
    mcTree  = MCTree()
    #print(f"root: {mcTree.root.is_leaf()}")
    #print(f"{mcTree.board}")
    t0  = time.time()
    for _ in range(1200):
        mcTree.perform_iter()

    
        # print(mcTree)
        # print(f"\n\n\n")
    #print(f"evals:")
    #print({str(move):eval for move,eval in mcTree.get_eval().items()})
    print(f"time in {(time.time()-t0):.2f}s")
    exit()
    
    
    
    
    print(f"root: {mcTree.root.is_leaf()}")

    print("next leaf is at ",mcTree.retrieve_next_move(), " next turn is ", mcTree.retrieve_next_move().turn)
    print(f"root is turn {mcTree.root.turn},best child is {mcTree.root.pick_best_child()}")
    print(f"child turn is {mcTree.root.pick_best_child().turn}")