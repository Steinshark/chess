import chess
from mctree import MCTree
import time


class ChessPlayer:

    def __init__(self,board:chess.Board,max_game_ply=200):
        self.max_game_ply   = max_game_ply

        pass 

    def select_move(board:chess.Board,time_left_in_sec:int)->chess.Move:
        raise NotImplementedError("select_move not implemented!")
    


class SteinChessPlayer(ChessPlayer):

    def __init__(self,board:chess.Board,max_game_ply:int):
        
        #Call super...
        super(SteinChessPlayer,self).__init__(board,max_game_ply=max_game_ply)
        
        self.eval_tree      = MCTree(max_game_ply=max_game_ply)

    
    #To prevent thousands of time calls,
    #   Run 1_000 iters of agorithm then check time 
    def select_move(self,board: chess.Board, time_left_in_sec: int) -> chess.Move:
        
        t_begin         = time.time()

        #Init common nodesa and run first iter 
        self.eval_tree.common_nodes = {}
        self.eval_tree.perform_iter(initial=True)

        #Run iters until timeout
        while time.time() - t_begin < time_left_in_sec:

            #Run 1_000 iters 
            self.eval_tree.perform_iter()

        #get resulting visits 
        return  {c.move:c.n_visits for c in self.eval_tree.root.children}


class HoomanChessPlayer(ChessPlayer):

    def __init__(self,board:chess.Board,max_game_ply:int):
        
        #Call super...
        super(HoomanChessPlayer,self).__init__(board,max_game_ply=max_game_ply)
        
        

    
    def select_move(board: chess.Board, time_left_in_sec: int) -> chess.Move:
        
        #Build 
    
    
