import chess
from mctree import MCTree

class ChessPlayer:

    def __init__(self):

        pass 

    def select_move(board:chess.Board,time_left_in_sec:int)->chess.Move:
        raise NotImplementedError("select_move not implemented!")
    


class SteinChessPlayer(ChessPlayer):

    def __init__(self):
        
        self.eval_tree      = MCTree(max_game_ply=500)

    
    def select_move(board: chess.Board, time_left_in_sec: int) -> chess.Move:
        
        #Build 
    
    
