#Author: Everett Stenberg
#Description:   a class to handle operations of the arbitrary chess player
#               as well as children classes for each type of player (engine,human)



import chess
from mctree import MCTree
import time

class InvalidMove(Exception):

    def __init__(self,message:str):
        super(InvalidMove,self).__init__(message)
        self.message    = message

    def __repr__(self):
        return self.message



class ChessPlayer:

    def __init__(self,board:chess.Board,max_game_ply=200):
        self.board          = board
        self.max_game_ply   = max_game_ply


    def select_move(self,board:chess.Board,time_left_in_sec:int)->chess.Move:
        raise NotImplementedError("select_move not implemented!")
    

    def manual_move(self,move:chess.Move):

        #Check move is valid
        if not move in self.board.generate_legal_moves():
            raise InvalidMove(f"move {move.uci()} not in board {self.board.fen()}")
        else:  
            #Push move 
            self.board.push(move)


    def apply_move(self,move:chess.Move):
        raise NotImplementedError("apply_move not implemented!")



class SteinChessPlayer(ChessPlayer):

    def __init__(self,board:chess.Board,max_game_ply:int,model):
        
        #Call super...
        super(SteinChessPlayer,self).__init__(board,max_game_ply=max_game_ply)
        self.eval_tree      = MCTree(max_game_ply=max_game_ply)
        self.eval_tree.load_dict(model)


    #Will allocate 20% of the time left to the computation
    #   as of now.
    def determine_time_on_calc(self,time_left_in_sec:float) -> float:
        return time_left_in_sec * .2

    
    #To prevent thousands of time calls,
    #   Run 1_000 iters of agorithm then check time 
    def select_move(self,time_left_in_sec: float,override=False) -> chess.Move:
        
        t_begin         = time.time()

        #Init common nodesa and run first iter 
        self.eval_tree.common_nodes = {}
        self.eval_tree.perform_iter(initial=True)

        #Run iters until timeout
        while time.time() - t_begin < self.determine_time_on_calc(time_left_in_sec):

            #Run 1_000 iters 
            self.eval_tree.perform_iter()

        #get resulting visits 

        top_move    = None 
        top_visits  = 0 
        for child in self.eval_tree.root.children:
            if child.n_visits > top_visits:
                top_move    = child.move 
                top_visits  = child.n_visits 
            
        return  top_move


    #Apply the move however needed
    #   will already be applied to board
    def apply_move(self,move:chess.Move):

        #Check that move is in children of root 
        if move in [c.move for c in self.eval_tree.root.children]:

            #Get next root
            next_root                   = self.eval_tree.root.traverse_to_child(move)

            #Delete old tree root 
            del self.eval_tree.root 

            #Set to next root
            self.eval_tree.root         = next_root
            self.eval_tree.root.parent  = None
            self.eval_tree.curdepth     = 0 



class HoomanChessPlayer(ChessPlayer):


    def __init__(self,board:chess.Board,max_game_ply:int):
        
        #Call super...
        super(HoomanChessPlayer,self).__init__(board,max_game_ply=max_game_ply)
        
        
    #Get manual move input 
    def input_move(self,time_left_in_sec: int) -> chess.Move:
        
        #Ask for move from input
        move_uci        = input(f"move (UCI): ")
        try: 
            self.manual_move(move_uci)
        except InvalidMove as e:
            print(f"{e}")
            self.input_move(time_left_in_sec=time_left_in_sec)

        return
    
    
    
    #Does nothing
    def apply_move(self,move:chess.Move):

        #literally nothing needed
        return


if __name__ == "__min__":

    tournament_board    = chess.Board()

    p1                  = HoomanChessPlayer(tournament_board,200)
    p2                  = SteinChessPlayer(tournament_board,200,"chess_model_iter1.dict")

    time_lim            = 10*60 
    increment           = 3 
    time_remain         = time_lim 

    while not tournament_board.is_game_over():

        t_move_start    = time.time()

        move            = p1.input_move(time_left_in_sec=time_remain) if tournament_board.turn else p2.select_move(time_left_in_sec=time_remain)

        #Push to board 
        tournament_board.push(move)

        #Update players 
        p1.apply_move(move)
        p2.apply_move(move)

        time_remain     -= (time.time() - t_move_start)
        time_remain     += increment