#Author: Everett Stenberg
#Description:   a collection of functions to aid in chess-related things
#               keeps other files cleaner



import torch
import numpy 
import chess 
import json 


PIECES 	        = {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
CHESSMOVES      = json.loads(open("chessmoves.txt","r").read())
MOVE_TO_I       = {chess.Move.from_uci(move):i for i,move in enumerate(CHESSMOVES)}
I_TO_MOVE       = {i:chess.Move.from_uci(move) for i,move in enumerate(CHESSMOVES)}

TENSOR_CHANNELS = 19

def fen_processor(fen:str):
    for i in range(1,9):
        fen 	= fen.replace(str(i),"e"*i)
    
    breakout    = fen.split(" ")

    #Return position, turn, castling rights
    return breakout[0].split("/"), breakout[1], breakout[2]


#Return a shape 19 tensor
def fen_to_tensor_lite(fen_info:list):
    position,turn,castling  = fen_info

    this_board              = numpy.zeros(shape=(7+7+4+1,8,8),dtype=numpy.float32)

    #Place pieces
    for rank_i,rank in enumerate(reversed(position)):
        for file_i,piece in enumerate(rank): 
            if not piece == "e":
                this_board[PIECES[piece],rank_i,file_i]	= 1.  
    
    #Place castling 
    this_board[-5,:,:]      = numpy.ones(shape=(8,8)) * 1. if "K" in castling else 0.            
    this_board[-4,:,:]      = numpy.ones(shape=(8,8)) * 1. if "Q" in castling else 0.            
    this_board[-3,:,:]      = numpy.ones(shape=(8,8)) * 1. if "k" in castling else 0.            
    this_board[-2,:,:]      = numpy.ones(shape=(8,8)) * 1. if "q" in castling else 0.            

    #Place turn 
    this_board[-1,:,:]      = numpy.ones(shape=(8,8)) * 1. if turn == "w" else -1.

    return this_board

    
def fen_to_tensor(fen):

    #Encoding will be an 15x8x8 tensor 
    #	7 for whilte, 7 for black 
    # 	1 for move 
    #t0 = time.time()
    board_tensor 	= numpy.zeros(shape=(7+7+1,8,8),dtype=numpy.int8)
    piece_indx 	    = {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
    
    #Go through FEN and fill pieces
    for i in range(1,9):
        fen 	= fen.replace(str(i),"e"*i)

    position	= fen.split(" ")[0].split("/")
    turn 		= fen.split(" ")[1]
    castling 	= fen.split(" ")[2]
    
    #Place pieces
    for rank_i,rank in enumerate(reversed(position)):
        for file_i,piece in enumerate(rank): 
            if not piece == "e":
                board_tensor[piece_indx[piece],rank_i,file_i]	= 1.  

    #Place turn 
    board_tensor[-1,:,:]    = numpy.ones(shape=(8,8),dtype=torch.int8) * 1. if turn == "w" else -1.

    return torch.from_numpy(board_tensor)


def batched_fen_to_tensor(fenlist):

    #Encoding will be an bsx15x8x8 tensor 
    #	7 for white, 7 for black 
    #   4 for castling
    # 	1 for move 
    
    #Clean fens
    fen_info_list   = map(fen_processor,fenlist)

    #get numpy lists 
    numpy_boards    = list(map(fen_to_tensor_lite,fen_info_list))
    numpy_boards    = numpy.asarray(numpy_boards,dtype=numpy.float16)


    return torch.from_numpy(numpy_boards)


def normalize(X,temperature=1):

    #apply temperature
    X           = [x**(1/temperature) for x in X]

    #apply normalization
    cumsum      = sum(X)
    return [x/cumsum for x in X]


def normalize_numpy(X,temperature=1):
    X           = numpy.power(X,1/temperature)

    return X / numpy.sum(X)



def temp_scheduler(ply:int):

    #Hold at 1 for first 10 moves 
    if ply < 10:
        return 1
    else:
        return max(1 - .02*(ply - 10),.01)


def movecount_to_prob(movecount,to_tensor=True):

    #Prep zero vector
    probabilities   = [0 for _ in CHESSMOVES]

    #Fill in move counts
    for move,count in movecount.items():
        move_i  = MOVE_TO_I[chess.Move.from_uci(move)]
        probabilities[move_i]   = count
    
    #Return normalized 
    norm    = normalize(probabilities)

    return torch.tensor(norm)




def clean_eval(evaluation):
    
    if evaluation > 0:
        return min(1500,evaluation) / 1500
    else:
        return max(-1500,evaluation) / 1500


if __name__ == "__main__":
    from matplotlib import pyplot 
    pyplot.plot([temp_scheduler(x) for x in range(100)])
    pyplot.show()