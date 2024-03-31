import torch
import numpy 
import chess 
import time 
import json 


PIECES 	        = {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
CHESSMOVES      = json.loads(open("C:/gitrepos/chess/chessmoves.txt","r").read())
MOVE_TO_I       = {chess.Move.from_uci(move):i for i,move in enumerate(CHESSMOVES)}
I_TO_MOVE       = {i:chess.Move.from_uci(move) for i,move in enumerate(CHESSMOVES)}

def fen_processor(fen:str):
    for i in range(1,9):
        fen 	= fen.replace(str(i),"e"*i)
    
    breakout    = fen.split(" ")

    #Return position, turn, castling rights
    return breakout[0].split("/"), breakout[1], breakout[2]


def fen_to_tensor_lite(fen_info:list):
    position,turn,castling  = fen_info

    this_board              = numpy.zeros(shape=(7+7+1,8,8),dtype=numpy.float32)

    #Place pieces
    for rank_i,rank in enumerate(reversed(position)):
        for file_i,piece in enumerate(rank): 
            if not piece == "e":
                this_board[PIECES[piece],rank_i,file_i]	= 1.  
    
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
    # 	1 for move 
    
    #Clean fens
    fen_info_list   = map(fen_processor,fenlist)

    #get numpy lists 
    numpy_boards    = list(map(fen_to_tensor_lite,fen_info_list))
    numpy_boards    = numpy.asarray(numpy_boards)


    return torch.from_numpy(numpy_boards)


def normalize(X,temperature=1):

    #apply temperature
    X           = [x**(1/temperature) for x in X]

    #apply normalization
    cumsum      = sum(X)
    return [x/cumsum for x in X]



def normalize_cuda(X:torch.Tensor,temperature=1):
    X           = torch.pow(X,1/temperature)
    normalized  = X / torch.sum(X)
    #input(f"normalized is {normalized}")
    return normalized

def temp_scheduler(ply:int):

    #Hold at 1 for first 10 moves 
    if ply < 10:
        return 1
    else:
        return max(1 - .02*(ply - 10),.01)

#Faster all at once?
def tester(bs=64):
    fens    = [item[0] for item in json.loads(open("C:/data/chess/exps/877").read())][:1024]
    t0      = time.time()

    #Try many small
    for fen in fens:
        
        tsor    = fen_to_tensor(fen).to(torch.device('cuda'))
        tsor2   = tsor * -1
    
    print(f"ran 1 in {(time.time()-t0):.2f}s\t to run 1M->({10*1024*(time.time()-t0)/3600}hr)")


    fens    = [item[0] for item in json.loads(open("C:/data/chess/exps/877").read())][:1024]
    t0      = time.time()

    #Try many small
    for i in range(int(1024/bs)):
        fenlist     = fens[i*bs:(i+1)*bs]
        tsor    = batched_fen_to_tensor(fenlist).to(torch.device('cuda'))
        tsor2   = tsor * -1
    
    print(f"ran 2 in {(time.time()-t0):.2f}s\t to run 1M->({10*1024*(time.time()-t0)/3600}hr)")


if __name__ == "__main__":
    from matplotlib import pyplot 
    pyplot.plot([temp_scheduler(x) for x in range(100)])
    pyplot.show()