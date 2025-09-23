#Author: Everett Stenberg
#Description:   a collection of functions to aid in chess-related things
#               keeps other files cleaner



import torch
import numpy 
import bulletchess 
import json 
import os
import settings 
import math

PIECES 	        = {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
CHESSMOVES      = json.loads(open("chessmoves.txt","r").read())
MOVE_TO_I       = {bulletchess.Move.from_uci(move):i for i,move in enumerate(CHESSMOVES)}
I_TO_MOVE       = {i:bulletchess.Move.from_uci(move) for i,move in enumerate(CHESSMOVES)}


#Add Color to terminal output
class Color:
    os.system("")
    blue    = '\033[94m'
    tan     = '\033[93m'
    green   = '\033[92m'
    red     = '\033[91m'
    bold    = '\033[1m'
    end     = '\033[0m'    


#Process fen strings to replace numbers with "e" (empty - i.e. no piece)
def fen_processor(fen:str):
    for i in range(1,9):
        fen 	= fen.replace(str(i),"e"*i)
    
    breakout    = fen.split(" ")

    #Return position, turn, castling rights
    return breakout[0].split("/"), breakout[1], breakout[2]


#Return a shape 19 tensor
def fen_to_tensor_lite(fen_info:list):
    position,turn,castling  = fen_info

    code                    = [0,0,0,0]

    this_board              = numpy.zeros(shape=(8,8),dtype=numpy.int8)

    #Place pieces
    for rank_i,rank in enumerate(reversed(position)):
        for file_i,piece in enumerate(rank): 
            if not piece == "e":
                this_board[rank_i,file_i]	= PIECES[piece]
    
    #Place castling '
    # todo: ADD BACK CASTLING
    code[0]      = 1 if "K" in castling else 0            
    code[1]      = 1 if "Q" in castling else 0            
    code[2]      = 1 if "k" in castling else 0            
    code[3]      = 1 if "q" in castling else 0            
    castling_code           = code[0]*(2**3) + code[1]*(2**2) + code[2]*(2**1) + code[3]*(2**0)

    #Place turn 
    turn                    = 1 if turn == 'w' else 0

    return this_board, turn, castling_code


#Process a batch of tensors
def batched_fen_to_tensor(fenlist) -> torch.Tensor:

    #Encoding will be an bsx15x8x8 tensor 
    #	6 for white pieces, 6 for black pieces {0,1} 
    #   4 for castling rights {0,1}
    # 	1 for move {-1,1}
    
    #Clean fens
    fen_info_list   = map(fen_processor,fenlist)

    

    #get numpy lists 
    board_reprs     = list(map(fen_to_tensor_lite,fen_info_list))
    #numpy_boards    = numpy.asarray(numpy_boards,dtype=numpy.float32)

    boards          = []
    turns           = []
    castling_rights = [] 

    for item in board_reprs:
        boards.append(item[0])
        turns.append(item[1])
        castling_rights.append(item[2])

    #Make boards 
    return numpy.asarray(boards), turns, castling_rights

    return torch.from_numpy(numpy_boards).type(settings.DTYPE)


#Normalize a list of values 
def normalize(X,temperature=1):

    #apply temperature
    X           = [x**(1/temperature) for x in X]

    #apply normalization
    cumsum      = sum(X)
    return [x/cumsum for x in X]


#Normalize a list of values assuming X is a numpy array
def normalize_numpy(X:numpy.ndarray,temperature=1):
    X           = numpy.power(X,1/temperature)
    #print(X,"\n\n\n")
    return X / numpy.sum(X)


#Normalize a list of values assuming X is a torch tensor 
def normalize_torch(X:torch.Tensor,temperature=1)->torch.Tensor:
    return torch.nn.functional.normalize(X,p=1/temperature,dim=0)

#Scheduler for the temperature parameter
def temp_scheduler(ply:int):

    #Hold at 1 for first 10 moves 
    if ply < 10:
        return 1
    else:
        return max(1 - .02*(ply - 10),.01)

#Convert a dict of move_uci->visit count to a normalized list 
#   in the same order that the moves were in the dict
def movecount_to_prob(movecount):

    #Prep zero vector
    probabilities   = [0. for _ in CHESSMOVES]

    #Fill in move counts
    for move,count in movecount.items():
        move_i  = MOVE_TO_I[bulletchess.Move.from_uci(move)]
        probabilities[move_i]   += count
    
    #Return normalized 
    norm    = normalize(probabilities)


    return torch.tensor(norm,dtype=torch.float)


#Convert an evaluation from the engine (-1_000_000,1_000_000) -> (-1,1)
def clean_eval(evaluation):
    
    if evaluation > 0:
        return min(1500,evaluation) / 1500
    else:
        return max(-1500,evaluation) / 1500


#Generate the key for a given chess Board 
def generate_board_key(board:bulletchess.Board):
    return " ".join(board.fen().split(" ")[:4])

#   General function to interpolate an array
#   To a smaller size  
def reduce_arr(arr,newlen):

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)

    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]


if __name__ == "__main__":
    b = bulletchess.Board()
    b.apply(bulletchess.Board.from_uci("e2e4"))
    b.apply(bulletchess.Board.from_uci("e7e5"))

    out2 = batched_fen_to_tensor([b.fen()])


    print(f"out2: {out2[0][16]}")
    # from matplotlib import pyplot as plt 
    # norm_vect   = normalize(ex,temperature=1)
    # plt.bar([1,3,5,7],height=norm_vect)
    # norm_vect   = normalize(ex,temperature=.25)
    # plt.bar([2,4,6,8],height=norm_vect)
    # plt.show()
    # print(norm_vect)