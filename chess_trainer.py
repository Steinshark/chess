import chess_utils 
import random 
import chess
from mctree import MCTree
import os 
import json
import time
import torch
import sys 


DATAPATH        = "data/"

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
print(f"\tset device to {DEVICE}")


def generate_data(n_games,n_iters,uid,offset,max_game_ply=200):
    data        = []
    print(f"\tsaving to {os.path.join(DATAPATH,uid+'_'+str(offset))}")
    n_moves     = 0
    for game_i in range(n_games):

        tree                = MCTree(max_game_ply=max_game_ply)
        result              = None 
        game_experiences    = []
        while result is None:

            #Run search
            move_probs      = tree.calc_next_move(n_iters=n_iters)

            #Take top move 
            #revised_probs   = chess_utils.normalize(list(move_probs.values()),temperature=chess_utils.temp_scheduler(tree.board.ply()))
            #probabilities   = [0 for move in chess_utils.CHESSMOVES] 
            #for move,prob in zip(move_probs,revised_probs):
                #probabilities[chess_utils.MOVE_TO_I[move]] = prob 
            
            #Append data            
            game_experiences.append([tree.board.fen(),{m.uci():n for m,n in move_probs.items()},0])
            
            #sample and make move 
            top_move        = None
            top_visits      = -1 
            for move,n_visits in move_probs.items():
                if n_visits > top_visits:
                    top_move    = move 
                    top_visits  = n_visits

            #Make move
            result          = tree.make_move(move)

        #Add game experiences
        for i in range(len(game_experiences)):
            game_experiences[i][2]  = result
        data += game_experiences

        n_moves += tree.board.ply()
        
    #Get file to save to 
    with open(os.path.join(DATAPATH,uid+"_"+str(offset)),'w') as file:
        file.write(json.dumps(data))

    return n_moves
if __name__ == "__main__":
    uid             = "".join([str(random.randint(0,9)) for _ in range(5)])
    offset          = 0
    
    while True:
        t0          = time.time()
        n_games     = 1
        n_moves     = generate_data(n_games,1000,uid,offset,max_game_ply=200)
        print(f"\tplayed {n_games} in {(time.time()-t0):.2f}s -> {(time.time()-t0)/n_games:.2f}s/game\t{(time.time()-t0)/n_moves:.2f}s/move\n")
        offset      += 1

