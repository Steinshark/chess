import chess_utils 
import random 
import chess
from mctree import MCTree
import os 
import json
import time
import torch
import sys 


DATAPATH        = "data/mcts_train"

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


def generate_data(n_games,n_iters,uid,offset):
    data    = []
    print(f"\tsaving to {os.path.join(DATAPATH,uid+'_'+str(offset))}")

    for game_i in range(n_games):

        tree                = MCTree(verbose= bool(game_i == 0),max_game=160)
        result              = None 
        game_experiences    = []
        while result is None:

            #Run search
            move_probs      = tree.calc_next_move(n_iters=n_iters)
            revised_probs   = chess_utils.normalize(list(move_probs.values()),temperature=chess_utils.temp_scheduler(tree.board.ply()))
            probabilities   = [0 for move in chess_utils.CHESSMOVES] 
            for move,prob in zip(move_probs,revised_probs):
                probabilities[chess_utils.MOVE_TO_I[move]] = prob 
            
            #Append data            
            game_experiences.append([tree.board.fen(),probabilities,0])
            
            #sample and make move 
            next_move_i     = random.choices(chess_utils.CHESSMOVES, probabilities,k=1)[0]
            result          = tree.make_move(chess.Move.from_uci(next_move_i))

        for i in range(len(game_experiences)):
            game_experiences[i][2]  = result
        data += game_experiences
        
    #Get file to save to 
    with open(os.path.join(DATAPATH,uid+"_"+str(offset)),'w') as file:
        file.write(json.dumps(data))

if __name__ == "__main__":

    uid             = "".join([str(random.randint(0,9)) for _ in range(5)])
    offset          = 0
    
    while True:
        t0          = time.time()
        n_games     = 4
        generate_data(n_games,1000,uid,offset)
        print(f"played {n_games} in {(time.time()-t0):.2f}s -> {(time.time()-t0)/n_games:.2f}s/game")
        offset      += 1

