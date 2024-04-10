#Author: Everett Stenberg
#Description:   run to generate training data.  
#               


import chess_utils 
import random 
from mctree import MCTree
import os 
import json
import time
import torch
import sys 


#Ensure local path exists
DATAPATH        = "data4/"
if not os.path.exists(DATAPATH):
    os.mkdir(DATAPATH)
    
    
#Determine device using availability and command line options
if sys.argv and "--cpu" in sys.argv:

    #Force CPU
    DEVICE      = torch.device('cpu')


elif sys.argv and "--cuda" in "".join(sys.argv):

    #If cuda specfied, use that device ID
    cuda_device = [command.replace('--','') for command in sys.argv if '--cuda' in command ][0]
    DEVICE      = torch.device(cuda_device)

    #attempt device check
    try:
        test    = torch.tensor([1,2,3],device=DEVICE)
    except RuntimeError:
        print(f"CUDA id:{cuda_device[6:]} does not exists on machine with {torch.cuda.device_count()} CUDA devices")
        exit()


else:

    #Default to CUDA device
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Report device used
print(f"\tset device to {DEVICE}")


#Handle remaining argsz
if sys.argv and "--niters:" in "".join(sys.argv):
    NITERS  = [command.replace('--niters:','') for command in sys.argv if '--niters' in command ][0]
else:
    NITERS  = 800

if sys.argv and "--depth:" in "".join(sys.argv):
    DEPTH   = [command.replace('--depth:','') for command in sys.argv if '--depth:' in command ][0]
else:
    DEPTH   = 160



#Function to generate training games.
#   Runs n_games iterations
def generate_data(n_games,n_iters,uid,offset,max_game_ply=160,model_dict="chess_model_iter2.dict"):
    data                    = []
    print(f"\tsaving to {os.path.join(DATAPATH,uid+'_'+str(offset))}")
    n_moves                 = 0

    for game_i in range(n_games):
        #Create game-specific components
        tree                = MCTree(max_game_ply=max_game_ply)
        tree.load_dict(model_dict)
        result              = None 
        game_experiences    = []

        #Calculate and push moves to board until result is not None
        while result is None:

            #Run search
            t0                  = time.time()
            move_probs      = tree.evaluate_root(n_iters=n_iters)
            
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
            #print(f"play {top_move} in {(time.time()-t0):.2f}s")
            result          = tree.make_move(top_move)


        #Add game experiences
        for i in range(len(game_experiences)):
            game_experiences[i][2]  = result

        #Add this game to the total 
        data += game_experiences

        #Track n_moves for timing reasons
        n_moves += tree.board.ply()
        

    #Write experiences in json format to file 
    with open(os.path.join(DATAPATH,uid+"_"+str(offset)),'w') as file:
        file.write(json.dumps(data))

    #Return number of moves made for timing reasons
    return n_moves



if __name__ == "__main__":
    uid             = "".join([str(random.randint(0,9)) for _ in range(5)])
    offset          = 0
    
    while True:
        t0          = time.time()
        n_games     = 1
        n_moves     = generate_data(n_games,1000,uid,offset,max_game_ply=160,model_dict="chess_model_iter4.dict")                               #Alphazero was trained with 800iters per search...
        print(f"\tplayed {n_games} in {(time.time()-t0):.2f}s -> {(time.time()-t0)/n_games:.2f}s/game\t{(time.time()-t0)/n_moves:.2f}s/move\n")
        offset      += 1

