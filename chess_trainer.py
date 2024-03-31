import chess_utils 
import math 
import random 
import chess
from mctree import MCTree
import os 
import json
import time
import torch
DATAPATH        = "C:/data/chess/mcts_train"

for path in ["C:/data","C:/data/chess","C:/data/chess/mcts_train"]:
    if not os.path.exists(path):
        os.mkdir(path)



def generate_data(n_games,n_iters):

    #Ensure training consistent
    random.seed(512)
    torch.manual_seed(512)

    data    = []

    for _ in range(n_games):

        tree                = MCTree()
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
    offset  = len([int(file) for file in os.listdir(DATAPATH)])
    print(f"saving to {os.path.join(DATAPATH,str(offset))}")
    with open(os.path.join(DATAPATH,str(offset)),'w') as file:
        file.write(json.dumps(data))

if __name__ == "__main__":
    while True:
        t0  = time.time()
        n_games     = 25
        generate_data(n_games,n_iters=1000)
        print(f"played {n_games} in {(time.time()-t0):.2f}s -> {(time.time()-t0)/n_games:.2f}s/game")