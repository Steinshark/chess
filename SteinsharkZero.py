import mctree
import settings
from model import ChessTransformer
import numpy
import random
import time 
import json 
import bulletchess 
import torch



def runcode(current_epoch:int):
    #Instantiate model 

    #Build tree 
    n_games         = 1
    dataset         = []
    t0              = time.time()
    for _ in range(n_games):
        
        #Build and load model
        model   = ChessTransformer(128,4,12,4)
        model.load_state_dict(torch.load(f'C:/code/chess/models/ep{current_epoch}.pt'))
        tree    = mctree.MCTree()
        tree.load_dict(model)

        try:
            game_experiences    = [] 

            while tree.game_over() is None:
                tree.evaluate_root(400)
                next_move       = tree.sample_move(temp=.7)
                game_experiences.append([tree.board.fen(),tree.counts,next_move,None])
                tree.make_move(bulletchess.Move.from_uci(next_move))   

            #Update experiences
            game_result     = tree.game_over()
            for i in range(len(game_experiences)):
                game_experiences[i][-1] = game_result
                dataset.append(game_experiences[i])

            del tree #attempt to clear some memory

            print(f"\t{len(game_experiences)} moves\t{len(game_experiences)/(time.time()-t0) :.2f}moves/sec -> {game_result}")
        except ValueError:
            pass
        
    with open(f"data/{random.randint(1_000_000_000,9_999_999_999)}.json",'w') as writefile:
        writefile.write(json.dumps(dataset))


for _ in range(32):
    print(f"\n\nStart new game iteration")
    runcode(2)   
