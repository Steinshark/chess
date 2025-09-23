import mctree
import settings
from model import ChessTransformer
import numpy
import random
import time 
import json 
import bulletchess 


dummy_model = ChessTransformer(emb_dim=settings.N_EMBED,num_layers=8,num_heads=8)
tree    = mctree.MCTree()
tree.load_dict(dummy_model)

def runcode(tree:mctree.MCTree):
    #Instantiate model 

    #Build tree 

    n_games         = 128
    collections     = []
    t0              = time.time()
    n_moves         = 0 
    for _ in range(n_games):
        game_experiences    = [] 

        while tree.game_over() is None:
            move_counts     = tree.evaluate_root(400)
            move_probs      = numpy.asarray(list(move_counts.values()))
            next_move       = random.choices(list(move_counts.keys()),weights=move_probs,k=1)[0]
            
            game_experiences.append([tree.board.fen(),move_counts,next_move,None])
            tree.make_move(bulletchess.Move.from_uci(next_move))   
            
        
            #print(f"{(time.time()-t0)/len(game_experiences):.2f}s/move")

        #Update experiences
        game_result     = tree.game_over()
        for i in range(len(game_experiences)):
            game_experiences[i][-1] = game_result
            collections.append(game_experiences[i])

        n_moves += tree.board.halfmove_clock
        tree    = mctree.MCTree()
        tree.load_dict(dummy_model)

        print(f"{len(collections)/(time.time()-t0) :.2f}moves/sec -> {game_result}")

    with open(f"data/{random.randint(1_000_000_000,9_999_999_999)}.json",'w') as writefile:
        writefile.write(json.dumps(collections))

runcode(tree)