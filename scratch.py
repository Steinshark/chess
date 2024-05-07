import model 
import torch
import utilities
import random
import numpy 

#load current model 
cmodel  = model.ChessModel(17,16)
cmodel.load_state_dict(torch.load("generations/gen_1.dict"))
cmodel.eval()
import chess 


while True:
    
    b = chess.Board()

    #Look for nan val 
    with torch.no_grad():
        p,v         = cmodel(utilities.batched_fen_to_tensor([b.fen()]).float())
        p           = p.detach().numpy()
        v           = v.detach().numpy()
        if numpy.isnan(p.sum()):
            input(f"P on fen {b.fen()}->\n{list(p)}") 
        if numpy.isnan(v.sum()):
            input(f"V on fen {b.fen()}->\n{list(v)}") 

        b.push(random.choice(list(b.generate_legal_moves())))

        if b.is_game_over():
            b = chess.Board()