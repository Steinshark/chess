import model 
import torch
import utilities
import random
import numpy 

#load current model 
cmodel  = model.ChessModel(17,16)
cmodel.load_state_dict(torch.load("generations/gen_1.dict"))
cmodel.eval()
cmodel 	= torch.jit.trace(cmodel,torch.randn((1,17,8,8)))
cmodel 	= torch.jit.freeze(cmodel)
import chess 



b       = chess.Board()
inp     = utilities.batched_fen_to_tensor([b.fen()]).float()

p,v     = cmodel(inp)
p       = p[0].detach().numpy()
move_p  = {move.uci(): p[utilities.MOVE_TO_I[move]] for move in b.legal_moves}
print
print(sorted(move_p.items(),key=lambda x:x[1],reverse=True))
