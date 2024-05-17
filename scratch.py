import model 
import torch
import utilities
import random
import numpy 
import settings
#load current model 
cmodel  = model.ChessModel(**settings.MODEL_KWARGS).bfloat16()
cmodel.load_state_dict(torch.load("generations/gen_58.dict"))
cmodel.float()
cmodel.eval()
cmodel 	= torch.jit.trace(cmodel,torch.randn((1,17,8,8)))
cmodel 	= torch.jit.freeze(cmodel)
import chess 



b       = chess.Board()
inp     = utilities.batched_fen_to_tensor([b.fen()]).float()

p,v     = cmodel(inp)
p       = p[0].detach().numpy()
p       = p[[utilities.MOVE_TO_I[move] for move in b.legal_moves]]
p       = utilities.normalize_numpy(p)
move_p  = {move.uci(): p_ for move,p_ in zip(b.legal_moves,p)}
print(sorted(move_p.items(),key=lambda x:x[1],reverse=True))
