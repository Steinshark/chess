import model 
import torch
import utilities
import random
import numpy 
import settings
#load current model 
cmodel  = model.ChessModel2().float()
cmodel.load_state_dict(torch.load("curstate.pt"))
cmodel.eval()
#cmodel 	= torch.jit.trace(cmodel,torch.randn((1,17,8,8)))
#cmodel 	= torch.jit.freeze(cmodel)
import bulletchess 



b       = bulletchess.Board(fen="rnbqkbnr/ppppp2p/8/5pp1/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 3")
inp     = utilities.batched_fen_to_tensor([b.fen()]).float()

p,v     = cmodel(inp)
p       = p[0].detach().numpy()
p       = p[[utilities.MOVE_TO_I[move] for move in b.legal_moves]]
p       = utilities.normalize_numpy(p)
move_p  = {move.uci(): p_ for move,p_ in zip(b.legal_moves,p)}
print(sorted(move_p.items(),key=lambda x:x[1],reverse=True))
import torchvision

