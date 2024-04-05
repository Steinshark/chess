import torch
import random 
import model
import json 
import os 
import chess
import chess_utils
import mctree
from mctree import MCTree
from torch.utils.data import Dataset,DataLoader

DEVICE      = mctree.DEVICE

class chessExpDataSet(Dataset):

    def __init__(self,filepath:str):
        
        self.fens           = [] 
        self.distros        = []
        self.z_vals         = [] 
        self.data           = []


        filelist        = os.listdir(filepath)
        random.shuffle(filelist)

        for filename in filelist:

            #Fix to full path
            filename    = os.path.join(filepath,filename)

            #Load game 
            with open(filename,"r") as file:
                game_data   = json.loads(file.read())

                for item in game_data:
                    fen             = item[0]
                    distribution    = item[1]
                    game_outcome    = item[2]

                    self.fens.append(fen)
                    self.distros.append(distribution)
                    self.z_vals.append(game_outcome)

        self.distros    = list(map(chess_utils.movecount_to_prob,self.distros))

    def __getitem__(self,i:int):
        return self.fens[i],self.distros[i],self.z_vals[i]
    
    def __len__(self):
        return len(self.fens)



def train_model(chess_model:model.ChessModel,dataset:chessExpDataSet,bs=1024,lr=.0001,wd=0,betas=(.5,.75),n_epochs=1):

    #Get data items together 
    dataloader      = DataLoader(dataset,batch_size=bs,shuffle=True)

    #Get training items together
    optimizer       = torch.optim.Adam(chess_model.parameters(),lr=lr,weight_decay=wd,betas=betas)
    loss_fn_v       = torch.nn.MSELoss()
    loss_fn_p       = torch.nn.CrossEntropyLoss()

    p_losses        = [] 
    v_losses        = [] 
    sum_losses      = [] 

    for ep_num in range(n_epochs):

        for i, batch in enumerate(dataloader):

            #Zero
            chess_model.zero_grad()

            #Unpack data
            fens,distr,z            = batch

            #Transform data to useful things
            board_repr              = chess_utils.batched_fen_to_tensor(fens).to(DEVICE).float()
            z_vals                  = z.unsqueeze(dim=1).float().to(DEVICE)
            distr                   = distr.to(DEVICE)

            #Get model out
            probs,evals             = chess_model.forward(board_repr)

            #Get losses 
            p_loss                  = loss_fn_p(probs,distr) 
            v_loss                  = loss_fn_v(z_vals,evals)
            model_loss              = p_loss + v_loss
            
            #Save 
            p_losses.append(p_loss)
            v_losses.append(v_loss)

            #Backward
            model_loss.backward()

            #Optim
            optimizer.step()


    p_loss_out  = torch.sum(torch.cat([p.unsqueeze(dim=0) for p in p_losses])) / len(p_losses)
    v_loss_out  = torch.sum(torch.cat([p.unsqueeze(dim=0) for p in v_losses])) / len(v_losses)

    print(f"\t\tp_loss:{p_loss_out.detach().cpu().item():.4f}\n\t\tv_loss:{v_loss_out.detach().cpu().item():.4f}\n")



def check_vs_stockfish(chess_model:model.ChessModel):

    #Get loss items ready
    loss_fn_v       = torch.nn.MSELoss()
    loss_fn_p       = torch.nn.CrossEntropyLoss()
    v_losses        = []
    p_losses        = []

    #Get baseline data ready 
    with open('baseline/moves.txt','r') as file:
        baseline_data   = json.loads(file.read())

    #Prep model
    chess_model     = chess_model.eval().to(DEVICE)
    
    with torch.no_grad():
        
        for experience in baseline_data:

            #Get data
            board_repr  = chess_utils.batched_fen_to_tensor([experience[0]]).to(DEVICE).float()
            board_eval  = torch.tensor([chess_utils.clean_eval(experience[1])]).to(DEVICE).float().unsqueeze(dim=0)
            probs       = [0 for _ in chess_utils.CHESSMOVES]
            probs[chess_utils.MOVE_TO_I[chess.Move.from_uci(experience[2])]]    = 1
            board_prob  = torch.tensor(probs).to(DEVICE).float().unsqueeze(dim=0)

            #Get model 
            prob,eval   = chess_model.forward(board_repr)

            p_losses.append(loss_fn_p(prob,board_prob).cpu().detach().item())
            v_losses.append(loss_fn_v(eval,board_eval).cpu().detach().item())

    return sum(p_losses)/len(p_losses), sum(v_losses)/len(v_losses)


def showdown_match(model1,model2,n_iters=2000):

    board           = chess.Board()
    max_game_ply    = 200 
    
    engine1         = MCTree(max_game_ply=max_game_ply)
    engine1.load_dict(model1)
    engine2         = MCTree(max_game_ply=max_game_ply)
    engine2.load_dict(model2)

    while not board.is_game_over() and not (board.ply() > max_game_ply):

        #Make white move 
        move_probs  = engine1.calc_next_move(n_iters=n_iters) if board.turn else engine2.calc_next_move(n_iters=n_iters)
        placehold   = engine2.perform_iter() if board.turn else engine1.perform_iter()

        #find best move
        top_move        = None
        top_visits      = -1 
        for move,n_visits in move_probs.items():
            if n_visits > top_visits:
                top_move    = move 
                top_visits  = n_visits

        #Make move
        board.push(top_move)
        engine1.make_move(top_move)
        engine2.make_move(top_move)
        torch.cuda.empty_cache()
    
    if board.result() == "1-0":
        return 1
    elif board.result() == '0-1':
        return -1 
    else:
        return 0


def matchup(n_games,model1,model2,n_iters):

    model1_wins         = 0
    model2_wins         = 0 
    draws               = 0 

    model1.eval()
    model2.eval()

    for _ in range(n_games//2):
        with torch.no_grad():
            game_result     = showdown_match(model1,model2,n_iters=n_iters)
        if game_result == 1:
            model1_wins += 1
        elif game_result == -1:
            model2_wins += 1
        else:
            draws += 1
    
    for _ in range(n_games//2):
        with torch.no_grad():
            game_result     = showdown_match(model2,model1,n_iters=n_iters)
        if game_result == -1:
            model1_wins += 1
        elif game_result == 1:
            model2_wins += 1
        else:
            draws += 1
    
    return model1_wins,model2_wins,draws


def perform_training():
    path            = "C:/gitrepos/chess/data"
    chess_model     = model.ChessModel2(19,24).to(DEVICE).float()
    dataset         = chessExpDataSet(path)
    for _ in range(3):
        print(f"TRAIN ITER {_}")
        print(f"\tTRAIN MODEL")
        train_model(chess_model,dataset,wd=.01)
        pl,vl   = check_vs_stockfish(chess_model)
        print(f"\tCHECK MODEL")
        print(f"\t\tp_baseline: {pl:.4f}\n\t\tv_baseline: {vl:.4f}\n\n")
    

    #Save to file 
    torch.save(chess_model.state_dict(),"chess_model_iter1.dict")

if __name__ == '__main__':
    
    #perform_training()
    #Good model
    model1          = model.ChessModel2(19,24)
    model1.load_state_dict(torch.load("chess_model_iter1.dict"))
    
    #Bad model
    model2         = model.ChessModel2(19,24)


    p1,l1           = check_vs_stockfish(model1)
    p2,l2           = check_vs_stockfish(model2)

    print(f"model1 v:{l1:.4f}\nmodel2 v:{l2:4f}")

    one,two,draw    = matchup(10,model1,model2,n_iters=1200)

    print(f"outcome: {one}-{two}:{draw}")