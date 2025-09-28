from model import ChessTransformer
from mctree import MCTree
import bulletchess as bc
from chess_data import ChessSet
from torch.utils.data import DataLoader
import torch 

def trainer(model:ChessTransformer,training_iter:int):


    BS              = 512 
    EP              = 1
    LR              = 1e-3
    WD              = 1e-4

    dataset         = ChessSet("C:/code/chess/data")
    dataloader      = DataLoader(dataset,batch_size=BS,shuffle=True)

    optimizer       = torch.optim.AdamW(model.parameters(),LR,weight_decay=WD)

    losses          = [0] 
    for ep in range(EP):

        print(f"EP {ep}")
        accu                            = 0 
        for i,batch in enumerate(dataloader):

            x,pi,_,z                    = batch

            logits,z_                   = model.forward(x,stage_one=False)

            log_pi_                     = torch.nn.functional.log_softmax(logits,dim=-1)

            prob_loss:torch.Tensor      = -(log_pi_ * pi).sum(dim=1).mean()
            val_loss:torch.Tensor       = torch.nn.functional.mse_loss(z_.flatten(),z)

            loss                        = prob_loss + val_loss 

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses[-1]                  += loss.cpu().item()
            accu                        += 1

            if i % 10 == 0 and not i ==0:
                print(f"i={i//10}/{len(dataloader)//10}\t" + f"loss={loss:.6f}")
                losses[-1]              /= accu 
                accu                    = 0 
                losses.append(0)


    torch.save(model.state_dict(),F'C:/code/chess/models/ep{training_iter}.pt')
    return losses


def determine_if_model_is_55_percent_better(challenger:ChessTransformer,locked_in_MFer:ChessTransformer):


    #Play 100 games  
    challenger_wins = 0
    locked_in_wins  = 0
    draws           = 0 
    
    #Challenger plays white
    for _ in range(50):
        outcome     = play_game(challenger,locked_in_MFer)
        if outcome == 1:
            challenger_wins += 1
        elif outcome == -1:
            locked_in_wins += 1 
        else:
            draws +=1 

    #Challenger plays black
    for _ in range(50):
        outcome     = play_game(locked_in_MFer,challenger)
        if outcome == -1:
            challenger_wins += 1
        elif outcome == 1:
            locked_in_wins += 1 
        else:
            draws +=1 


    #And?
    print(f"challenger:\t{challenger_wins}\ncurrent:\t{locked_in_wins}\ndraws:\t{draws}")



def play_game(white_model:ChessTransformer,black_model:ChessTransformer,n_eval=1000):

    white_tree  = MCTree()
    white_tree.load_dict(white_model)

    black_tree  = MCTree()
    black_tree.load_dict(black_model)
    
    gameboard   = bc.Board()

    while True:
        
        #Make whites move 
        white_tree.evaluate_root(n_eval,training=False)
        next_move       = white_tree.sample_move(temp=.01)

        #Add moves 
        white_tree.make_move(next_move)
        black_tree.make_move(next_move)
        gameboard.apply(bc.Move.from_uci(next_move))

        if gameover(gameboard):
            return gamestatus(gameboard)
        
        black_tree.evaluate_root(n_eval,training=False)
        next_move       = black_tree.sample_move(temp=.01)

        #Add moves 
        black_tree.make_move(next_move)
        white_tree.make_move(next_move)
        gameboard.apply(bc.Move.from_uci(next_move))

        if gameover(gameboard):
            return gamestatus(gameboard)
        

#Helper functions to abstract out from the larger functions 
def gameover(board:bc.Board):

    return board in bc.CHECKMATE or board in bc.FORCED_DRAW


def gamestatus(board:bc.Board):

    if board in bc.CHECKMATE:
        
        if board.turn is bc.BLACK:
            return 1
        
        if board.turn is bc.WHITE:
            return-1 
    
    return 0 



if __name__ == "__main__":
    runs    = {}
    from matplotlib import pyplot as plt 

    model       = ChessTransformer(128,4,12,4)
    model.load_state_dict(torch.load("C:/code/chess/models/ep1.pt"))
    print(f"model is {sum([p.numel() for p in model.parameters()])}")
    pre_weight  = model.state_dict()

    lossouts    = trainer(model,2)    
    plt.plot(lossouts,label=f"n_embed={128}")
    
    plt.legend()
    plt.show()
