#Author: Everett Stenberg
#Description:   defines functions for training the network from
#               start to finish



import multiprocessing
import multiprocessing.pool
import torch
from model import ChessModel2
import mctree
from mctree import MCTree
import chess
from itertools import combinations
from trainer import TrainerExpDataset
from torch.utils.data import DataLoader
import chess_utils
from collections import OrderedDict

n_training_iterations       = 10
top_model                   = 0
model_list                  = {0:ChessModel2(19,24).cuda().state_dict()}



#Used to play 1 game of model1 vs model2
def showdown_match(args_package):
    model1,model2,n_iters,max_game_ply  = args_package
    board           = chess.Board()
    
    engine1         = MCTree(max_game_ply=max_game_ply)
    engine1.load_dict(model1)
    engine2         = MCTree(max_game_ply=max_game_ply)
    engine2.load_dict(model2)

    while not board.is_game_over() and (board.ply() <= max_game_ply):

        #Make move with current player
        move_probs  = engine1.evaluate_root(n_iters=n_iters) if board.turn else engine2.evaluate_root(n_iters=n_iters)
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
    
    if board.result() == "1-0":
        return 1
    elif board.result() == '0-1':
        return -1 
    else:
        return 0



def find_best_model(model_params:dict,max_game_ply,n_iters):
    top_model           = 0 
    
    

    for challenger_i,challenger_params in model_params.items():
        
        #Skip first challenger (well use it as first opponent)
        if challenger_i == 0:
            continue 
        top_model_params        = model_params[top_model]

        #Play 10 as W 
        with multiprocessing.Pool(4) as pool:
            results_w           = pool.map(showdown_match,[(challenger_params,top_model_params,n_iters,max_game_ply) for _ in range(5)])
            challenger_wins     = list(results_w).count(1)
            champion_wins       = list(results_w).count(-1)
        #Play 10 as B
        with multiprocessing.Pool(4) as pool:
            results_b           = pool.map(showdown_match,[(top_model_params,challenger_params,n_iters,max_game_ply) for _ in range(5)])
            challenger_wins     += list(results_w).count(-1)
            champion_wins       += list(results_w).count(1)

        #Set new champion for any number of better wins
        if champion_wins > challenger_wins:
            top_model           = challenger_i
    
    return top_model


#Plays one game using the specified model and returns all experiences from that game
#   max_game_ply is the max moves per game 
#   n_iters is the number of iterations run by the MCTree to evaluate the position
def play_game(model_dict:str|OrderedDict|torch.nn.Module,max_game_ply=160,n_iters=800,wildcard=None):

    #Create board and tree
    engine              = MCTree(max_game_ply=max_game_ply)
    game_experiences    = []
    result              = None
    engine.load_dict(model_dict)

    #Play out game
    while result is None:
        
        #Check wildcard 
        if not wildcard is None:
            if "running" in wildcard.__dict__:
                if not wildcard.running:
                    return None
        #Evaluate move 
        move_probs      = engine.evaluate_root(n_iters=n_iters)

        #Add experiences
        game_experiences.append([engine.board.fen(),{m.uci():n for m,n in move_probs.items()},0])

        #get best move by visit count
        top_move        = None
        top_visits      = -1 
        for move,n_visits in move_probs.items():
            if n_visits > top_visits:
                top_move    = move 
                top_visits  = n_visits
        
        #Push move to board and setup engine for next mve
        result          = engine.make_move(top_move)

    #update game outcome in set of experiences
    for i in range(len(game_experiences)):
        game_experiences[i][2]  = result

    return game_experiences


#Generates a full generation of training data
def generate_training_games(model_dict:str,max_game_ply=160,n_iters=800,n_training_games=1024):

    training_game_experience_set    = []
    with multiprocessing.Pool(8) as pool:
        game_experiences    = pool.imap_unordered(play_game,
                                                  [(model_dict,max_game_ply,n_iters) for _ in range(n_training_games)])

    #Add all individual experiences 
    for experience_list in game_experiences:
        training_game_experience_set += experience_list

    return training_game_experience_set


#Trains the model on the given experiences
def train_model(iter:int,chess_model:ChessModel2,experiences,bs=1024,lr=.001,wd=.01,betas=(.5,.75),n_epochs=1):

    #Get data items together 
    dataloader      = DataLoader(TrainerExpDataset(experiences),batch_size=bs,shuffle=True)

    #Get training items together
    optimizer       = torch.optim.Adam(chess_model.parameters(),lr=lr,weight_decay=wd,betas=betas)
    loss_fn_v       = torch.nn.MSELoss()
    loss_fn_p       = torch.nn.CrossEntropyLoss()

    #Get model together
    chess_model.train()

    #Save losses
    p_losses        = [] 
    v_losses        = [] 
    sum_losses      = [] 

    for i, batch in enumerate(dataloader):

        #Zero
        chess_model.zero_grad()

        #Unpack data
        fens,distr,z            = batch

        #Transform data to useful things
        board_repr              = chess_utils.batched_fen_to_tensor(fens).to(mctree.DEVICE).float()
        z_vals                  = z.unsqueeze(dim=1).float().to(mctree.DEVICE)
        distr                   = distr.to(mctree.DEVICE)

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

        print(f"\t\tITER[{iter}]\tp_loss:{p_loss_out.detach().cpu().item():.4f}\n\t\tv_loss:{v_loss_out.detach().cpu().item():.4f}\n")


if __name__ == "__main__":
    for iteration in range(n_training_iterations):

        #Generate training games
        generation_experiences  = generate_training_games(model_list[top_model])

        #Initialize a clone of current top model and train on current experience set 
        cur_gen_model           = ChessModel2(19,24).to(mctree.DEVICE).float().train()
        cur_gen_model.load_state_dict(top_model)
        train_model(iteration,cur_gen_model,generation_experiences)
        model_list[iteration]   = cur_gen_model.state_dict()
        
        #Find current best model 
        best_model              = find_best_model() 


        input(f"gen: {iteration} done\n\ttop model: {top_model}")
