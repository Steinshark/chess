#Author: Everett Stenberg
#Description:   trains an initial version of the network so a s
#               to reduce overhead



import chess
import chess_utils
from model import ChessModel2
import numpy 
import torch
import trainer
from torch.utils.data import DataLoader
import random 
import multiprocessing
import os 


import random

def sample_data(experiences, n):
    white_wins = []
    black_wins = []
    draws = []
    
    # Shuffle experiences while preserving original order
    #random.seed(42)  # Setting a seed for reproducibility
    shuffled_experiences = list(experiences)
    random.shuffle(shuffled_experiences)
    
    for experience in shuffled_experiences:
        fen, chosen_move_i, prior_prob_vector, outcome = experience
        if outcome == 1 and len(white_wins) < n:
            white_wins.append(experience)
        elif outcome == -1 and len(black_wins) < n:
            black_wins.append(experience)
        elif outcome == 0 and len(draws) < n:
            draws.append(experience)
        
        if len(white_wins) == n and len(black_wins) == n and len(draws) == n:
            print(f"\tconditions satisfied n={n}")
            break
    
    return white_wins, black_wins, draws


def pretrain(chess_model:ChessModel2,exps,bs=4096,lr=.001,wd=.01,betas=(.5,.75)):

    dataset                 = trainer.PretrainDataset(exps)
    dataloader              = DataLoader(dataset,batch_size=bs,shuffle=True)
    optim                   = torch.optim.AdamW(chess_model.parameters(),lr=lr,weight_decay=wd,betas=betas)


    for batch_i,batch in enumerate(dataloader):
        
        optim.zero_grad()

        fens,moves,priors,zs     = batch 
        # print(f"shape move_i is {moves.shape}")
        # print(f"priors shape is {priors.shape}")
        # print(f"shape z_val is {zs.shape}")


        #Reinforce move in win 
        stacker             = []
        with torch.no_grad():
            for game_i in range(len(fens)):
                pass 
                #NO PROB TRAIN
                turn            = 1 if fens[game_i].split(" ")[1] == 'w' else -1
                chosen_move     = moves[game_i]
                prior_probs     = priors[game_i]
                outcome         = zs[game_i]

                # print(f"turn: {turn}, outzome: {outcome}")
                new_probs       = torch.zeros(size=(1968,1),dtype=torch.float)

                #If we won, then use that move 
                if outcome == turn:
                    new_probs[chosen_move] = 1 
                    probs               = new_probs
        
                elif outcome == (turn * -1):
                    correction_factor   = .5*prior_probs[chosen_move]
                    new_probs           = prior_probs.unsqueeze_(dim=1)
                    new_probs[chosen_move] -= correction_factor
                    
                    #Recorrect using softmax 
                    probs               = torch.softmax(new_probs,1)

                #Else just dont even do anything
                else:
                    probs       = prior_probs.unsqueeze_(dim=1)

                stacker.append(probs)
            
            probabilities               = torch.stack(stacker)




            

        #Get model output 
        board_rerps         = chess_utils.batched_fen_to_tensor(fens).float()
        #input(f"reprs are shape {board_rerps.shape}")
        probs,v             = chess_model.forward(board_rerps)

        #Train 
        zs                  = zs.unsqueeze(dim=-1).float()
        v_error             = torch.nn.functional.mse_loss(zs,v)
        #probs               = probs.unsqueeze(dim=-1).float()
        #p_error             = torch.nn.functional.cross_entropy(probs,probabilities)

        #total_err           = v_error + p_error
        v_error.backward()

        optim.step()


def dirichlet_scheduler(move_n):

    return numpy.exp(-.1*move_n) / 4 


def generate_experiences(pack):#chess_model:ChessModel2,n_iters=16*4096):
    chess_model,n_iters     = pack
    dataset             = []

    with torch.no_grad():
        while len(dataset) < n_iters:

            #Create a chess board x
            board           = chess.Board()
            local_exp       = [] 


            #Make moves while its not gameover 
            while not board.is_game_over() and board.ply() < 200:
                
                #Get repr
                position_fen            = board.fen()

                #Make one pass with the network to get prior_probs
                prior_probs,v           = chess_model.forward(chess_utils.batched_fen_to_tensor([position_fen]).float())

                #Correct for legal moves
                board_moves             = list(board.legal_moves)
                revised_numpy_probs     = numpy.take(prior_probs[0].numpy(),[chess_utils.MOVE_TO_I[move] for move in board_moves])
                revised_numpy_probs     = chess_utils.normalize_numpy(revised_numpy_probs,1)

                full_prob_dist          = numpy.zeros(1968,numpy.float32)

                #use dirichlet to add randomness
                dirichlet               = numpy.random.dirichlet([.1 for _ in revised_numpy_probs]) 
                dir_epsilon             = dirichlet_scheduler(board.ply())
                for prob,diri,move in zip(revised_numpy_probs,dirichlet,board_moves):

                    move_i              = chess_utils.MOVE_TO_I[move]
                    full_prob_dist[move_i]  = ((1-dir_epsilon)*prob + (dir_epsilon)*diri)

                #record data    
                move_i                  = numpy.argmax(full_prob_dist)

                #Push the move 
                board.push(chess_utils.I_TO_MOVE[move_i])
                local_exp.append([position_fen,move_i,full_prob_dist,0])

            #Check result and update 
            result  = 0 
            if board.result() == "1-0":
                result = 1
            elif board.result() == "0-1":
                result = -1 
            elif board.result() == "1/2-1/2" or board.result() == "*":
                result = 0 
            else:
                print(f"got strange {board.result()}")

            
            for exp in local_exp:
                exp[-1] = result
                dataset.append(exp)

            #print(f"\tresult of game was {result} - {len(dataset)}")


    
    return dataset


if __name__ == "__main__":

    chess_model     = ChessModel2(19,16,act=torch.nn.LeakyReLU).float().eval()
    #chess_model.load_state_dict(torch.load("pretrain/8.dict"))
    iteration       = 50
    exp_size        = 32768
    full_exps       = [] 

    n_workers       = 4
    with multiprocessing.Pool(n_workers) as pool:
        results         = pool.map(generate_experiences, [(chess_model,4096) for _ in range(n_workers)])
    
    for res in results:
        full_exps += res

    for iter_n in range(iteration):
        print(f"\n\n\tStart Data Collection [{iter_n}/{iteration}]")
        chess_model.eval()
        with multiprocessing.Pool(n_workers) as pool:
            results         = pool.map(generate_experiences, [(chess_model,exp_size//n_workers) for _ in range(n_workers)])
        
        for res in results:
            full_exps += res
        #ds              = generate_experiences(chess_model,exp_size)

        white_w         = [1 for exp in full_exps if exp[-1] == 1]
        black_w         = [1 for exp in full_exps if exp[-1] == -1]

        print(f"\tw: {sum(white_w)}\tb: {sum(black_w)}")

        #Write a function to sample equal amounts of B,W,D
        
        scalar          = min(1.85+((1+iter_n)*.5), 4)
        full_exps       = full_exps[-int(scalar*exp_size):]
        white,black,draw= sample_data(full_exps,2048)
        trainset        = white + black + draw 
        random.shuffle(trainset)
        #trainset        = random.sample(full_exps,int(exp_size))
        
        print(f"\ttraining on [{len(trainset)}/{len(full_exps)}]")
        chess_model.train()
        pretrain(chess_model,trainset,bs=1024,lr=.0001,wd=0.01,betas=(.75,.9))
        p,v             = trainer.check_vs_stockfish(chess_model)
        print(f"\titer: {iter_n}: p_loss={p:.4f}\tv_loss={v:.4f}")
        #Get d4,e4 provs 
        prob,v_s        = chess_model.eval().forward(chess_utils.batched_fen_to_tensor([chess.Board().fen()]).float())
        mate_fen        = 'r1bqkbnr/2pppppp/ppn5/8/2B5/4PQ2/PPPP1PPP/RNB1K1NR w KQkq - 2 4'
        bmate_fen       = 'rnbqkbnr/pppp1ppp/4p3/8/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 2'
        e4              = prob[0][chess_utils.MOVE_TO_I[chess.Move.from_uci("e2e4")]]
        f3              = prob[0][chess_utils.MOVE_TO_I[chess.Move.from_uci("f2f3")]]
        prob,v_m        = chess_model.eval().forward(chess_utils.batched_fen_to_tensor([chess.Board(fen=mate_fen).fen()]).float())
        f7              = prob[0][chess_utils.MOVE_TO_I[chess.Move.from_uci("f3f7")]]
        c4              = prob[0][chess_utils.MOVE_TO_I[chess.Move.from_uci("c4f7")]]
        prob,v_b        = chess_model.eval().forward(chess_utils.batched_fen_to_tensor([chess.Board(fen=bmate_fen).fen()]).float())
        h4              = prob[0][chess_utils.MOVE_TO_I[chess.Move.from_uci("d8h4")]]
        a6              = prob[0][chess_utils.MOVE_TO_I[chess.Move.from_uci("a7a6")]]
        print(f"\t START v:{v_s.item():.4f}\te4: {e4:.5f}\tf3:{f3:.5f}")
        print(f"\t WMATE v:{v_m.item():.4f}\tf7: {f7:.5f}\tc4:{c4:.5f}")
        print(f"\t BMATE v:{v_b.item():.4f}\th4: {f7:.5f}\ta6:{c4:.5f}")
        if not os.path.exists("pretrain"):
            os.mkdir("pretrain")
        torch.save(chess_model.state_dict(),os.path.join("pretrain",f"{iter_n}.dict"))
