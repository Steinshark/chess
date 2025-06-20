#Author: Everett Stenberg
#Description:   trains an initial version of the network so a s
#               to reduce overhead



import chess
import chess.engine
import utilities
from model import ChessModel,ChessModel2, ChessTransformer
import numpy 
import torch
import trainer
from collections import defaultdict
from torch.utils.data import DataLoader
import random 
import os 
import random
from typing import List,Tuple,Dict
import json 
from matplotlib import pyplot as plt  
from data import combine_ds, PositionSet
import data 
import gc
import time 


def parse_score(snippet:str):
    if "0-1" in snippet:
        return -1 
    elif "1-0" in snippet:
        return 1 
    else:
        return 0 
    

def parse_games(filetext:str,elo_thresh:int=2100,save_mod=False,fname2='',timethresh=500):
    print(f"Parsing")
    #Split on Event 
    split_text              = filetext.split("[Event")[1:]

    #Get ELOs
    elo_splits              = [text.split('WhiteElo "')[1] for text in split_text]
    del split_text
    gc.collect()
    #elo_splits              = [text for text in elo_splits if not text[0] in ['?','"']]
    white_elos              = [int(text.split('"')[0]) if not text[0] in ['?','"'] else 0 for text in elo_splits]

    elo_splits              = [text.split('BlackElo "')[1] for text in elo_splits]
    black_elos              = [int(text.split('"')[0]) if not text[0] in ['?','"'] else 0 for text in elo_splits]


    #Get time control 
    time_splits             = [text.split('TimeControl "')[1] for text in elo_splits]
    time_controls           = [text.split('"')[0] for text in time_splits]
    time_controls           = [int(text.split("+")[0]) if not text[0] == '-' else 0 for text in time_controls]
    del time_splits
    gc.collect()

    #print(f"\tfound {len(elo_splits)} candidate games")
    #Take out games with ELOS below 
    good_games              = [(text.split('\n\n')[1].split(" "),welo,belo,time_control) for text,welo,belo,time_control in zip(elo_splits,white_elos,black_elos,time_controls) if welo > elo_thresh and belo > elo_thresh and not 'eval' in text and time_control >= timethresh]
    good_games              = [([m for m in game[0][:-1] if not '.' in m],game[1],game[2],game[3],game[0][-1]) for game in good_games]

    good_games              = [pack for pack in good_games if len(pack[0]) > 2]
    print(f"\tparsed {len(good_games)} {elo_thresh} elo games -> {(100*len(good_games)/len(elo_splits)):.2f}%")

    if save_mod:
        with open(f"{fname2.replace(".pgn","redo.txt").replace("pgns","files")}",'w') as file:
            for moveslist,welo,belo,timecontrol,outcome in good_games:
                moves   =  ",".join(moveslist)
                line    = f"{moves}.{welo}.{belo}.{timecontrol}.{outcome}\n"
                file.write(line)
            print(f"\tsaved games")
        file.close()
    return good_games


#Takes the pack of experiences ([moves as str],w_elo,b_elo,time_control,outcome) and returns a list of (fen,move_played) for all games provided
def run_game(game_pack):
    move_list, elo_w, elo_b, t_control, outcome         = game_pack
    board                   = chess.Board()

    experiences             = [(board.fen(),board.push_san(move).uci(),board.turn,outcome) for move in move_list]

    return experiences


#loads tuples of (fen,move,outcome) 
def load_games(fname:str):
    # moves   =  ",".join(moveslist)
    #             line    = f"{moves}.{welo}.{belo}.{timecontrol}.{outcome}\n"
    #             file.write(line)
    allmoves    = [line.split('.') for line in open(fname,'r').readlines()]
    newmoves    = [(game[0].split(","),int(game[1]),int(game[2]),int(game[3]),game[4].replace("\n","")) for game in allmoves] 
    print(f"generating {len(newmoves)} games")
    allgames    = [] 
    for game in newmoves:
        allgames += run_game(game)
    return allgames

#Returns (fen,uci,outcome)
def get_experiences(filename:str,saving_only=True):
    good_games              = parse_games(open(filename,'r').read(),elo_thresh=2100,save_mod=True,fname2=filename,timethresh=180)
    if not saving_only:
        total_experiences       = list(map(run_game,good_games))
        return total_experiences


def pretrain(chess_model:ChessModel,exps,bs=4096,lr=.0003,wd=.01,betas=(.5,.75)):

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
        board_rerps         = utilities.batched_fen_to_tensor(fens).float()
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


def prepare_batch(game_lists:list[list],prob_func=lambda x: random.random() < 1/(1+2.718**(-.2*(x-10)))):

    positions               = {}
    n_encounters            = 0
    for game in game_lists:

        for position_pack in game:

            fen,move,outcome            = position_pack
            move_i                      = utilities.MOVE_TO_I[chess.Move.from_uci(move)]

            if fen in positions:
                probs,eval              = positions[fen]
                probs[move_i]           += 1 
                eval                    += outcome
                positions[fen]          = [probs,eval]
            else:
                distribution            = numpy.zeros(shape=1968)
                distribution[move_i]    = 1 
                positions[fen]          = [distribution,outcome]
            n_encounters += 1
    
    #Remove all items based on 'prob_func'
    dataset     = {}
    for position, pack in positions.items():
        if prob_func(pack[0].sum()):
            dataset[position] = pack
    print(f"\t{len(dataset)} datapoints created")
    return dataset


def batch_trainer(model:ChessModel2,games_bs=10_000,bs=1024,lr=.0001,wd=.1,n_ep=8):
    
    test_acc                = []
    test_evl                = []
    train_acc               = []
    train_evl               = []
    #plt.ion()
    #plt.show()
    engine 	                = chess.engine.SimpleEngine.popen_uci("C:/gitrepos/stockfish/stockfish-windows-x86-64.exe")

    #Load one of the files 
    for ep_n in range(n_ep):
        for file in os.listdir("C:/data/chess/game_data"):
            if not '.txt' in file:
                continue
            experience_set          = json.loads(open("C:/data/chess/game_data/" + file,'r').read())
            print(f"train on {file} -> {sum([len(l) for l in experience_set])} candidate datapoints")
            model                   = model.cuda().train().float()
            optimizer               = torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.99),weight_decay=wd,amsgrad=True)

            test_set                = prepare_batch(experience_set[:5])
            experience_set          = experience_set[5:]
            print(f"\ttrainset = {len(experience_set)}")

            model                   = model.eval()
            test_loader             = DataLoader(ChessSet(test_set),batch_size=1024)
            test_correct           = 0 
            test_total             = 0 

            eval_list               = []
            with torch.no_grad():
                for batch in test_loader:
                    fen,distr,evl,n = batch
                    inp             = utilities.batched_fen_to_tensor(fen).cuda().float()
                    distr           = distr.cuda().float()
                    evl             = evl.cuda().float()
                    distr_,evl_     = model(inp)

                    #Get accuracy
                    post_max            = distr_.max(dim=1)[1]
                    prior_max           = distr.max(dim=1)[1]
                    test_total          += inp.shape[0]
                    test_correct        += len(post_max[post_max==prior_max])

                    #Get error 
                    last_10_eval    = evl_.detach().cpu()[-1]
                    last_10_evl_    = torch.tensor([engine.analyse(chess.Board(fen=f),limit=chess.engine.Limit(time=1/32))['score'].white().score(mate_score=15300)/15_300 for f in fen[-1:]] )
                    err             = torch.nn.functional.mse_loss(last_10_eval,last_10_evl_).detach().cpu().item()
                    eval_list.append(err)
        
            test_evl.append(sum(eval_list)/len(eval_list))
            train_evl.append(sum(eval_list)/len(eval_list))
            test_acc.append(test_correct/test_total)
            train_acc.append(test_correct/test_total)

            while experience_set:


                #Create an experience batch
                breakoff            = experience_set[:games_bs]
                experience_set      = experience_set[games_bs:]
                positions_data      = prepare_batch(breakoff)
                #print(f"positions {sum([len(l) for l in breakoff])} -> {len(positions_data)}")

                #Prepare data 
                dataset             = ChessSet(positions_data)
                dataloader          = DataLoader(dataset,bs,True)

                #Prepare model
                model               = model.train()


                train_correct       = 0 
                train_total         = 0 
                eval_list           = []
                for batch in dataloader:

                    #Load data 
                    fen,distr,evl,n = batch
                    inp             = utilities.batched_fen_to_tensor(fen).cuda().float()
                    distr           = distr.cuda().float()
                    evl             = evl.cuda().float()

                    #Train 
                    distr_,evl_     = model(inp)

                    #Get loss
                    #Scale error by the frequency of the encounter 
                    err_scale       = (2- 2.718**(-.5*(n.mean().item()-2)))/2
                    p_loss          = torch.nn.functional.cross_entropy(distr_,distr)*err_scale
                    v_loss          = torch.nn.functional.mse_loss(evl_.flatten(),evl)*err_scale
                    comb_loss       = p_loss + v_loss

                    #Step model
                    optimizer.zero_grad()
                    comb_loss.backward()
                    optimizer.step()

                    #Get acc
                    post_max            = distr_.max(dim=1)[1]
                    prior_max           = distr.max(dim=1)[1]
                    train_correct       += len(post_max[post_max==prior_max])
                    train_total         += inp.shape[0]
                    #Get error 
                    last_10_eval    = evl_.detach().cpu()[-1]
                    last_10_evl_    = torch.tensor([engine.analyse(chess.Board(fen=f),limit=chess.engine.Limit(time=.1))['score'].white().score(mate_score=15300)/15_300  for f in fen[-1:]])
                    err             = torch.nn.functional.mse_loss(last_10_eval,last_10_evl_).detach().cpu().item()
                    eval_list.append(err)
                train_evl.append(sum(eval_list)/len(eval_list))

                model                   = model.eval()
                test_loader             = DataLoader(ChessSet(test_set),batch_size=1024)
                test_correct            = 0 
                test_total              = 0 
                eval_list               = []
                with torch.no_grad():
                    for batch in test_loader:
                        fen,distr,evl,n = batch
                        inp             = utilities.batched_fen_to_tensor(fen).cuda().float()
                        distr           = distr.cuda().float()
                        evl             = evl.cuda().float()
                        distr_,evl_     = model(inp)

                        #Get accuracy
                        post_max        = distr_.max(dim=1)[1]
                        prior_max       = distr.max(dim=1)[1]
                        test_total      += inp.shape[0]
                        test_correct    += len(post_max[post_max==prior_max])
                        #Get error 
                        last_10_eval    = evl_.detach().cpu()[-1]
                        last_10_evl_    = torch.tensor([engine.analyse(chess.Board(fen=f),limit=chess.engine.Limit(time=.1))['score'].white().score(mate_score=15300)/15_300  for f in fen[-1:]])
                        err             = torch.nn.functional.mse_loss(last_10_eval,last_10_evl_).detach().cpu().item()
                        eval_list.append(err)
                        
                    test_acc.append(test_correct/test_total)
                    train_acc.append(train_correct/train_total)
                    test_evl.append(sum(eval_list)/len(eval_list))

    
                plt.plot(test_acc,label='Test Acc')
                plt.plot(train_acc,label='Train Acc')
                plt.plot(test_evl,label='Test Err')
                plt.plot(train_evl,label='Train Err')
                plt.title("Accuracy vs. Train Iter")
                plt.legend()
                plt.draw()
                plt.pause(2)
                plt.clf()
                plt.cla()
                print(f"\tacc={test_acc[-1]:.4f}")
                torch.save(model.state_dict(),"curstate.pt")

                #Get it ready for the network
            plt.clf()
            plt.cla()


def check_model(model:ChessModel2):
    for file in os.listdir("C:/data/chess"):
        if not '.txt' in file or "2014" in file or "01" in file or "02" in file:
            continue
        file                    = "C:/data/chess/" + file
        print(f"train on {file}")
        experience_set          = json.loads(open(file,'r').read())
        model                   = model.cuda().eval().float()

        test_set                = prepare_batch(experience_set[:2048])
        ds                      = ChessSet(test_set)
        for item in ds:
            position,distr,evl,n    = item 
            if n < 10:
                continue
            #distr                   = numpy.zeros(shape=1968)
            prob,ind                = torch.sort(torch.from_numpy(distr),descending=True)
            top_moves               = [] 
            j = 0
            for p,i in zip(prob,ind):
                top_moves.append((utilities.I_TO_MOVE[i.item()].uci(),p.item()))
                j += 1 
                if j > 5:
                    break
            print(f"{position}-> {top_moves}")

            model_distr             = model(utilities.batched_fen_to_tensor([position]).cuda().float())[0][0].detach().cpu()
            prob,ind                = torch.sort(model_distr,descending=True)
            model_top_moves         = [] 
            j = 0
            for p,i in zip(prob,ind):
                model_top_moves.append((utilities.I_TO_MOVE[i.item()].uci(),str(p.item())[:6]))
                j += 1 
                if j > 5:
                    break
            input(f"{position}-> {model_top_moves}")


#trains with a list of experiences where each is (fen,outcome,probs)
def train_model(model:torch.nn.Module,exp_set:List[Tuple[str,str,Dict[str,int]]],prob_set,n_iters=1_000_000,bs:int=64,lr=.0001):

    t0                  = time.time()
    #Build dataset 
    ds                  = PositionSet(exp_set,prob_set)
    loader              = DataLoader(ds,batch_size=bs,shuffle=True,collate_fn=lambda x: (torch.tensor(numpy.array([it[0] for it in x])),[it[1] for it in x],torch.tensor(numpy.array([it[2] for it in x])), torch.tensor(numpy.array([it[3] for it in x ])),torch.tensor(numpy.array([it[4] for it in x ]))) )
    
    optimizer           = torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.95))
    for i,batch in enumerate(loader):

        optimizer.zero_grad()

        #Sample bs number of items 
        x       = batch[0].long().cuda()    
        probs   = batch[1]
        move_i  = batch[2].cuda().float()
        outcome = batch[3].unsqueeze(dim=-1).cuda().float()
        turn    = batch[4].unsqueeze(dim=-1).cuda().long()

        #real probs 
        real_probs  = torch.zeros(size=(x.size(0),len(data.move_to_i)))
        for i,game in enumerate(probs):
            for move_i in game:
                real_probs[i][move_i] = game[move_i]
        real_probs  = real_probs.float().cuda()
        real_probs  = real_probs + 1e-8
        real_probs  = real_probs / real_probs.sum(dim=1,keepdim=True)

        #Get predictions
        p,v     = model(x,turn)

        #Use KL_div
        log_preds   = torch.log(p+1e-9)
        loss_p      = torch.nn.functional.kl_div(log_preds,real_probs,reduction='batchmean') 
        loss_v      = torch.nn.functional.mse_loss(v,outcome)

        total_loss  = loss_p + loss_v
        total_loss.backward()

        optimizer.step()

        print(total_loss)

        #save every min 
        if time.time() - t0 > 60:
            torch.save(model.state_dict(),"E:/data/chess/models/top_model.pt")
            t0 = time.time()
            print(f"saved model")

    return 0



if __name__ == "__main__":
    #cmodel  = ChessModel2()
    #cmodel.load_state_dict(torch.load('curstate.pt'))
    #batch_trainer(cmodel,n_ep=8)
    #exit()
    #check_model(cmodel)
    #exit()
    #for fname in ["C:/data/chess/032015.pgn.zst","C:/data/chess/042015.pgn.zst","C:/data/chess/052015.pgn.zst","C:/data/chess/062015.pgn.zst",'C:/data/chess/072015.pgn.zst']:
    #    decompress_zst(fname)
    #exit()
    #create_probs_from_pgn("C:/data/chess/022015.pgn")
    #exit()
    all_data                = [] 
    #with multiprocessing.Pool(1) as pool:
    

    #Build the list of experiences
    args    = ["E:/data/chess/game_pgns/"+fname for fname in os.listdir("E:/data/chess/game_pgns") if ".pgn" in fname]
    experience_pool     = []
    for file in args[:20]:

        existing_fpath  = file.replace(".pgn","redo.txt").replace("game_pgns","game_files")
        if os.path.exists(existing_fpath):
            games   = load_games(existing_fpath)
        else:
            games   = get_experiences(file,saving_only=False)
        
        experience_pool += games

    #compute the probabilities given fen -> move
    print(f"Computing probabilities for {len(experience_pool)} positions")
    move_to_i           = {uci:i for i, uci in enumerate(json.loads(open("chessmoves.txt",'r').read()))}
    i_to_move           = {move_to_i[uci]:uci for uci in move_to_i}
    probabilities       = {}
    for position in experience_pool:
        fen,move,outcome,turn       = position 

        if not fen in probabilities:
            #Get legal moves 
            probabilities[fen]      = {move_to_i[m.uci()]:0 for m in list(chess.Board(fen=fen).generate_legal_moves())}

        probabilities[fen][move_to_i[move]] += 1


    print(f"found {len(probabilities)} unique positions")

        #print(f"{type(games)}->{games[0]}")
        #results = pool.imap(get_experiences,args)

    #     #experience_list     = get_experiences(f'C:/data/chess/game_pgns/{fname}')
    #     for experience_list in results:    
    #         all_data += experience_list
    # with open("C:/data/chess/all_games.txt",'w') as file:
    #     file.write(json.dumps(all_data))
    accuracies      = { #(torch.nn.RReLU,64,.0002,0,True):[],
                        #(torch.nn.PReLU,128,.0002,0,True):[],
                        #(torch.nn.PReLU,256,.0005,0,True):[],
                        #(ChessModel, 2048,.0001,.001,True):[],
                        #(ChessModel2,4096,.001,.01,True):[],
                        (ChessTransformer ,1024,.0001,.01):[],
                        (ChessModel2,1024,.0001,.01):[]}
    
    #training_set            = parse_probabilities(n_visits=8)
    #random.shuffle(training_set)
    #print(f"loaded {len(training_set)} datapoints")
    print(f"created {len(experience_pool)} experiences")
    random.shuffle(experience_pool)
    split_i         = int(.85 * len(experience_pool)) 
    training_set    = experience_pool[:split_i]
    test_set        = experience_pool[split_i:]

    print(f"train with {len(training_set)}\ttest with {len(test_set)}")

    for params in accuracies:
        mod                     = params[0]
        bs                      = params[1]
        lr                      = params[2]
        wd                      = params[3]
        chess_model             = mod().float().eval().cuda()

        acc                 = train_model(chess_model,training_set,probabilities,bs=bs)
        accuracies[params]  += acc

        plt.plot(accuracies[params],label=str(params).replace("<class 'torch.nn.modules.activation.",'').replace("'>",''))
    plt.xlabel("Batch number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()    
    torch.save(chess_model.state_dict(),"pram_train.pt")
    exit()