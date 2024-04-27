#Author: Everett Stenberg
#Description:   trains an initial version of the network so a s
#               to reduce overhead



import chess
import chess.engine
import chess_utils
from model import ChessModel
import numpy 
import torch
import trainer
from torch.utils.data import DataLoader
import random 
import multiprocessing
import os 
import random
import time 
import json 
from matplotlib import pyplot as plt  


class probPreSet:

    def __init__(self,move_visits):
        self.positions      = [] 
        self.distributions  = [] 

        for position, move_count in move_visits:
            self.positions.append(position)
            self.distributions.append(chess_utils.movecount_to_prob(move_count))
    
    def __getitem__(self,i):
        return (self.positions[i],self.distributions[i])
    
    def __len__(self):
        return len(self.distributions)


def parse_multiple(gametest_lists):
    common_dict         = []

    list_of_lists       = [parse_movelist(gametext) for gametext in gametest_lists]
    common_dict         = [item for single_list in list_of_lists for item in single_list]

    return common_dict


def parse_movelist(gametext):
    if "eval" in gametext:
        return {}
     
    split_string        = '\n\n['
    move_list           = gametext.replace(split_string,'').split(' ')[:-1]

    micro_pos_map       = []
    board               = chess.Board()
    move_list
    move_list           = [move for move in move_list if not "." in move]

    for move in move_list:
        #Get key
        key             = " ".join(board.fen().split(" ")[:4])
        move            = board.parse_san(move)

        #Update dict 
        micro_pos_map.append((key,move.uci(),list(board.generate_legal_moves())))
        board.push(move)
        
    return micro_pos_map


def train_probs(model:ChessModel,dataset:probPreSet,testset:probPreSet,lr=.001,bs=1024,betas=(.5,.9),wd=.01):

    #Prep model
    model.train().float()
    optimizer               = torch.optim.Adam(model.parameters(),lr=lr,betas=betas,weight_decay=wd)
    #optimizer               = torch.optim.SGD(model.parameters(),lr=lr,momentum=.5,nesterov=True)
    accuracies              = [] 
    
    dataloader              = DataLoader(dataset,batch_size=bs,shuffle=True)
    testloader              = DataLoader(testset,batch_size=len(testset),shuffle=False)

    for batch_i,batch in enumerate(dataloader):
        optimizer.zero_grad()

        position            = chess_utils.batched_fen_to_tensor(batch[0]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
        post_probs          = batch[1].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
        prior_probs,evals   = model.forward(position)

        loss                = torch.nn.functional.cross_entropy(prior_probs,post_probs)
        loss.backward()
        optimizer.step()


        #get testloss
        for batch in testloader:
            with torch.no_grad():
                model.eval()
                position            = chess_utils.batched_fen_to_tensor(batch[0]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
                post_probs          = batch[1].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
                prior_probs,evals   = model.forward(position)
                post_max        = post_probs.max(dim=1)[1]
                prior_max       = prior_probs.max(dim=1)[1]
                accuracy        = len(post_max[post_max==prior_max]) / position.shape[0]

                if bs < 1024:
                    if (batch_i*bs) % 1024 == 0:
                        accuracies.append(accuracy) 
                else:
                    for _ in range(bs//1024):
                        accuracies.append(accuracy)
                model.train()
            

    return accuracies 
    

def create_probs_from_pgn(pgn_file:str):


    #position_map            = {}
    position_map            = json.loads(open("positions.dict",'r').read())
    startfne    = " ".join(chess.Board().fen().split(" ")[:4])
    #Open file and parse all games into their moves
    with open(pgn_file,'r') as file:


        #Parse a game at a time
        #   start with [Event...]
        #   end with  
        
        #games           = file.read().split("Event")
        for _ in range(50_000_000):
            file.__next__()
        games           = "".join([file.__next__() for _ in range(50_000_000)]).split("Event")
        print(f"Read games")
        split_2         = '\n\n'
        games           = [gametext for gametext in games if "\n\n1." in gametext]
        games           = [gametext.split(split_2)[1] for gametext in games]
        print(f"Parsed {len(games)} games")

        move_lists      = [] 
        window_size     = 10_000
        i               = 0 
        while games:
            t0 = time.time()

            #Grab window number of games
            positions   = games[:window_size]
            games       = games[window_size:]

            #Process them
            window      = 1000
            import math 
            # work_chunks = [] 
            # while positions:
            #     work_chunks += positions[:window]
            #     positions   = positions[window:]
            work_chunks = [positions[i*window:(i+1)*window] for i in range(math.ceil(window_size/window)) if len(positions[i*window:(i+1)*window]) == window]
            with multiprocessing.Pool(8) as pool:
                move_lists  = pool.map(parse_multiple,work_chunks)
            pool.close()

            #game_dicts     = map(parse_movelist,positions)

            #Add them to the main dict 
            for move_list in move_lists:
                for pack in move_list:
                    position,move,moves  = pack

                    if position in position_map:
                        position_map[position][move] += 1 

                    else:
                        position_map[position] = {m.uci():0 for m in moves}
                        position_map[position][move] = 1

            #Print results
            i += window_size
            print(f"{i}/{len(games)} in {(time.time()-t0):.2f}s")
            resdict     = {m: c for m,c in position_map[startfne].items()}
            print(f"{resdict}")
            old_len         = len(list(position_map.keys()))
            print(f'Total {old_len}',end="->")

            #Prune search
            markdel         = []
            for key in position_map:
                total = 0 
                for move,count in position_map[key].items():
                    total += count 
                if total < 2:
                    markdel.append(key)
            
            for key in markdel:
                del position_map[key]
            new_len         = len(list(position_map.keys()))
            print(f'{new_len} keys\n\n')
            with open("positions.dict","w") as writefile:
                writefile.write(json.dumps(position_map))
            writefile.close()


        
        
    
        print(f"parsed {len(move_lists)} games")
   

def parse_probabilities(n_visits=3):
    position_map            = json.loads(open("positions.dict",'r').read())
    training_data           = [(position,position_map[position]) for position in position_map if sum([v for m,v in position_map[position].items()]) > n_visits] 
    print(f"found {len(training_data)} positions")
    return training_data


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


def pretrain(chess_model:ChessModel,exps,bs=4096,lr=.001,wd=.01,betas=(.5,.75)):

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

    #create_probs_from_pgn("C:/users/evere/Downloads/March2016.pgn")

    accuracies      = { (torch.nn.RReLU,64,.0002,0,True):[],
                        (torch.nn.PReLU,512,.0005,0,True):[],
                        (torch.nn.PReLU,1024,.0005,0,True):[],
                        (torch.nn.PReLU,2048,.001,0,True):[]}
    
    training_set            = parse_probabilities()
    print(f"loaded {len(training_set)} datapoints")


    testing_set     = training_set[-16384:]
    training_set    = training_set[:-16384]
    print(f"test: {len(testing_set)}\t train: {len(training_set)}")

    dataset         = probPreSet(training_set)
    testset         = probPreSet(testing_set)

    for params in accuracies:
        act                     = params[0]
        bs                      = params[1]
        lr                      = params[2]
        wd                      = params[3]
        all_cn                  = params[4]
        chess_model             = ChessModel(19,16,lin_act=act,conv_act=act,all_prelu=all_cn).float().eval()
        
        print(f"test {params}")
        acc                 = train_probs(chess_model,dataset,testset,lr=lr,bs=bs,wd=wd,betas=(.5,.99))
        accuracies[params]  += acc

        plt.plot(accuracies[params],label=str(params).replace("<class 'torch.nn.modules.activation.",'').replace("'>",''))
    plt.xlabel("Batch number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()    
        
    exit()
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
