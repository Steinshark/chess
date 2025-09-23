import os 
import random 
import json 
import bulletchess 
import utilities
import torch 
import multiprocessing
import math 
import numpy
from torch.utils.data import Dataset


#Load chess moves 
move_to_i           = {uci:i for i, uci in enumerate(json.loads(open("chessmoves.txt",'r').read()))}
i_to_move           = {move_to_i[uci]:uci for uci in move_to_i}
class probPreSet:

    def __init__(self,move_visits):

        self.data           = [(position,utilities.movecount_to_prob(move_count)) for position, move_count in move_visits]

    
    def __getitem__(self,i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)



class ChessSet:

    def __init__(self,positions_map):

        self.data           = [(position,utilities.normalize_numpy(pack[0]),pack[1]/pack[0].sum(),pack[0].sum()) for position,pack in positions_map.items()]
        random.shuffle(self.data)
    def __getitem__(self,i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)
    

class PositionSet(Dataset):

    def __init__(self,positions,probs):

        self.data   = positions 
        self.probs  = probs

    def __getitem__(self,i):
        outcomes                = {'1-0':1,'0-1':-1,'1/2-1/2':0}
        fen,move,turn,outcome   = self.data[i]
        x                       = to_64_len_str(bulletchess.Board(fen))
        probs                   = self.probs[fen]
        move_i                  = move_to_i[move]
        outcome                 = outcomes[outcome]
        return x, probs, move_i, outcome, int(turn) 
    
    def __len__(self):
        return len(self.data)



class StreamDS:

    def __init__(self,path="C:/data/chess/game_data/",n_games=1_000_000,elo_lim=2500,size=1024*1024):
        self.data           = combine_ds(path) if not 'template' in path else json.loads(open(path,'r').read())
        self.counts         = [sum(x[1].values()) for x in self.data]
        self.total_counts   = sum(self.counts)
        self.size           = len(self.data) * 1
    
    def __getitem__(self,i):


        #Get random datapoint 
        next_i          = random.choice(range(len(self.data)))
        next_count      = self.counts[next_i]

        while random.random() < math.sqrt((next_count)/self.total_counts):
            next_i          = random.choice(range(len(self.data)))
            next_count      = self.counts[next_i]

        fen,moves,eval  = self.data[next_i]

        #Find the fen
        fen             = fen
        
        #Build distribution
        distribution    = torch.zeros(1968)
        for move in moves:
            distribution[utilities.MOVE_TO_I[bulletchess.Move.from_uci(move)]]    += 1
        distribution    = utilities.normalize_torch(distribution)

        #Return items
        return fen, distribution, eval / next_count
        

    def __len__(self):
        return self.size


def tokenize_fen(board:bulletchess.Board,req_grad:bool):

    #Check for castling
    parts               = board.fen().split(" ")
    castle_code         = int("K" in parts[2])*8 + int("Q" in parts[2])*4 + int("k" in parts[2])*2 + int('q' in parts[2])
    turn_code           = 0 if parts[1] == 'w' else 1

    #Get postion    
    keys                = "-pnbrqkPNBRQK"
    board_str           = board.__str__().replace("\n","").replace(" ","") 
    board_tokens        = [keys.index(p) for p in board_str] + [castle_code+len(keys),turn_code+len(keys)+16]
    tokens              = torch.tensor(board_tokens,dtype=torch.long)
    return tokens

def parse_gamefile_line(line_text:str,elo_limit=2000,time_limit=180):
    board                           = bulletchess.Board()
    if not line_text:
        return []
    movetext,welo,belo,tc,outcome      = line_text.split(".") 
    if "1-0" in outcome:
        outcome = 1
    elif '0-1' in outcome:
        outcome = -1
    else:
        outcome = 0
    

    if int(welo) > elo_limit and int(belo) > elo_limit and int(tc) > time_limit:
        return [(board.fen(),board.push_san(move).uci(),outcome) for move in movetext.split(",")]

    else:
        return []
    

def create_ds_from_redo(path:str,save_thresh=32768,elo_limit=1950,time_limit=180):

    games           = []
    filenames       = [os.path.join(path,fname) for fname in os.listdir(path)]

    with multiprocessing.Pool(8) as pool:
        for file in filenames:
            with open(file,'r') as file:

                ggs           = pool.imap(parse_gamefile_line,file.readlines())
            
            for newgame in ggs:
                #for line in file:
                #    newgame     = parse_gamefile_line(line,elo_limit=elo_limit,time_limit=time_limit)
                #    #input(newgame)
                    games       += [position + (i+1,len(newgame)) for i, position in enumerate(newgame)]

                    if len(games) > save_thresh:
                        fname           = "C:/data/chess/game_data/" + str(random.randint(10_000,99_999)) + '.txt'
                        while os.path.exists(fname):
                            fname           = "C:/data/chess/game_data/" + str(random.randint(10_000,99_999)) + '.txt'

                        with open(fname,'w') as file:
                            file.write(json.dumps(games))
                            games       = []


        file.close()

    return
            

def process_batch(batch):
    fens,distrs,evals   = batch 
    reprs               = utilities.batched_fen_to_tensor(fens).float().cuda()
    distrs              = distrs.float().cuda()
    #input(evals)
    evals               = evals.float().cuda()

    return reprs,distrs,evals


def test_accuracy(model,data):
    return


def combine_ds(path='C:/data/chess/game_data/',min_count=3,prune_every=10_000_000,flip_p=.0001)->tuple[str,dict[str,int],int]:
    positions   = {} 
    evals       = {}

    datapoints  = [] 
    counter     = 0 

    for fname in os.listdir(path):
        filename = path + fname
        with open(filename,'r') as file:
            gamelists   = json.loads(file.read())

            for pack in gamelists:
                counter += 1
                #print(pack)
                fen,move,eval,move_i,n_moves    = pack 
                
                if fen in positions:
                    evals[fen] += eval
                    if move in positions[fen]:
                        positions[fen][move] += 1
                    else:
                        positions[fen][move] = 1
                else:
                    evals[fen]      = eval
                    positions[fen]  = {move:1,'flag':0}

                    #Keep with odds relative to distance to end of game 
                    moves_from_end  = n_moves - move_i
                    if random.random() < 2.718**(-.4*moves_from_end):
                        positions[fen]['flag'] = -999999
                

                if (counter % prune_every) == 0 and not counter == 0:

                    #Collect items to delete here
                    markdel     = []

                    #Scan the current dataset
                    for fen in positions:
                        visits          = sum(positions[fen].values()) - positions[fen]['flag']
                        
                        #Flip flag for approx .01% of the items 
                        if random.random() < flip_p:
                            positions[fen]['flag'] -= 1

                        #Remove items that are flagged and less than min count
                        if visits < 2:
                            if positions[fen]['flag'] == 1:
                                markdel.append(fen)
                            else:
                                positions[fen]['flag'] += 1

                    print(f"{len(positions)}->",end='')
                    for item in markdel:
                        del positions[item]
                    print(f"{len(positions)}")

    for position in positions:
        if (sum(positions[position].values()) - positions[position]['flag']) > min_count and not positions[position]['flag'] < -99:
            #Remove flag 
            del positions[position]['flag']
            datapoints.append((position,positions[position],evals[position]))
    
    #print(f"created {len(datapoints)} data")
    with open("ds_template.txt",'w') as file:
        file.write(json.dumps(datapoints))
    file.close()
    return datapoints
    
    
def compute_accuracy(targets:torch.Tensor,predictions:torch.Tensor)->float:


    max_target_indices      = torch.max(targets,dim=1)[1]
    max_pred_indices        = torch.max(predictions,dim=1)[1]

    return len(max_target_indices[max_target_indices==max_pred_indices]) / max_pred_indices.shape[0]


#Parse legal with either be 'False' or it will be the fen of the position
def observe_predictions(predictions:torch.Tensor,top_n=10,parse_legal=False)->list[tuple[str,float]]:
    
    if parse_legal:
        gameboard   = bulletchess.Board(fen=parse_legal)
        moves       = list(gameboard.generate_legal_moves())
    move_decisions  = []
    for i,p in enumerate(predictions):
        if not parse_legal:
            move_decisions.append((utilities.I_TO_MOVE[i].uci(),str(p.detach().cpu().item())[:9]))
        else:
            if utilities.I_TO_MOVE[i] in moves:
                move_decisions.append((utilities.I_TO_MOVE[i].uci(),str(p.detach().cpu().item())[:9]))


    
    move_decisions.sort(reverse=True,key=lambda x:x[1])

    return move_decisions[:top_n]


def parse_puzzles(puzzle_file:str):
    with open("C:/data/chess/puzzles/puzzles.csv",'r') as file:
        file.readline()

        experiences     = [] 

        for puzzle in file.readlines():
            PuzzleId,fen,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags = puzzle.split(",")

            this_puzzle     = [] 
            board           = bulletchess.Board(fen=fen)
            moves           = Moves.split(' ')
            puzzle_turn     = not board.turn

            #Play first move (Bad)
            board.push(bulletchess.Move.from_uci(moves.pop(0)))
            record  = True 
            for move in moves:
                if record:
                    this_puzzle.append((board.fen(),{move:1},'?'))
                
                board.push(bulletchess.Move.from_uci(move))
                record = not record

            #Check puzzle outcome 
            if board.result() == '1-0':
                this_puzzle = [(item[0],item[1],1) for item in this_puzzle]
            elif board.result() == '0-1':
                this_puzzle = [(item[0],item[1],-1) for item in this_puzzle]
            elif board.result() == '1/2-1/2':
                this_puzzle = [(item[0],item[1],0) for item in this_puzzle]
            else:
                #Apply random score in favor of puzzler 
                r_score     = (.5 + (.2* random.random())) * (1 if puzzle_turn else -1)
                #print(f"R score was {r_score}")
                this_puzzle = [(item[0],item[1],r_score) for item in this_puzzle]
            #print(Moves)    
            #print(board.result())
            #print(board)
            #input(this_puzzle)
            experiences += this_puzzle
    return experiences


if __name__ == "__main__":
    pds     = parse_puzzles("C:/data/chess/puzzles/puzzles.csv")
    print(f"created {len(pds)} datapoints")
    #exit()
    #create_ds_from_redo("C:/data/chess/game_files",save_thresh=131072,elo_limit=1900,time_limit=180)   
    ds      = combine_ds(prune_every=25_000_000)
    ds      = ds + pds
    random.shuffle(ds)
    print(len(ds))
    test    = ds[:4096]
    train   = ds[4096:]
    with open("C:/gitrepos/chess/ds_template_train2.txt",'w') as trainfile, open("C:/gitrepos/chess/ds_template_test2.txt",'w') as testfile:
        trainfile.write(json.dumps(train))
        testfile.write(json.dumps(test))

    exit()
