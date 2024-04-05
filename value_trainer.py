import torch 
import model 
import os 
import random 
import chess_utils
import json
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt  

class ChessDataset(Dataset):

    def __init__(self,positions,evals):

        self.positions  = positions 
        self.evals      = evals 
    
    def __getitem__(self,i):
        return (self.positions[i],self.evals[i])

    def __len__(self):
        return len(self.evals)


def build_dataset(data_path,bs=16,train_size=500,test_size=50):

    data_files      = [os.path.join(data_path,file) for file in os.listdir(data_path)]
    random.shuffle(data_files)

    train_positions = []
    train_evals     = [] 

    for file in data_files[:train_size]:
        contents            = open(file,'r').read()
        fen_to_score_list   = json.loads(contents)

        positions           = chess_utils.batched_fen_to_tensor([item[0] for item in fen_to_score_list])

        evals               = [torch.tensor(chess_utils.clean_eval(item[1]),dtype=torch.float32) for item in fen_to_score_list]

        train_positions += (positions)
        train_evals += (evals)

    test_positions = []
    test_evals     = [] 

    for file in data_files[train_size:train_size+test_size]:
        contents            = open(file,'r').read()
        fen_to_score_list   = json.loads(contents)

        positions           = chess_utils.batched_fen_to_tensor([item[0] for item in fen_to_score_list])
        evals               = [torch.tensor(chess_utils.clean_eval(item[1]),dtype=torch.float32) for item in fen_to_score_list]

        test_positions += (positions)
        test_evals += (evals)

    return DataLoader(ChessDataset(train_positions,train_evals),batch_size=bs,shuffle=True), DataLoader(ChessDataset(test_positions,test_evals),batch_size=bs,shuffle=True)


def clean(tsr):
        np  = tsr.detach().cpu().numpy()[:2]
        return [(str(item[0])+"    ")[:5] for item in np]


def train_v_dict(eval_model:model.ChessModel):
    eval_model.cuda().train()

    learning_params     = [{"name":"Train1_0","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_1","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_2","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_3","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_4","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_5","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_6","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_7","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           {"name":"Train1_8","ep":1,"lr":.00005,"betas":(.5,.6),"bs":4096,"trainset":(128,16)},
                           
                           {"name":"Train2_0","ep":1,"lr":.0001,"betas":(.5,.75),"bs":4096,"trainset":(128,16)},
                           {"name":"Train2_1","ep":1,"lr":.0001,"betas":(.5,.75),"bs":4096,"trainset":(128,16)},
                           {"name":"Train2_2","ep":1,"lr":.0001,"betas":(.5,.75),"bs":4096,"trainset":(128,16)},
                           {"name":"Train2_3","ep":1,"lr":.0001,"betas":(.5,.75),"bs":4096,"trainset":(128,16)},

                           {"name":"Train3_0","ep":1,"lr":.0001,"betas":(.5,.9),"bs":2048,"trainset":(256,16)},
                           {"name":"Train3_1","ep":1,"lr":.0001,"betas":(.5,.9),"bs":2048,"trainset":(256,16)}]
    
    #Track training
    train_losses        = [] 
    test_losses         = [] 
    

    for param_set in learning_params:
        print(f"\t\tBegin iter {param_set['name']}")
        optim               = torch.optim.AdamW(eval_model.parameters(),lr=param_set['lr'],betas=param_set['betas'],weight_decay=.01)       
        loss_fn             = torch.nn.MSELoss()

        #Generate data
        trainset,testset    = build_dataset("C:/data/chess/exps",bs=param_set['bs'],train_size=param_set['trainset'][0],test_size=param_set['trainset'][1])

        #Run training with params 
        for ep_num in range(param_set['ep']):
            

            #TRAIN
            eval_model.train()
            train_loss_sum  = 0
            for batch_num,batch in enumerate(trainset):
                
                #Zero
                eval_model.zero_grad()

                #Load data
                positions           = batch[0].cuda().float()
                eval                = batch[1].cuda().unsqueeze(dim=1).float()

                #Get eval
                p,ai_eval             = eval_model.forward(positions)

                #Train on delta
                eval_delta          = loss_fn(eval,ai_eval)
                train_loss_sum      += eval_delta
                eval_delta.backward()

                #Update params
                optim.step()

                
            


                

            train_losses.append(train_loss_sum/batch_num)


            #TEST
            eval_model.eval()
            with torch.no_grad():
                test_loss_sum       = 0
                for batch_num,batch in enumerate(testset):
                    
                    #Load data
                    positions           = batch[0].cuda().float()
                    eval                = batch[1].cuda().unsqueeze(dim=1).float()

                    #Get eval
                    p,ai_eval           = eval_model.forward(positions)

                    #Check error
                    test_loss_part      = loss_fn(eval,ai_eval).cpu().item()
                    test_loss_sum       += test_loss_part

                   


                test_losses.append(test_loss_sum/batch_num)
            
            reals   = clean(eval)
            fakes   = clean(ai_eval)

            printout    = f"[{reals[0]},{fakes[0]}]\t[{reals[1]},{fakes[1]}]"
            print(f"\t\t\t[{ep_num}/{param_set['ep']}]\ttrainerr: {train_losses[-1].detach().cpu().item():.3f}\ttesterr: {test_losses[-1]:.3f}\t{printout}")

        
            

    print(f"training complete")
    plt.plot([item.detach().cpu().numpy() for item in train_losses],label='train losses',color='orange')
    plt.plot(test_losses,label='test losses',color='blue')
    plt.legend()
    plt.show()

    #Save model params 
    torch.save(eval_model.state_dict(),"chess2modelparams.pt")


if __name__ == "__main__":
    eval_model          = model.ChessModel2(19,32).cuda()
    train_v_dict(eval_model=eval_model)
    