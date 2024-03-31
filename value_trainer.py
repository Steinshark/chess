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

    
def clean_eval(evaluation):
    
    if evaluation > 0:
        return min(1500,evaluation) / 1500
    else:
        return max(-1500,evaluation) / 1500


def build_dataset(data_path,bs=16,train_size=500,test_size=50):

    data_files      = [os.path.join(data_path,file) for file in os.listdir(data_path)]
    random.shuffle(data_files)

    train_positions = []
    train_evals     = [] 

    for file in data_files[:train_size]:
        contents            = open(file,'r').read()
        fen_to_score_list   = json.loads(contents)

        positions           = chess_utils.batched_fen_to_tensor([item[0] for item in fen_to_score_list])

        evals               = [torch.tensor(clean_eval(item[1]),dtype=torch.float32) for item in fen_to_score_list]

        train_positions += (positions)
        train_evals += (evals)

    test_positions = []
    test_evals     = [] 

    for file in data_files[train_size:train_size+test_size]:
        contents            = open(file,'r').read()
        fen_to_score_list   = json.loads(contents)

        positions           = chess_utils.batched_fen_to_tensor([item[0] for item in fen_to_score_list])
        evals               = [torch.tensor(clean_eval(item[1]),dtype=torch.float32) for item in fen_to_score_list]

        test_positions += (positions)
        test_evals += (evals)

    return DataLoader(ChessDataset(train_positions,train_evals),batch_size=bs,shuffle=True), DataLoader(ChessDataset(test_positions,test_evals),batch_size=bs,shuffle=True)


def train_v_dict(eval_model:model.ChessModel):
    eval_model.cuda().train()

    learning_params     = [{"name":"Train1_0","ep":1,"lr":.00002,"betas":(.5,.75),"bs":4096,"trainset":(32,8)},
                           {"name":"Train1_1","ep":1,"lr":.00002,"betas":(.5,.75),"bs":4096,"trainset":(32,8)},
                           {"name":"Train1_2","ep":1,"lr":.00002,"betas":(.5,.75),"bs":4096,"trainset":(32,8)},
                           {"name":"Train1_3","ep":1,"lr":.00002,"betas":(.5,.75),"bs":4096,"trainset":(32,8)},
                           
                           {"name":"Train2_0","ep":1,"lr":.0001,"betas":(.5,.9),"bs":4096,"trainset":(128,16)},
                           {"name":"Train2_1","ep":1,"lr":.0001,"betas":(.5,.9),"bs":4096,"trainset":(128,16)},
                           {"name":"Train2_2","ep":1,"lr":.0001,"betas":(.5,.9),"bs":4096,"trainset":(128,16)},
                           {"name":"Train2_3","ep":1,"lr":.0001,"betas":(.5,.9),"bs":4096,"trainset":(128,16)},

                           {"name":"Train3_0","ep":1,"lr":.0002,"betas":(.5,.999),"bs":1024,"trainset":(16,4)},
                           {"name":"Train3_1","ep":1,"lr":.0002,"betas":(.5,.999),"bs":1024,"trainset":(16,4)}]
    
    #Track training
    train_losses        = [] 
    test_losses         = [] 
    

    for param_set in learning_params:
        print(f"\t\tBegin iter {param_set['name']}")
        optim               = torch.optim.Adam(eval_model.parameters(),lr=param_set['lr'],betas=param_set['betas'])       
        loss_fn             = torch.nn.MSELoss()

        #Generate data
        trainset,testset    = build_dataset("C:/gitrepos/chess/exps",bs=param_set['bs'],train_size=param_set['trainset'][0],test_size=param_set['trainset'][1])

        #Run training with params 
        for ep_num in range(param_set['ep']):
            

            #TRAIN
            eval_model.train()
            train_loss_sum  = 0
            checkpoints     = [int(len(trainset)*.2*n) for n in range(5)]
            for batch_num,batch in enumerate(trainset):
                
                #Zero
                eval_model.zero_grad()

                #Load data
                positions           = batch[0].cuda()
                eval                = batch[1].cuda().unsqueeze(dim=1)

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
                checkpoints     = [int(len(trainset)*.2*n) for n in range(5)]
                for batch_num,batch in enumerate(testset):
                    
                    #Load data
                    positions           = batch[0].cuda()
                    eval                = batch[1].cuda().unsqueeze(dim=1)

                    #Get eval
                    p,ai_eval           = eval_model.forward(positions)

                    #Check error
                    test_loss_part      = loss_fn(eval,ai_eval).cpu().item()
                    test_loss_sum       += test_loss_part

                   


                test_losses.append(test_loss_sum/batch_num)
            
            print(f"\t\t\t[{ep_num}/{param_set['ep']}]\ttrainerr: {train_losses[-1].detach().cpu().item():.3f}\ttesterr: {test_losses[-1]:.3f}\t[{eval[0].detach().cpu().item():.2f}->{ai_eval[0].detach().cpu().item():.2f}]")

            
            

    print(f"training complete")
    plt.plot([item.detach().cpu().numpy() for item in train_losses],label='train losses',color='orange')
    plt.plot(test_losses,label='test losses',color='blue')
    plt.legend()
    plt.show()

    #Save model params 
    torch.save(eval_model.state_dict(),"chessmodelparams.pt")



        


if __name__ == "__main__":
    eval_model          = model.ChessModel(15).cuda()

    optim               = torch.optim.Adam(eval_model.parameters(),lr=.0002,betas=(.5,.75))
    loss_fn             = torch.nn.MSELoss()

    test_losses         = [] 
    train_losses        = [] 

    trainset,testset    = build_dataset("C:/data/chess/exps",bs=4096)

    for ep_num in range(10):

        #TRAIN PORTION
        train_loss_sum      = 0
        for batch_num,batch in enumerate(trainset):
            eval_model.zero_grad()

            positions           = batch[0].cuda()
            eval                = batch[1].cuda().unsqueeze(dim=1)

            p,ai_eval             = eval_model.forward(positions)

            eval_delta          = loss_fn(eval,ai_eval)
            train_loss_sum      += eval_delta
            eval_delta.backward()

            optim.step()
        train_losses.append(train_loss_sum/batch_num)

        with torch.no_grad():
            eval_model.eval()
            test_loss_sum       = 0
            
            for batch_num,batch in enumerate(testset):

                positions           = batch[0].cuda()
                eval                = batch[1].cuda().unsqueeze(dim=1)

                p,ai_eval             = eval_model.forward(positions)
                test_loss_sum       += loss_fn(eval,ai_eval).cpu().item()
                #print(f"actual:{eval[0]},pred:{ai_eval[0]}")
            test_losses.append(test_loss_sum/batch_num)
            print(f"test[{ep_num}] losses={test_losses[-1]:.4f} ({eval[0].cpu().item()},{ai_eval[0].cpu().item():.1f})")


    plt.plot([item.detach().cpu().numpy() for item in train_losses],label='train losses',color='orange')
    plt.plot(test_losses,label='test losses',color='blue')
    plt.legend()
    plt.show()

    #Save model params 
    # if "y" in input("save?"):
    #     torch.save(rollm.state_dict(),"C:/gitrepos/ChessEngine/modelparams.pt")
    torch.save(rollm.state_dict(),"C:/gitrepos/ChessEngine/chessmodelparams.pt")
