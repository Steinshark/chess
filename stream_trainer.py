import chess 
import torch 
import model 
import data 
from torch.utils.data import DataLoader
import utilities
from matplotlib import pyplot as plt 


EPOCHS          = 128
BS              = 8192
LR              = .0001
BETAS           = (.95,.999)
WD              = 0.0
MODEL           = model.ChessModel2().cuda().float()
optimizer       = torch.optim.AdamW(MODEL.parameters(),LR,betas=BETAS,weight_decay=WD,amsgrad=True)
#optimizer       = torch.optim.SGD(MODEL.parameters(),lr=LR,weight_decay=WD,momentum=.1)

START_POS       = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",utilities.batched_fen_to_tensor(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]).float().cuda()
MATE_IN_1_W     = "r1bqkbnr/2pp1ppp/1pn5/p3N3/P3P3/2PP1QP1/5PKP/8 w kq - 1 19", utilities.batched_fen_to_tensor(["r1bqkbnr/2pp1ppp/1pn5/p3N3/P3P3/2PP1QP1/5PKP/8 w kq - 1 19"]).float().cuda()
MATE_IN_1_B     = "r1b1k1nr/2pp1ppp/1pn5/pBb1N3/4P2q/P7/1PPP1PPP/RNBQK2R b KQkq - 0 8", utilities.batched_fen_to_tensor(["r1b1k1nr/2pp1ppp/1pn5/pBb1N3/4P2q/P7/1PPP1PPP/RNBQK2R b KQkq - 0 8"]).float().cuda()
DRAWN_ENDGAME   = "8/5k1p/8/8/5K2/7P/8/8 b - - 2 42", utilities.batched_fen_to_tensor(["8/5k1p/8/8/5K2/7P/8/8 b - - 2 42"]).float().cuda()

def stream_train(path):

    #Load data
    train_ds            = data.StreamDS("C:/gitrepos/chess/ds_template_train2.txt")
    test_ds             = data.StreamDS("C:/gitrepos/chess/ds_template_test2.txt")
    print(f"model_params=\t{sum(p.numel() for p in MODEL.parameters())/1_000_000:.3f}M")
    print(f"train_len=\t{len(train_ds.data)}")
    print(f"test_len=\t{len(test_ds.data)}")
    train_dl            = DataLoader(train_ds,BS)
    test_dl             = DataLoader(test_ds,len(test_ds.data))
    train_ds.size       = 1024*128

    train_eval_loss     = [] 
    test_eval_loss      = [] 
    train_accuracies    = [] 
    test_accuracies     = [] 

    plt.ion()
    plt.show()

    for ep in range(EPOCHS):

        #TRAIN 
        MODEL.train()
        for i,batch in enumerate(train_dl):

            optimizer.zero_grad()
            reprs,distros,evals     = data.process_batch(batch)
            distros_,evals_         = MODEL(reprs)

            #Do loss
            p_loss                  = torch.nn.functional.cross_entropy(distros_,distros)
            v_loss                  = torch.nn.functional.mse_loss(evals_.flatten(),evals)
            #t_loss                  = p_loss+v_loss
            t_loss                  = v_loss

            #Find accuracy 
            train_eval_loss.append(v_loss.detach().cpu().item())
            train_accuracies.append(data.compute_accuracy(distros,distros_))
            t_loss.backward()

            #Clip 
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(),1)
            optimizer.step()
        
        #TEST 
        MODEL.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dl):

                reprs,distros,evals     = data.process_batch(batch)
                distros_,evals_         = MODEL(reprs)
                v_loss                  = torch.nn.functional.mse_loss(evals_.flatten(),evals)
                test_eval_loss.append(v_loss.detach().cpu().item())
                test_accuracies.append(data.compute_accuracy(distros,distros_))
            for test in [START_POS,MATE_IN_1_W,MATE_IN_1_B,DRAWN_ENDGAME]:
                fen,repr                = test
                p,v                     = MODEL(repr)
                print(data.observe_predictions(p[0],top_n=4,parse_legal=fen), str(v.detach().cpu().item())[:5])
            print("accuracy:\t",test_accuracies[-1])
            print("evaluation:\t",test_eval_loss[-1],"\n\n")
            plt.plot(utilities.reduce_arr(test_accuracies,newlen=64),label='test accuracy',color='dodgerblue')
            plt.plot(utilities.reduce_arr(train_accuracies,newlen=64),label='train accuracy',color='green')
            plt.plot(utilities.reduce_arr(train_eval_loss,newlen=64),label='train error',color='gold')
            plt.plot(utilities.reduce_arr(test_eval_loss,newlen=64),label='test error',color='goldenrod')
            plt.title("Accuracy vs. Train Iter")
            plt.legend()
            plt.draw()
            plt.pause(2)
            plt.clf()
            plt.cla()
    
    input("done trianing")
            


if __name__ == "__main__":
    stream_train("None")
    input("done trianing")
