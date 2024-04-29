#Author: Everett Stenberg
#Description:   Contains various pytorch Neural Networks to act as the
#               board evaluation. ChessModel2 is the current one used



import torch
import chess_utils

class ChessModel(torch.nn.Module):


    def __init__(self,in_ch:int=19,n_channels:int=16,lin_act=torch.nn.PReLU,conv_act=torch.nn.PReLU,all_prelu=False,p=.5):

        super(ChessModel,self).__init__()
        #n_channels is set to 16 currently
        v_conv_n      = n_channels
        h_conv_n      = n_channels
        f_conv_n      = n_channels//2

        inter_sum     = v_conv_n+h_conv_n+f_conv_n

        if "PReLU" in str(conv_act) and all_prelu:
            act_kwargs      = {'num_parameters':v_conv_n}
        else:
            act_kwargs      = {} 



        #LAYER 1
        self.vconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,v_conv_n,kernel_size=(7+7+1,1),stride=1,padding=(7,0),bias=False),
                                                  torch.nn.BatchNorm2d(v_conv_n),
                                                  conv_act(**act_kwargs))

        self.hconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,h_conv_n,kernel_size=(1,7+7+1),stride=1,padding=(0,7),bias=False),
                                                  torch.nn.BatchNorm2d(h_conv_n),
                                                  conv_act(**act_kwargs))

        self.fullconv1      = torch.nn.Sequential(torch.nn.Conv2d(in_ch,f_conv_n,kernel_size=(7),stride=1,padding=3,bias=False),
                                                  torch.nn.BatchNorm2d(f_conv_n),
                                                  conv_act())


        #LAYER 2
        self.vconv2         = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,v_conv_n,kernel_size=(7+7+1,1),stride=1,padding=(7,0),bias=False),
                                                  torch.nn.BatchNorm2d(v_conv_n),
                                                  conv_act(**act_kwargs))

        self.hconv2         = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,h_conv_n,kernel_size=(1,7+7+1),stride=1,padding=(0,7),bias=False),
                                                  torch.nn.BatchNorm2d(h_conv_n),
                                                  conv_act(**act_kwargs))

        self.fullconv2      = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,f_conv_n,kernel_size=(7),stride=1,padding=3,bias=False),
                                                  torch.nn.BatchNorm2d(f_conv_n),
                                                  conv_act())


        #LAYER3
        self.vconv3         = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,v_conv_n,kernel_size=(7+7+1,1),stride=1,padding=(7,0),bias=False),
                                                  torch.nn.BatchNorm2d(v_conv_n),
                                                  conv_act(**act_kwargs))

        self.hconv3         = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,h_conv_n,kernel_size=(1,7+7+1),stride=1,padding=(0,7),bias=False),
                                                  torch.nn.BatchNorm2d(h_conv_n),
                                                  conv_act(**act_kwargs))

        self.fullconv3      = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,f_conv_n,kernel_size=(7),stride=1,padding=3,bias=False),
                                                  torch.nn.BatchNorm2d(f_conv_n),
                                                  conv_act())


        #FINAL LAYER
        self.final_conv     = torch.nn.Sequential(torch.nn.Conv2d(inter_sum,32,3,1,1,bias=False),
                                                  torch.nn.BatchNorm2d(32),
                                                  conv_act(),
                                                  torch.nn.Flatten(start_dim=1)
        )


        #P
        self.prob_head      = torch.nn.Sequential(torch.nn.Linear(32*8*8,1968),
                                                  torch.nn.Softmax(dim=1))

        #V
        self.val_head       = torch.nn.Sequential(torch.nn.Linear(32*8*8,512),
                                                  torch.nn.Dropout(p=p),
                                                  lin_act(),
                                                  torch.nn.Linear(512,1),
                                                  torch.nn.Tanh())


    def forward(self,x:torch.Tensor) -> torch.Tensor:

        #Round1
        v1                  = self.vconv1(x)
        h1                  = self.hconv1(x)
        f1                  = self.fullconv1(x)
        c1                  = torch.cat([v1,h1,f1],dim=1)

        #Round2
        v2                  = self.vconv2(c1)
        h2                  = self.hconv2(c1)
        f2                  = self.fullconv2(c1)
        c2                  = torch.cat([v2,h2,f2],dim=1)

        #Round3
        v3                  = self.vconv3(c2)
        h3                  = self.hconv3(c2)
        f3                  = self.fullconv3(c2)
        c3                  = torch.cat([v3,h3,f3],dim=1)

        #Final
        y                   = self.final_conv(c3)

        #Get outs
        return self.prob_head(y),self.val_head(y)


if __name__ == "__main__":
    import time
    torch.jit.enable_onednn_fusion(True)
    m       = ChessModel(19,24).float().eval()
    m = torch.jit.trace(m, [torch.randn(size=(16,19,8,8),device=torch.device('cpu'),dtype=torch.float32)])
    # Invoking torch.jit.freeze
    m = torch.jit.freeze(m)

    with torch.no_grad():
        t0  = time.time()
        for _ in range(800):
            inv     = torch.randn(size=(16,19,8,8),device=torch.device('cpu'),dtype=torch.float32,requires_grad=False)

            p,v       = m.forward(inv)

        print(f"out is {p.shape},{v.shape} in {(time.time()-t0):.3f}s")
