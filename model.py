#Author: Everett Stenberg
#Description:   Contains various pytorch Neural Networks to act as the
#               board evaluation. ChessModel2 is the current one used



import torch
import settings


class ChessBlock(torch.nn.Module):

    def __init__(self,in_ch:int,v_conv_ch,h_conv_ch,f_conv_ch):
        super(ChessBlock,self).__init__()
        
        self.vconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,v_conv_ch,kernel_size=(7+7+1,1),stride=1,padding=(7,0),bias=False),
                                                  torch.nn.BatchNorm2d(v_conv_ch),
                                                  torch.nn.PReLU(v_conv_ch))

        self.hconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,h_conv_ch,kernel_size=(1,7+7+1),stride=1,padding=(0,7),bias=False),
                                                  torch.nn.BatchNorm2d(h_conv_ch),
                                                  torch.nn.PReLU(h_conv_ch))

        self.fullconv1      = torch.nn.Sequential(torch.nn.Conv2d(in_ch,f_conv_ch,kernel_size=(7),stride=1,padding=3,bias=False),
                                                  torch.nn.BatchNorm2d(f_conv_ch),
                                                  torch.nn.PReLU(f_conv_ch))
        
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        #Pass
        v_conv              = self.vconv1(x)
        h_conv              = self.hconv1(x)
        f_conv              = self.fullconv1(x)

        #Restack
        return torch.cat([v_conv,h_conv,f_conv],dim=1)


class ChessModel(torch.nn.Module):


    def __init__(self,repr_ch:int=17,conv_ch:int=16,lin_act=torch.nn.PReLU,conv_act=torch.nn.PReLU,all_prelu=False,p=.5):

        super(ChessModel,self).__init__()
        
        #n_channels is set to 16 currently
        v_conv_n      = conv_ch
        h_conv_n      = conv_ch
        f_conv_n      = conv_ch//2

        inter_sum     = v_conv_n+h_conv_n+f_conv_n

        if "PReLU" in str(conv_act) and all_prelu:
            act_kwargs      = {'num_parameters':v_conv_n}
        else:
            act_kwargs      = {} 



        #Chess Layers
        self.layer1         = ChessBlock(repr_ch,16,16,32)

        self.layer2         = ChessBlock(16+16+32,16,16,32)

        self.layer3         = ChessBlock(16+16+32,16,16,32)

        self.layer4         = ChessBlock(16+16+32,16,16,32)

        #Standard layers    in=(bsx64x8x8)
        self.layer5         = torch.nn.Sequential(torch.nn.Conv2d(16+16+32,64,3,1,0,bias=False),
                                                  torch.nn.BatchNorm2d(64),
                                                  torch.nn.PReLU(64))
        #                   in=(bsx48x7x7)
        self.layer6         = torch.nn.Sequential(torch.nn.Conv2d(64,128,3,1,0,bias=False),
                                                  torch.nn.BatchNorm2d(128),
                                                  torch.nn.PReLU(128))



        #Final Layer        in=(bsx32x6x6)
        self.layer7         = torch.nn.Sequential(torch.nn.Conv2d(128,256,3,1,0,bias=False),
                                                  torch.nn.BatchNorm2d(256),
                                                  conv_act(),
                                                  torch.nn.Flatten(start_dim=1))


        #P
        self.prob_head      = torch.nn.Sequential(torch.nn.Linear(256*2*2,1968),
                                                  torch.nn.Softmax(dim=1))

        #V
        self.val_head       = torch.nn.Sequential(torch.nn.Linear(256*2*2,256),
                                                  torch.nn.Dropout(p=p),
                                                  lin_act(),
                                                  torch.nn.Linear(256,1),
                                                  torch.nn.Tanh())

        self.to(settings.DTYPE)


    def forward(self,x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:

        #Round1
        x                   = self.layer1(x)
        x                   = self.layer2(x)
        x                   = self.layer3(x)
        x                   = self.layer4(x)
        x                   = self.layer5(x)
        x                   = self.layer6(x)
        x                   = self.layer7(x)
        #Get outs
        return self.prob_head(x),self.val_head(x)


if __name__ == "__main__":
    import time
    dev     = torch.device('cuda')
    torch.jit.enable_onednn_fusion(False)
    torch.backends.cudnn.enabled= True
    m       = ChessModel(17).float().eval().to(dev)
    m = torch.jit.trace(m, [torch.randn(size=(16,17,8,8),device=dev,dtype=torch.float32)])
    # Invoking torch.jit.freeze
    m = torch.jit.freeze(m)

    with torch.no_grad():
        t0  = time.time()
        for _ in range(800):
            inv     = torch.randn(size=(16,17,8,8),device=dev,dtype=torch.float32,requires_grad=False)

            p,v       = m.forward(inv)

        print(f"out is {p.shape},{v.shape} in {(time.time()-t0):.3f}s")
