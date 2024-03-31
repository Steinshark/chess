import torch 
import chess_utils 

class RolloutModel(torch.nn.Module):

    def __init__(self):

        super(RolloutModel,self).__init__()

        #Receives data in (bs,15,8,8)
        self.layer_1        = torch.nn.Sequential(torch.nn.Conv2d(15,64,3,1,1),torch.nn.BatchNorm2d(64),torch.nn.PReLU(64))

        self.layer_2        = torch.nn.Sequential(torch.nn.Conv2d(64,32,3,1,1),torch.nn.BatchNorm2d(32),torch.nn.PReLU(32))

        self.layer_3        = torch.nn.Sequential(torch.nn.Conv2d(32,16,3,1,1),torch.nn.BatchNorm2d(16),torch.nn.PReLU(16))

        self.layer_4        = torch.nn.Sequential(torch.nn.Conv2d(16,1,3,1,1))

        #self.layer_5        = torch.nn.Sequential(torch.nn.Conv2d(8,1,3,1,1),torch.nn.PReLU(1))

        self.output_layer = torch.nn.Sequential(torch.nn.Flatten(start_dim=-3),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(64,32),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(32,1))
    
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        y   = self.layer_1(x)
        y   = self.layer_2(y)
        y   = self.layer_3(y)
        y   = self.layer_4(y)
        return self.output_layer(y)

class ChessModel(torch.nn.Module):


    def __init__(self,in_ch:int=15,n_channels:int=32):

        super(ChessModel,self).__init__()

        self.v_conv_n      = n_channels
        self.h_conv_n      = n_channels
        self.q_conv_n      = n_channels

        self.conv_act       = torch.nn.functional.leaky_relu
        self.lin_act        = torch.nn.functional.relu
        self.softmax        = torch.nn.functional.softmax
        self.lin_upsc       = 1 

        #Views of board Layer1
        self.vert_conv1     = torch.nn.Conv2d(in_ch,self.v_conv_n,kernel_size=(8+8+1,1),stride=1,padding=(8,0))
        self.horz_conv1     = torch.nn.Conv2d(in_ch,self.h_conv_n,kernel_size=(1,8+8+1),stride=1,padding=(0,8))
        self.quad_conv1     = torch.nn.Conv2d(in_ch,self.q_conv_n,kernel_size=(7),stride=1,padding=3)
        self.linear_l1      = torch.nn.Sequential(torch.nn.Flatten(1),torch.nn.Linear(in_ch*8*8,in_ch*8*8*self.lin_upsc),torch.nn.Unflatten(dim=1,unflattened_size=(in_ch*self.lin_upsc,8,8)))

        #Views of board Layer2 
        self.vert_conv2     = torch.nn.Conv2d(self.lin_upsc*in_ch+self.v_conv_n+self.h_conv_n+self.q_conv_n,self.v_conv_n,kernel_size=(8+8+1,1),stride=1,padding=(8,0))
        self.horz_conv2     = torch.nn.Conv2d(self.lin_upsc*in_ch+self.v_conv_n+self.h_conv_n+self.q_conv_n,self.h_conv_n,kernel_size=(1,8+8+1),stride=1,padding=(0,8))
        self.quad_conv2     = torch.nn.Conv2d(self.lin_upsc*in_ch+self.v_conv_n+self.h_conv_n+self.q_conv_n,self.q_conv_n,kernel_size=(7),stride=1,padding=3)
        self.linear_l2      = torch.nn.Sequential(torch.nn.Flatten(1),torch.nn.Linear(in_ch*8*8*self.lin_upsc,in_ch*8*8*self.lin_upsc*self.lin_upsc),torch.nn.Unflatten(dim=1,unflattened_size=(in_ch*self.lin_upsc,8,8)))

        #Views of board Layer3
        self.vert_conv3     = torch.nn.Conv2d(self.lin_upsc*in_ch+self.v_conv_n+self.h_conv_n+self.q_conv_n,self.v_conv_n,kernel_size=(8+8+1,1),stride=1,padding=(8,0))
        self.horz_conv3     = torch.nn.Conv2d(self.lin_upsc*in_ch+self.v_conv_n+self.h_conv_n+self.q_conv_n,self.h_conv_n,kernel_size=(1,8+8+1),stride=1,padding=(0,8))
        self.quad_conv3     = torch.nn.Conv2d(self.lin_upsc*in_ch+self.v_conv_n+self.h_conv_n+self.q_conv_n,self.q_conv_n,kernel_size=(7),stride=1,padding=3)
        self.linear_l3      = torch.nn.Sequential(torch.nn.Flatten(1),torch.nn.Linear(in_ch*8*8*self.lin_upsc,in_ch*8*8*self.lin_upsc*self.lin_upsc),torch.nn.Unflatten(dim=1,unflattened_size=(in_ch*self.lin_upsc,8,8)))

        self.flatten        = torch.nn.Flatten()

        self.p_module       = torch.nn.Sequential(torch.nn.Linear(8*8*(self.v_conv_n+self.h_conv_n+self.q_conv_n+in_ch*self.lin_upsc),len(chess_utils.CHESSMOVES)),
                                                  torch.nn.Softmax(dim=1))
        
        self.v_module       = torch.nn.Sequential(
                        torch.nn.Linear(8*8*(self.v_conv_n+self.h_conv_n+self.q_conv_n+in_ch*self.lin_upsc),1024),
                        torch.nn.Dropout(p=.5),
                        torch.nn.PReLU(1),
                        torch.nn.Linear(1024,256),
                        torch.nn.Dropout(p=.1),
                        torch.nn.PReLU(1),
                        torch.nn.Linear(256,1),
                        torch.nn.Tanh()
        )

            
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        #ITER1 Get vertical, horizontal, and square convolutions 
        vert_convolutions1  = self.conv_act(self.vert_conv1(x))                                         #Out    = (32,8,8)
        horz_convolutions1  = self.conv_act(self.horz_conv1(x))                                         #Out    = (32,8,8)
        quad_convolutions1  = self.conv_act(self.quad_conv1(x))                                         #Out    = (32,8,8)
        linear_outputs1      = self.linear_l1(x)
        comb_convolutions1  = torch.cat([vert_convolutions1,horz_convolutions1,quad_convolutions1,linear_outputs1],dim=1)

        vert_convolutions2  = self.conv_act(self.vert_conv2(comb_convolutions1))                        #Out    = (96,8,8)
        horz_convolutions2  = self.conv_act(self.horz_conv2(comb_convolutions1))                        #Out    = (96,8,8)
        quad_convolutions2  = self.conv_act(self.quad_conv2(comb_convolutions1))                        #Out    = (96,8,8)
        linear_outputs2     = self.linear_l2(linear_outputs1)
        comb_convolutions2  = torch.cat([vert_convolutions2,horz_convolutions2,quad_convolutions2,linear_outputs2],dim=1)

        vert_convolutions3  = self.conv_act(self.vert_conv3(comb_convolutions2))                        #Out    = (96 ,8,8)
        horz_convolutions3  = self.conv_act(self.horz_conv3(comb_convolutions2))                        #Out    = (96 ,8,8)
        quad_convolutions3  = self.conv_act(self.quad_conv3(comb_convolutions2))                        #Out    = (96 ,8,8)
        linear_outputs3     = self.linear_l3(linear_outputs2)
        comb_convolutions3  = torch.cat([vert_convolutions3,horz_convolutions3,quad_convolutions3,linear_outputs3],dim=1)

        x                   = self.flatten(comb_convolutions3)

        #Get p val 
        return self.p_module(x),self.v_module(x)
        




if __name__ == "__main__":

    m       = ChessModel(15).to(torch.device('cuda'))

    inv     = torch.randn(size=(16,15,8,8),device=torch.device('cuda'))

    y       = m.forward(inv)

    print(f"out is {y[0].shape},{y[1].shape}")
