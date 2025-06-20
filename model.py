#Author: Everett Stenberg
#Description:   Contains various pytorch Neural Networks to act as the
#               board evaluation. ChessModel2 is the current one used



import torch
import settings
import torch.nn as nn
import chess 
import data
import numpy 
from typing import Union 

class ChessBlock(torch.nn.Module):

    def __init__(self,in_ch:int,v_conv_ch,h_conv_ch,f_conv_ch):
        super(ChessBlock,self).__init__()
        
        self.vconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,v_conv_ch,kernel_size=(7+7+1,1),stride=1,padding=(7,0),bias=False),
                                                  torch.nn.BatchNorm2d(v_conv_ch),
                                                  torch.nn.PReLU(v_conv_ch))

        self.hconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,h_conv_ch,kernel_size=(1,7+7+1),stride=1,padding=(0,7),bias=False),
                                                  torch.nn.BatchNorm2d(h_conv_ch),
                                                  torch.nn.PReLU(h_conv_ch))

        self.fullconv1      = torch.nn.Sequential(torch.nn.Conv2d(in_ch,f_conv_ch,kernel_size=(3,3),stride=1,padding=1,bias=False),
                                                  torch.nn.BatchNorm2d(f_conv_ch),
                                                  torch.nn.PReLU(f_conv_ch))
        
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        #Pass
        v_conv              = self.vconv1(x)
        h_conv              = self.hconv1(x)
        f_conv              = self.fullconv1(x)

        #Restack
        #return f_conv
        return torch.cat([v_conv,h_conv,f_conv],dim=1)


class ConvBlock(torch.nn.Module):

    def __init__(self,in_ch:int,conv_ch,kernel:tuple[int,int]=(3,3),pad=True):
        super(ConvBlock,self).__init__()

        self.fullconv1      = torch.nn.Sequential(torch.nn.Conv2d(in_ch,conv_ch,kernel_size=kernel,stride=1,padding=kernel[0]//2 if pad else 0,bias=False),
                                                  torch.nn.BatchNorm2d(conv_ch),
                                                  torch.nn.PReLU(conv_ch))
        
    def forward(self,x:torch.Tensor) ->torch.Tensor:

        return self.fullconv1(x)



class ChessModel(torch.nn.Module):


    def __init__(self,repr_ch:int=17,p=.5):

        super(ChessModel,self).__init__()
        



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
                                                  torch.nn.PReLU(256),
                                                  torch.nn.Flatten(start_dim=1))


        #P
        self.prob_head      = torch.nn.Sequential(torch.nn.Linear(256*2*2,1968),
                                                  torch.nn.Softmax(dim=1))

        #V
        self.val_head       = torch.nn.Sequential(torch.nn.Linear(256*2*2,256),
                                                  torch.nn.Dropout(p=p),
                                                  torch.nn.PReLU(),
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


class ChessModel2(torch.nn.Module):


    def __init__(self,repr_ch:int=17,p=0):

        super(ChessModel2,self).__init__()

        n_v                 = 32#32 
        n_h                 = 32#32 
        n_c                 = 32
        n_ch                = n_c+n_v+n_h

        n_channels          = 32


        #Chess Layers
        self.layer1_1       = ConvBlock(repr_ch,n_channels)#ChessBlock(repr_ch,n_v,n_h,n_c)
        self.layer1_2       = ConvBlock(n_channels,n_channels)#ConvBlock(n_v+n_h+n_c,128)
        self.layer2_1       = ConvBlock(n_channels,n_channels)
        self.layer2_2       = ConvBlock(n_channels,n_channels)
        self.layer3_1       = ConvBlock(n_channels,n_channels)
        self.layer3_2       = ConvBlock(n_channels,n_channels)
        self.layer4_1       = ConvBlock(n_channels,n_channels)
        self.layer4_2       = ConvBlock(n_channels,n_channels)



        #Split to prob and value heads 
        #Standard layers    in=(bsx64x8x8)
        self.val_head       = torch.nn.Sequential(ConvBlock(n_channels,n_channels),
                                                  ConvBlock(n_channels,n_channels),
                                                  torch.nn.AvgPool2d(kernel_size=2),
                                                  ConvBlock(n_channels,256),
                                                  ConvBlock(256,512,kernel=(4,4),pad=False),
                                                  
                                                  torch.nn.Flatten(start_dim=1),

                                                  torch.nn.Linear(512,1),
                                                  torch.nn.Tanh())

        #P
        self.prob_head      = torch.nn.Sequential(ConvBlock(n_channels,n_channels*2),
                                                  ConvBlock(n_channels*2,n_channels*4),
                                                  torch.nn.MaxPool2d(kernel_size=2),
                                                  ConvBlock(n_channels*4,2048),
                                                  ConvBlock(2048,1968,kernel=(4,4),pad=False),
                                                  
                                                  torch.nn.Flatten(start_dim=1),
                                                  torch.nn.Softmax(dim=1))

        self.to(settings.DTYPE)


    def forward(self,x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:

        #Round1
        x1                  = self.layer1_1(x)
        x2                  = self.layer1_2(x1)

        x1                  = self.layer2_1(x2)
        x2                  = self.layer2_2(x1)     + x2

        x1                  = self.layer3_1(x2)  
        x2                  = self.layer3_2(x1)     + x2

        x1                  = self.layer4_1(x2)  
        x2                  = self.layer4_2(x1)     + x2
        #Get outs
        return self.prob_head(x2),self.val_head(x2)


class ChessTransformer(nn.Module):
    
    def __init__(self, emb_dim=256, num_layers=8, num_heads=8):
        super().__init__()
        self.n_embed            = emb_dim
        self.embedding          = nn.Embedding(13,emb_dim)
        self.pos_embed          = nn.Parameter(torch.randn(1,64, emb_dim))  # Learned pos enc
        self.white_embd         = nn.Parameter(torch.randn(64,emb_dim))
        self.black_embd         = nn.Parameter(torch.randn(64,emb_dim))
        encoder_layer           = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * emb_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1968),    # Number of possible legal moves (from UCI move encoding)
            nn.Softmax(dim=1)
        )

    def forward(self, x:torch.Tensor,turn:torch.Tensor):  # x: (B, 17, 8, 8)

        #Get piece embeddings 
        BS          = x.size(0)
        piece_emb   = self.embedding(x.long())
        board_emb   = self.pos_embed.repeat(BS,1,1)
        turn_emb    = torch.where(turn.view(-1,1,1).expand(-1,64,self.n_embed)==1,self.white_embd,self.black_embd)
        #x = x.view(B, 64, 17)
        x           = piece_emb + board_emb + turn_emb
        x           = self.transformer_layers(x)

        value       = self.value_head(x)
        policy      = self.policy_head(x)
        return policy, value

    def encode_fens(self,fen_batch:list[chess.Board],as_torch=False) -> Union[torch.Tensor,numpy.array]:
        as_numpy    =  numpy.asarray(list(map(data.to_64_len_str,fen_batch)),dtype=numpy.float32)
        if as_torch:
            return torch.from_numpy(as_numpy)
        else:
            return as_numpy


if __name__ == "__main__":
    import time
    dev     = torch.device('cuda')
    ty      = torch.bfloat16
    torch.jit.enable_onednn_fusion(True)
    torch.backends.cudnn.enabled= True
    m       = ChessModel2().eval().to(dev).type(ty)
    # m = torch.jit.trace(m, [torch.randn(size=(16,17,8,8),device=dev,dtype=ty)])
    # # Invoking torch.jit.freeze
    # m = torch.jit.freeze(m)

    # with torch.no_grad():
    #     t0  = time.time()
    #     for _ in range(16384):
    inv     = torch.randn(size=(16,17,8,8),device=dev,dtype=ty,requires_grad=False)

    p,v       = m.forward(inv)

    print(f"out is {p.shape},{v.shape}s")
    print(sum([p.numel() for p in m.parameters()]))
