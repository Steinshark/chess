#Author: Everett Stenberg
#Description:   configuration file to standardize various variables found all over the code 
import torch


#MODEL SETTINGS 
REPR_CH                                 = 17 
CONV_CH                                 = 16
DTYPE                                   = torch.bfloat16

#TRAINING SETTINGS 
MAX_PLY                                 = 300   #(150 move game)
DATASIZE                                = 2048
SEARCH_ITERS                            = 850
PARALLEL_TREES                          = 8
TRAIN_EVERY                             = 131072*2


#TREE SEARCH SETTINGS
DIR_A                                   = .3
DIR_E                                   = .3


#CHESS SETTINGS                 
N_CHESS_MOVES                           = 1968


#DISPLAY SETTINGS 
UPDATE_ITER                             = 120











#USEFUL COMBINATIONS OF SETTINGS (dont touch, these will be automatic. Only change the above settings)
MODEL_KWARGS                            = {"in_ch":REPR_CH,"n_channels":CONV_CH,"lin_act":torch.nn.PReLU,"conv_act":torch.nn.PReLU}
INPUT_SHAPE                             = (1,REPR_CH,8,8)
