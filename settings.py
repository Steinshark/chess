#Author: Everett Stenberg
#Description:   configuration file to standardize various variables found all over the code 
import torch


#MODEL SETTINGS 
REPR_CH                                 = 17 
CONV_CH                                 = 16
DTYPE                                   = torch.float32

#TRAINING SETTINGS 
MAX_PLY                                 = 160   #(80 move game)
DATASIZE                                = 1024
SEARCH_ITERS                            = 400
PARALLEL_TREES                          = 4


#TREE SEARCH SETTINGS
DIR_A                                   = .3
DIR_E                                   = .2


#CHESS SETTINGS                 
N_CHESS_MOVES                           = 1968


#DISPLAY SETTINGS 
UPDATE_ITER                             = 30











#USEFUL COMBINATIONS OF SETTINGS (dont touch, these will be automatic. Only change the above settings)
MODEL_KWARGS                            = {"in_ch":REPR_CH,"n_channels":CONV_CH,"lin_act":torch.nn.PReLU,"conv_act":torch.nn.PReLU}
INPUT_SHAPE                             = (1,REPR_CH,8,8)
