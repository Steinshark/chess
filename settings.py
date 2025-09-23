#Author: Everett Stenberg
#Description:   configuration file to standardize various variables found all over the code 
import torch


#MODEL SETTINGS 
REPR_CH                                 = 17 
N_EMBED                                 = 256
DTYPE                                   = torch.float
JIT_SHAPE                               = (1,8,8,N_EMBED)


#TRAINING SETTINGS 
MAX_PLY                                 = 40   #Start with (50 move/game) to start bootstrapping
DATASIZE                                = 800
SEARCH_ITERS                            = 450
PARALLEL_TREES                          = 4
TRAIN_EVERY                             = 16384
BS                                      = 1024
LR                                      = .0001
BETAS                                   = (.9,.99)


#TREE SEARCH SETTINGS
DIR_A                                   = .3
DIR_E                                   = .25


#CHESS SETTINGS                 
N_CHESS_MOVES                           = 1968


#DISPLAY SETTINGS 
UPDATE_ITER                             = 120











#USEFUL COMBINATIONS OF SETTINGS (dont touch, these will be automatic. Only change the above settings)
MODEL_KWARGS                            = {"repr_ch":REPR_CH,"p":.5}
INPUT_SHAPE                             = (1,REPR_CH,8,8)
