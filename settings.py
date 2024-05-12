#Author: Everett Stenberg
#Description:   configuration file to standardize various variables found all over the code 
import torch


#MODEL SETTINGS 
REPR_CH                                 = 17 
DTYPE                                   = torch.bfloat16

#TRAINING SETTINGS 
MAX_PLY                                 = 200   #(100 move/game)
DATASIZE                                = 2048
SEARCH_ITERS                            = 350
PARALLEL_TREES                          = 8
TRAIN_EVERY                             = 65536
BS                                      = 4096
LR                                      = .0001
BETAS                                   = (.75,.99)


#TREE SEARCH SETTINGS
DIR_A                                   = .3
DIR_E                                   = .4


#CHESS SETTINGS                 
N_CHESS_MOVES                           = 1968


#DISPLAY SETTINGS 
UPDATE_ITER                             = 120











#USEFUL COMBINATIONS OF SETTINGS (dont touch, these will be automatic. Only change the above settings)
MODEL_KWARGS                            = {"repr_ch":REPR_CH,"p":.5}
INPUT_SHAPE                             = (1,REPR_CH,8,8)
