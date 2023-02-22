#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch

#***********************************************************************************************#
#                                                                                               #
#   Description:                                                                                #
#   unflatten the passed model to be used for synchronization.                                  #
#                                                                                               #
#***********************************************************************************************#
def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param