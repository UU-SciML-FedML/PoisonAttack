#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch.optim as optim

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create and return the desired model along with optimizer & hyperparams.   #
#                                                                             #
#*****************************************************************************#
def get_model(model):

    # load required model    
    if model == "lenet_mnist":
        from .lenet_mnist import lenet_mnist
        return (lenet_mnist, optim.Adam, {"lr":0.001, "weight_decay":0.0})
