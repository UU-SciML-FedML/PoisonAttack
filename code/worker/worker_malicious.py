#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
from .worker_base import Worker
import torch
from utils import flatten, unflatten

#-----------------------------------------------------------------------------#
#                                                                             #
#   Define global parameters to be used through out the program               #
#                                                                             #
#-----------------------------------------------------------------------------#
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   class that implements worker node logic for Federated Distillation.       #
#                                                                             #
#*****************************************************************************#
class WorkerMalicious(Worker):

    def __init__(self, model_fn, optimizer_fn, tr_loader, idnum=None):
        super().__init__(model_fn, optimizer_fn, tr_loader, idnum)
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def get_global_weights (self, server):
        # fetch global aggregated labels from server / contract
        global_weights =  server.global_model()
        
        # decompress and apply to local model
        unflatten(self.tr_model, global_weights)
        self.tr_model.to(device)


    def send_weights_to_server (self, server):
        # compress and send weights to server
        local_weights = flatten(self.tr_model)
        server.receive_update(self.id, local_weights, self.samples)