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
    #   Bypass the run_fl_round function by doing nothing.                #
    #                                                                     #
    #---------------------------------------------------------------------#
    def run_fl_round(self, server, rounds=10):
        self.get_global_weights(server)
        self.samples = 1
        self.send_weights_to_server(server)
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def get_global_weights (self, server):
        # fetch global aggregated labels from server / contract
        self.global_weights =  server.global_model()


    def send_weights_to_server (self, server):
        server.receive_update(self.id, self.global_weights, self.samples)