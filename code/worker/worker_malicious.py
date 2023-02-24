#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
from .worker_base import Worker
import torch
#from utils import flatten, unflatten

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

    def __init__(self, model_fn, optimizer_fn, tr_loader, mw_params, idnum=None):
        super().__init__(model_fn, optimizer_fn, tr_loader, idnum)
        # setup malicious worker based on parameters provided
        self.mw_parameters = mw_params
        self.attacks_dict = { 
            'mpaf': self.perform_mpaf,
            'rand': self.perform_random,
        }

    #---------------------------------------------------------------------#
    #                                                                     #
    #   Bypass the run_fl_round function by doing nothing.                #
    #                                                                     #
    #---------------------------------------------------------------------#
    def perform_mpaf(self):
        
        # check if the target model has not already been loaded
        # load it and use it to perform attack calculation
        if not hasattr(self, "target_weights"):
            self.target_weights = torch.load(self.mw_parameters['target'])
            self.scaling_factor = self.mw_parameters['lambda']
        
        # perform the attack update and store it in local mode
        self.local_update = self.scaling_factor * (self.target_weights - self.global_weights)
        self.samples = 1
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Bypass the run_fl_round function by doing nothing.                #
    #                                                                     #
    #---------------------------------------------------------------------#
    def perform_random(self):
        pass        
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Bypass the run_fl_round function by doing nothing.                #
    #                                                                     #
    #---------------------------------------------------------------------#
    def run_fl_round(self, server, rounds=10):
        self.get_global_weights(server)
        # check what type of malicious activity is requested and then perform
        # that requested malicious activity.
        self.attacks_dict[self.mw_parameters['type']]()
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
        server.receive_update(self.id, self.local_update, self.samples)


