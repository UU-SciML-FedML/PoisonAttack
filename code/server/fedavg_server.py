#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch


#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
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
#   class that implements contract logic for Federated Distillation.          #
#                                                                             #
#*****************************************************************************#
class FedAvg_Server():
    def __init__(self, n_workers, model_fn, gamma=1):
        # meta-information about dataset
        self.n_workers = n_workers
        self.gamma = gamma
        self.wr_update = []
        # setup model and its weights
        self.model = model_fn().to(device)
        self.GLB_WEIGHTS = flatten(self.model)


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used to compute the aggregated labels and reward.       #
    #                                                                     #
    #---------------------------------------------------------------------#
    def federatedAverage(self):
        total_samples = 0
        agg_weights = torch.zeros_like(self.wr_update[0][1])
        
        # compute aggregate        
        for w_id, weights, samples in self.wr_update:
            total_samples += samples
            agg_weights += weights * samples
        
        # update global model
        self.GLB_WEIGHTS = agg_weights / total_samples


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used to compute the aggregated labels and reward.       #
    #                                                                     #
    #---------------------------------------------------------------------#
    def evaluateServer(self, ts_loader):
        """Evaluation function to check performance"""
        assert(ts_loader is not None)
        
        # setup latest weights into server model        
        unflatten(self.model, self.GLB_WEIGHTS)
        
        # start evaluation of the model
        self.model.eval()
        samples, correct = 0, 0
        
        # temporary variables
        #current_label = 0
        #predicts = dict()
        
        with torch.no_grad():
            for x, y in ts_loader:
                
                x, y = x.to(device), y.to(device)
                
                y_ = self.model(x)
                _, predicted = torch.max(y_.detach(), 1)
                
                samples += y.shape[0]
                correct += (predicted == y).sum().item()
        
        # return evaluation statistics
        return {"accuracy" : correct/samples}

    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used to clear local memory.                             #
    #                                                                     #
    #---------------------------------------------------------------------#            
    def clear_caches(self):
        # delete old data
        del self.wr_update
        # allocate new buffers
        self.wr_update = []


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#        
    def receive_update(self, w_id, weights, samples):
        self.wr_update.append([w_id, weights, samples])


    def global_model(self):
        return self.GLB_WEIGHTS
    
        