#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.optim as optim
import torch.nn as nn
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
class WorkerNormal():

    def __init__(self, model_fn, optimizer_fn, tr_loader, idnum=None):
        self.id = idnum
        # local models and dataloaders
        self.tr_model = model_fn().to(device) #copy.deepcopy(model_fn()).to(device)
        self.tr_loader = tr_loader
        # optimizer parameters        
        self.optimizer_fn = optimizer_fn
        self.optimizer = optimizer_fn(self.tr_model.parameters())   
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Train Worker using its local dataset.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def run_fl_round(self, server, rounds=10):
        self.get_global_weights(server)
        self.train(rounds=rounds)
        self.send_weights_to_server(server)
        
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Train Worker using its local dataset.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def train(self, rounds):
        """Training function to train local model"""
        
        # start training the worker using local dataset
        self.tr_model.train()  
        running_loss, samples = 0.0, 0
        train_rounds = 0
        
        while True:
            
            # end training if exceeding training rounds
            train_rounds += 1
            if (train_rounds >= rounds):
                break
            
            # train next epoch
            for i, x, y in self.tr_loader:   
                
                x, y = x.to(device), y.to(device)
                
                self.optimizer.zero_grad()
                
                loss = nn.CrossEntropyLoss()(self.tr_model(x), y)
                
                running_loss += loss.item()*y.shape[0]
                samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()  

        train_stats = {"loss" : running_loss / samples}
        self.samples = samples
        
        # return training statistics
        return train_stats


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Evaluate Worker to see if it is improving as expected or not.     #
    #                                                                     #
    #---------------------------------------------------------------------#
    def evaluate(self, ts_loader):
        """Evaluation function to check performance"""
        assert(ts_loader is not None)
        
        # start evaluation of the model
        self.tr_model.eval()
        samples, correct = 0, 0
        
        with torch.no_grad():
            for x, y in ts_loader:
                
                x, y = x.to(device), y.to(device)
                
                y_ = self.tr_model(x)
                _, predicted = torch.max(y_.detach(), 1)
                
                samples += y.shape[0]
                correct += (predicted == y).sum().item()
        
        # return evaluation statistics
        return {"accuracy" : correct/samples}

        
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