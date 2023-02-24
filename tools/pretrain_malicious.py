#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

#-----------------------------------------------------------------------------#
#                                                                             #
#   Define global parameters to be used through out the program               #
#                                                                             #
#-----------------------------------------------------------------------------#
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   Model training logic for the pre-training the malicious model.            #
#                                                                             #
#*****************************************************************************#
class Trainer( ):
    
    def __init__(self, model_fn, optimizer_fn, tr_loader):
        # local models and dataloaders
        self.tr_model = model_fn().to(device)
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
    #def run_fl_round(self, server, rounds=10):
    #    self.get_global_weights(server)
    #    self.train(rounds=rounds)
    #    self.send_weights_to_server(server)
        
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
            for x, y in self.tr_loader:   
                
                x, y = x.to(device), y.to(device)
                
                self.optimizer.zero_grad()
                
                loss = nn.CrossEntropyLoss()(self.tr_model(x), y)
                
                running_loss += loss.item()*y.shape[0]
                samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()  
                
            print(running_loss/samples)

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
        
        print(correct/samples)
        
        # return evaluation statistics
        return {"accuracy" : correct/samples}

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load and return the training and testing sets of MNIST dataset.           #
#                                                                             #
#*****************************************************************************#
def load_mnist(path):
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.EMNIST(
        root=path+"EMNIST", train=True, split="mnist", download=True, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=path+"EMNIST", train=False, split="mnist", download=True, transform=transform
    )

    # return the entire datasets
    return trainset, testset

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create LENET mnist model using the specifications provided.               #
#                                                                             #
#*****************************************************************************#
class lenet_mnist(torch.nn.Module):
    def __init__(self):
        super(lenet_mnist, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        #self.bn1 = torch.nn.BatchNorm2d(6)
        #self.bn2 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.binary = torch.nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   perform some sort of data manipulation to create a specific target model. #
#                                                                             #
#*****************************************************************************#
def mainpulate_data(tr_data, ts_data):
    
    tr_mal_data = copy.deepcopy(tr_data)
    ts_mal_data = copy.deepcopy(ts_data)
    tr_mal_data.targets[tr_data.targets==3] = 8
    ts_mal_data.targets[ts_data.targets==3] = 8
    
    return tr_mal_data, ts_mal_data


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   routine to train the model based on some sort of malicious data.          #
#                                                                             #
#*****************************************************************************#
def pretrain_model():
    # set up model parameters
    model_fn, optimizer, optimizer_hp = lenet_mnist, optim.SGD, {"lr":0.01, "weight_decay":0.0}
    optimizer_fn = lambda x : optimizer(x, **optimizer_hp)
    
    # load dataset
    train_data, test_data = load_mnist("../runs/data/")
    
    # manipulate the data
    tr_mal_data, ts_mal_data = mainpulate_data(train_data, test_data)
    
    # create dataloaders for all datasets loaded so far
    tr_loader = DataLoader(tr_mal_data, batch_size=32, shuffle=True)
    ts_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    ts_mal_loader = DataLoader(ts_mal_data, batch_size=32, shuffle=True)
    
    # create an instance of trainer and perform model training
    trainer = Trainer(model_fn, optimizer_fn, tr_loader)
    trainer.train(10)
    
    # evaluate model
    trainer.evaluate(ts_loader)
    trainer.evaluate(ts_mal_loader)
    
    # flatten parameters and store them to disk
    vec = []
    for param in trainer.tr_model.parameters():
        vec.append(param.data.view(-1))
    flat_params = torch.cat(vec)
    
    print(flat_params)
    
    # save the trained model to disk
    torch.save(flat_params, '../runs/malicious.pt')
    

if __name__ == '__main__':
    pretrain_model()