#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.nn.functional as F

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