#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torchvision
import torchvision.transforms as transforms


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