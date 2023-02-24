#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import numpy as np
from torch.utils.data import Dataset


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load training dataset as requested.                                       #
#                                                                             #
#*****************************************************************************#
def load_data(dataset, path, splits=1, alpha=1.0, double_stochastic=True):
    """Load the requested dataset and split it if requested."""
    
    # perform check and balances on provided parameters
    assert splits >= 1, "# of splits should be greater or equal to 1."
    assert (alpha > 0) and (alpha <= 100.0), "Dirichlet parameter alpha must be between 0 and 100."
    
    
    # load the dataset that has been requested
    if dataset == "mnist":
        from .load_mnist import load_mnist
        train_set, test_set = load_mnist(path)
    elif dataset == "cifar10":
        pass
    else:
        raise Exception(f"Unidentified dataset {dataset} requested.")
    
    # perform the required splitting of the data
    split_train, lbl_counts_train =  split_data(train_set, alpha, double_stochastic, splits)
    
    return split_train, lbl_counts_train, test_set

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create custom dataset, used for stl10 (labels -> targets) mapping.        #
#                                                                             #
#*****************************************************************************#
class CustomDataset(Dataset):
    r"""
    Create a dataset with given data and labels
    Arguments:
        dataset (Dataset): The whole Dataset
        labels(sequence) : targets as required for the indices. 
                                will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset.targets[indices]

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]][0]
        target = self.dataset[self.indices[idx]][1]
        return (data, target)

    def __len__(self):
        return len(self.indices)


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   split given dataset among workers using dirichlet distribution.           #
#                                                                             #
#*****************************************************************************#
def split_dirichlet(labels, n_workers, alpha, double_stochstic):
    """Splits data among the workers using dirichlet distribution"""

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels)+1
    
    # get label distibution
    label_distribution = np.random.dirichlet([alpha]*n_workers, n_classes)
   
    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]
    
    worker_idcs = [[] for _ in range(n_workers)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            worker_idcs[i] += [idcs]

    worker_idcs = [np.concatenate(idcs) for idcs in worker_idcs]

    print_split(worker_idcs, labels)
  
    return worker_idcs

def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   a class to create custom subsets with indexes appended.                   #
#                                                                             #
#*****************************************************************************#
class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return (idx, *self.dataset[self.indices[idx]])

    def __len__(self):
        return len(self.indices)

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   split and return datasets for workers.                                    #
#                                                                             #
#*****************************************************************************#
def split_data(dataset, alpha, double_stochstic, n_workers):
    """Split data among worker nodes."""
    
    # get meta information
    labels = dataset.targets
    labels = labels.numpy()
    n_classes = np.max(labels) + 1
    total_data = len(dataset)
    samples_per_class = int(total_data / n_classes)
    
    # get label indcs
    label_idcs = {l : np.random.permutation(
        np.argwhere(np.array(labels)==l).flatten()).tolist() for l in range(n_classes)}
    
    # create a new dataset with given indices
    chosen_idcs = [value[:samples_per_class] for key, value in label_idcs.items()]
    chosen_idcs = np.random.permutation(np.concatenate(chosen_idcs))

    # create the actual dataset
    chosen_dataset = CustomDataset(dataset, chosen_idcs)
    #print(f"Length of chosen dataset is: {len(chosen_dataset)}")
    
    # Find allocated indices using dirichlet split
    subset_idx = split_dirichlet(chosen_dataset.targets, n_workers, alpha, double_stochstic)
    
    # Compute labels per worker
    label_counts = [np.bincount(np.array(chosen_dataset.targets)[i], minlength=10) for i in subset_idx]
    
    # Get actual worker data
    worker_data = [IdxSubset(chosen_dataset, subset_idx[i]) for i in range(n_workers)]

    # Return worker data splits
    return worker_data, label_counts


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper function to print data splits made for workers.                    #
#                                                                             #
#*****************************************************************************#
def print_split(idcs, labels):
    
    n_labels = np.max(labels) + 1 
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Worker {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)
    
    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()