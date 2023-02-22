#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#

#################   Hyperparameters Key   #################

# model                -    string    - Choose from: [simple-cnn, mlp-mnist]

# dataset              -    string    - Choose from: [mnist, cifar10, cifar100]

# n_workers            -     int      - Number of Workers.

# alpha                 -    float    - Parameter alpha for dirichlet distribution
#                                       required if worker_data splits not defined,
#                                       used only if worker_data not defined.

# batch_size            -     int     - Batch-size used by the Workers.

# communication_rounds  -     int     - Total number of communication rounds.

# local_rounds          -     int     - Local training epochs at every worker.

# random_seed           -     int     - Random seed for model initializations.

# log_path              -    string   - Path to store the log files.

#-----------------------------------------------------------------------------#
#                                                                             #
#   Define hyperparameters that will assit in running experiments.            #
#                                                                             #
#-----------------------------------------------------------------------------#
def init( ):
    # create a list of possible experimental setups
    global hyperparams
    hyperparams =[
        ##################################################################
        ####################   alpha = 100 / mnist  ######################
        ##################################################################
        {
 			'model': 'lenet_mnist',
 			'dataset': 'mnist',
 			'n_classes': 10,
 			'n_workers': 1,
 			'm_workers': 10,
 			'classes_per_worker': 0,
 			'total_data': 1.0,
 			'alpha': 1.0,
            'beta': 0.1,
 			'tr_batch_size': 128,
 			'ts_batch_size': 1000,
 			'communication_rounds': 5,
 			'local_rounds': 5,
 			'random_seed': 42,
 			'log_path': 'exp_baseline/'
		},
    ]
