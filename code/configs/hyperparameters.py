#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#

#################   Hyperparameters Key   #################

# model                 -    string    - Choose from: [simple-cnn, mlp-mnist]
# dataset               -    string    - Choose from: [mnist, cifar10, cifar100]
# n_workers             -     int      - Number of normal workers.
# m_workers             -     int      - Number of malicious workers.
# alpha                 -    float     - Parameter alpha for dirichlet distribution
# beta                  -    float     - Sample rate for workers per communication round.
# tr_batch_size         -     int      - Batch-size used by the workers to train.
# ts_batch_size         -     int      - Batch-size used by the Server on test data.
# communication_rounds  -     int      - Total number of communication rounds.
# local_rounds          -     int      - Local training epochs at every worker.
# random_seed           -     int      - Random seed for model initializations.
# log_path              -    string    - Path to store the log files.

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
 			'n_workers': 80,
 			'm_workers': 20,
            'mw_params': {
                'type': 'mpaf',
                'target': '../runs/malicious.pt',
                'lambda': 1000,
            },
 			'alpha': 100.0,
            'beta': 0.10,
 			'tr_batch_size': 128,
 			'ts_batch_size': 1000,
 			'communication_rounds': 2000,
 			'local_rounds': 10,
 			'random_seed': 42,
 			'log_path': 'exp_baseline/'
		},
    ]
