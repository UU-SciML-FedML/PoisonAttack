#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import argparse, time
from torch.utils.data import DataLoader
import numpy as np

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
from configs.hyperparameters import hyperparams as hp_dicts
import experiment_manager as expm
import model, data
from worker import WorkerNormal, WorkerMalicious
from server import FedAvg_Server
from utils import random_sample

#from graph import plot_graphs
np.set_printoptions(precision=4, suppress=True)

#-----------------------------------------------------------------------------#
#                                                                             #
#   Parse passed arguments to get meta parameters.                            #
#                                                                             #
#-----------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)
args = parser.parse_args()

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   run individual experiment using the information passed.                   #
#                                                                             #
#*****************************************************************************#
def run_experiment(exp, exp_count, n_experiments):
    # print log information
    print("Running Experimemt {} of {} with".format(exp_count+1, n_experiments))
    print(exp)
    
    # get hyperparameters of current experiment
    hp = exp.hyperparameters
    model_fn, optimizer, optimizer_hp = model.get_model(hp["model"])
    optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()})
    
    # get datasets needed for training and distillation
    train_data, label_counts, test_data = data.load_data(
        hp["dataset"], 
        args.DATA_PATH, 
        splits=hp["n_workers"],
        alpha=hp["alpha"])
    
    # setup up random seed as defined in hyperparameters
    np.random.seed(hp["random_seed"])
    
    # create dataloaders for all datasets loaded so far
    worker_loaders = [
        DataLoader(local_data, 
                   batch_size=hp["tr_batch_size"], 
                   shuffle=True) for local_data in train_data]
    
    test_loader = DataLoader(
        test_data, 
        batch_size=hp["ts_batch_size"], 
        shuffle=False)
    
    # create instances of normal / honest workers
    normal_workers = [
        WorkerNormal(model_fn,
                     optimizer_fn,
                     tr_loader=loader,
                     idnum = i
                     )
        for i, (loader, counts) in enumerate(zip(worker_loaders, label_counts))
    ]
    
    # create instances of malicious workers
    malice_workers = [
        WorkerMalicious(model_fn,
                     optimizer_fn,
                     tr_loader=None,
                     mw_params=hp["mw_params"],
                     idnum = k + hp["n_workers"]
                     ) for k in range(hp["m_workers"])
    ]
    
    # append malicious and normal workers
    workers = normal_workers + malice_workers
    
    # create a FedAvg Server
    server = FedAvg_Server(n_workers=hp["n_workers"], model_fn=model_fn)
    
    print("Starting Distributed Training..\n")
    t1 = time.time()
    
    # start training each client individually
    for c_round in range(0, hp["communication_rounds"]):
        print(f"Communication Round: {c_round+1}")
        # sample workers for current round of training
        sampled_workers = random_sample(workers, hp["beta"])
        exp.log({"Sampled_Workers" : np.array([worker.id for worker in sampled_workers])})

        for worker in sampled_workers:
            print(f"Train WORKER: {worker.id}")
            # Running local FL Training rounds
            worker.run_fl_round(server=server, rounds=hp["local_rounds"])
        
        # run Federated Averaging rountine
        server.federatedAverage()
        
        # evaluate server's performance
        eval_stats = server.evaluateServer(test_loader)
        print("\n")
        print(eval_stats)
        print("\n")

    print("Experiment: ({}/{})".format(exp_count+1, n_experiments))

    # save logging results to disk
    try:
      exp.save_to_disc(path=args.RESULTS_PATH, name=hp['log_path'])
    except:
      print("Saving results Failed!")

    # compute total time taken by the experiment
    print("Experiment {} took time {} to run..".format(exp_count+1, 
                                                       time.time() - t1))
    
    # Free up memory
    del server; workers.clear()


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   run all experiments as specified by the hyperparameters file.             #
#                                                                             #
#*****************************************************************************#
def run():
    # create instances of experiment manager class for each setup
    experiments = [expm.Experiment(hyperparameters=hp, log_id=i) for i, hp in enumerate(hp_dicts)]
    
    # run all experiments
    print("Running a total of {} Experiments..\n".format(len(experiments)))
    for exp_count, experiment in enumerate(experiments):
        run_experiment(experiment, exp_count, len(experiments))

# main program starts here
if __name__ == "__main__":
    run()