import os
import shutil
from pretraining import pretraining
from mixture_of_nets import train_mixture
from utility import make_dir, log_exceptions


def alpha_beta(num_clusters, bootstrap_number, use_pretraining, current_address, results_address, data_address, cache_address, optimization_params, regularization_weight, CUDA_DEVICE):

    current_subfolder = "num_clusters_%d__num_epochs_%d__use_pretraining_%s__regularization_weight_%.2f"%(num_clusters, optimization_params['epochs'], str(use_pretraining), regularization_weight)

    exception_log_address = os.path.join (current_address, results_address)
    results_address = os.path.join (current_address, results_address, current_subfolder)
    data_address = os.path.join (current_address, data_address)
    cache_address = os.path.join (current_address, cache_address, current_subfolder)

    for address in [results_address, data_address, cache_address]:
        make_dir (address)

    try:
        #raise Exception ('asd')
        if use_pretraining:
            #print ('sadasadsdasdasdasdasadsadsads'*10)
            pretraining(num_clusters, bootstrap_number, results_address, data_address, cache_address, optimization_params, CUDA_DEVICE)

        train_mixture(num_clusters, bootstrap_number, use_pretraining, results_address, data_address, cache_address, optimization_params, regularization_weight, CUDA_DEVICE)

    except Exception as e:
        log_exceptions (str(e), {'num_clusters':num_clusters, 'epochs':optimization_params['epochs'], 'use_pretraining':use_pretraining, 'regulairzation weight': regularization_weight}, exception_log_address)
        #shutil.rmtree(results_address)
        raise
