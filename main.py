from joblib import Parallel, delayed
import multiprocessing

import sys, os, inspect

from alpha_beta import alpha_beta
from utility import assign_cuda_bootstraps_others

###################################
num_clusters = 5
bootstraps = range(1,6)
use_pretraining = [True]
results_address = './results/'
data_address = './data/'
cache_address = './CACHE/'
optimization_params = {'epochs': 50, 'lr': 0.001, 'momentum': 0.9, 'step': 30}

regularization_weight = 0
cuda_devices = range(1,6)
###################################

num_processes = len(cuda_devices) * len (use_pretraining) * len(bootstraps)

bootstrap_regul_cuda_assignment = assign_cuda_bootstraps_others (cuda_devices, bootstraps, use_pretraining, [None])

for x in [(_CUDA_DEVICE, _use_pretraining, _bootstrap_number) for (_CUDA_DEVICE, _use_pretraining, _, _bootstrap_number) in bootstrap_regul_cuda_assignment]:
    print (x)

current_address = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
results = Parallel(n_jobs=num_processes)(delayed(alpha_beta)(num_clusters, _bootstrap_number, _use_pretraining, current_address, results_address, data_address, cache_address, optimization_params, regularization_weight, _CUDA_DEVICE) for (_CUDA_DEVICE, _use_pretraining, _, _bootstrap_number) in bootstrap_regul_cuda_assignment)
