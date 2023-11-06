import numpy as np
import matplotlib.pyplot as plt
import os, sys, pdb, json, time

from utils import *
from Meso_CollectData_pop3ad import simulate_3ddata
from Meso_PostProcessData_pop3 import post_process_3ddata
from train_pop3ad import train_3dmodel
from postTrainingModel_pop3 import post_process_trained_3dmodel


# python main.py --lb 0.4 --ub 2.0 --sim_t_end 50000 --phase 1
#### all I need is lb, ub, sim_t_end
#### and phase
#### sample_N_perparam should be 5, setting_N should be something larger than 5**5=3375
exp_params = experiment_params()


setting_0_perphase = 10000


exp_params.args.setting_0 = setting_0_perphase * exp_params.args.phase+1
data_folder, _, _, folder_path, prefix = exp_params.return_folders()

## generate param configs  == only for phase 1 ==
sample_3dparams_uniform(exp_params.args.lb, exp_params.args.ub, exp_params.args.sample_N_perparam,
                            data_folder, exp_params.args.setting_0)

## simulate data
# python Meso_CollectData_pop3ad.py --lb 0.4 --ub 2.0 --sim_t_end 5000 --sample_N_perparam 5 --phase 1 --setting_0 10000
simulate_3ddata(exp_params)

# post-process data
# python Meso_PostProcessData_pop3.py --lb 0.4 --ub 2.0 --phase 1 
post_process_3ddata(exp_params)

# # train model
# train_3dmodel(exp_params, lr=3e-3, epochs=500, batch_size=1024)
train_3dmodel(exp_params, lr=1e-3, epochs=500, batch_size=1024)
# train_3dmodel(exp_params, lr=3e-4, epochs=500, batch_size=1024)

# ## post-process model
# epoch_min = 300
# epoch_max = 500
# post_process_trained_3dmodel(exp_params, epoch_min, epoch_max, setting_0_perphase)

