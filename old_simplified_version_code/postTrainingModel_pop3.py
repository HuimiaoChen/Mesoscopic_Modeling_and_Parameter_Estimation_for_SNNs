import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os, sys, pdb
from itertools import product

from utils import *


def post_process_trained_3dmodel(exp_params, epoch_min, epoch_max, setting_0_perphase):
	nstds = {1: [-5,-2,0,2,5], 2:  [-4,-2,0,2,4], 3: [-3,-1,0,1,3], 4:  [-2,-1,0,1,2], 5:  [-2,-1,0,1,2], 6:  [-2,-1,0,1,2], 7:  [-2,-1,0,1,2]}
	args = exp_params.parse_args()
	_, data_folder, trained_model_folder, folder_path, prefix = exp_params.return_folders()

	trained_modelnames = [fn for fn in os.listdir(trained_model_folder) if (fn.startswith('model') and fn.endswith('.npy'))]
	params = []
	for f in trained_modelnames:
		epoch = int(f.split('.')[-2].split('_')[-1])

		if epoch >= epoch_min and epoch <= epoch_max:
			params.append(np.load(os.path.join(trained_model_folder, f)))

	params = np.array(params)
	print(params.mean(axis=0) + nstds[args.phase][0]*params.std(axis=0))
	print(params.mean(axis=0) + nstds[args.phase][-1]*params.std(axis=0))

	params = np.array([params.mean(axis=0)+i*params.std(axis=0) for i in nstds[args.phase] ]).T
	params = product(*params)
	params = np.array([p for p in params])

	np.savez(f"{trained_model_folder}/params.npz", params=params, N_perparam=len(nstds[args.phase]), setting_0=setting_0_perphase*(args.phase+1))
	_temp = f"{folder_path}/{experiment_params.return_prefix(not args.not_adapt, args.lb, args.ub, args.phase+1)}"
	print(_temp)
	if not os.path.exists(_temp):
		os.makedirs(_temp)
	np.savez(f"{_temp}/params.npz", params=params, N_perparam=len(nstds[args.phase]), setting_0=setting_0_perphase*(args.phase+1))


if __name__ == '__main__':
	epoch_min = 50
	epoch_max = 200
	setting_0_perphase = 10000

	exp_params = experiment_params()
	post_process_trained_3dmodel(exp_params, epoch_min, epoch_max, setting_0_perphase)