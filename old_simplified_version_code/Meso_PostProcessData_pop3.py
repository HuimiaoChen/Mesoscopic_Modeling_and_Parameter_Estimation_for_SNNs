import os, sys
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import pdb
from tqdm import tqdm

from utils import *


'''
Setups of post-processing
'''
def post_process_3ddata(exp_params, mean_fr_dur=50):
    args = exp_params.args
    data_folder, post_data_folder, _, folder_path, prefix = exp_params.return_folders()

    filenames = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder) if fn.endswith('.json')]
    
    params = []
    mean_frs = []
    std_frs = []
    for i, f in tqdm(enumerate(filenames)):
        _sim = f.split('/')[-1].split('.')[0]
        if os.path.exists(f"{post_data_folder}/{_sim}.json"):
        	# continue
            with open(f"{post_data_folder}/{_sim}.json", 'r') as json_file:
                data_dict = json.loads(json_file.read())

            mean_fr = np.array(data_dict['mean_fr']).mean(0)
            std_fr = np.array(data_dict['mean_fr']).std(0)

        else:
            with open(f, 'r') as json_file:
                data_dict = json.loads(json_file.read())

            A_N = np.array(data_dict["A_N"])

            A_N_reshaped = A_N.reshape(-1, mean_fr_dur, 3)
            mean_fr = A_N.mean(0)
            std_fr = A_N_reshaped.mean(1).std(0)

        params.append([data_dict['J_syn'][0][0], data_dict['J_syn'][0][1], data_dict['mu'][0], data_dict['tau_m'][0], data_dict['V_th'][0]])
        # params.append(flat_2dlist(data_dict['J_syn']) + data_dict['mu'] + data_dict['tau_m'] + data_dict['V_th'])
        mean_frs.append(mean_fr)
        std_frs.append(std_fr)

        if i % 500 == 0:
            np.savez(f"{post_data_folder}/frs0.npz", params=np.array(params),
                mean_frs=np.array(mean_frs), std_frs=np.array(std_frs))


        # if _sim.startswith('GT'):
    	#     fig, axes = plt.subplots(1,3)
    	#     for i in range(3):
    	#         n, bins, _ = axes[i].hist(mean_fr[:,i], bins=20, density=True)
    	#         gm = GaussianMixture(n_components=GM_ncomp, random_state=0).fit(mean_fr[:,i][:,np.newaxis])
    	#         _x = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/100)
    	#         axes[i].plot(_x, np.exp(gm.score_samples(_x[:,np.newaxis])))
    	#     # plt.show()
    	#     plt.savefig(f"{post_data_folder}/{_sim}.png")


        # data_dict = {"setting": data_dict['setting'], 
        #         "seed_num": data_dict['seed_num'], 
        #         "J_syn": data_dict['J_syn'], 
        #         "mu": data_dict['mu'], 
        #         "tau_m": data_dict['tau_m'], 
        #         "V_th": data_dict['V_th'], 
        #         "J_theta": data_dict['J_theta'], 
        #         "tau_theta": data_dict['tau_theta'],
        #         "std_fr": std_fr.tolist(),
        #         "mean_fr": mean_fr.tolist(),
        #         "g": g,
        #         "mean_fr_dur": mean_fr_dur,
        #         }
        # json_str = json.dumps(data_dict)

        # with open(f"{post_data_folder}/{_sim}.json", 'w') as json_file:
        #     json_file.write(json_str)


    # filenames = [f"{post_data_folder}/{f.split('/')[-1]}" for f in filenames]
    # filenames =  np.array(filenames)
    # if not os.path.exists(f"{post_data_folder}/all_files.npz"):
    #     print("Creating all_files.npz")
    #     np.savez(f"{post_data_folder}/all_files.npz", filenames=filenames)
    # else:
    #     saved_filenames = np.load(f"{post_data_folder}/all_files.npz")['filenames']
    #     if len(saved_filenames) < len(filenames):
    #         print("Updating all_files.npz")
    #         np.savez(f"{post_data_folder}/all_files.npz", filenames=filenames)

    if args.phase > 1:
        total_data = np.load(f"{folder_path}/{experiment_params.return_prefix(not args.not_adapt, args.lb, args.ub, args.phase-1)}_post_processed/frs.npz")
        params = total_data['params'].tolist()+params
        mean_frs = total_data['mean_frs'].tolist()+mean_frs
        std_frs = total_data['std_frs'].tolist()+std_frs

    print(f"{len(params)} files in total.")
    np.savez(f"{post_data_folder}/frs.npz", params=np.array(params),
                mean_frs=np.array(mean_frs), std_frs=np.array(std_frs))


if __name__ == '__main__':
    exp_params = experiment_params()
    post_process_3ddata(exp_params)
