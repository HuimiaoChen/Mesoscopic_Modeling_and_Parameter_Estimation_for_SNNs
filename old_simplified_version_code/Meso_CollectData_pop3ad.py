import numpy as np
import matplotlib.pyplot as plt
import nest
import time
import json
import os, sys
from scipy.ndimage import gaussian_filter1d

from Meso_CollectData_func_pop3 import mesoscopic
from utils import *


def simulate_3ddata(exp_params):
    args = exp_params.args
    data_folder, _, _, _, _ = exp_params.return_folders()

    generate_GTdata = (args.lb==1.0) and (args.ub==1.0)
    _prefix = "GTdata" if generate_GTdata else "data"

    #### params.npz should be ready
    params_dict = np.load(f"{data_folder}/params.npz")
    for setting in range(args.setting_0, args.setting_0+args.setting_N):
        if setting-params_dict['setting_0'] >= len(params_dict['params']):
            continue

        print(f"current setting is {setting}")       

        pops, pops_prop, pconn, tau_sfa_exc, tau_sfa_inh, J_sfa_exc, J_sfa_inh, tau_theta, J_theta = get_gt_3dotherparams() 

        # pops = np.array([400, 200, 400])
        # pops_prop = np.array([1, -1, 1]) # np.random.choice([1, -1], size=len(pops)) # # 1: excitatory, -1: inhibitory

        # pconn = np.array([[1, 1, 0],
        #       [1, 1, 1],
        #       [0, 1, 1]]) # np.random.randint(0, 2, (len(pops), len(pops)))
        # np.fill_diagonal(pconn, 1)
        # pconn_coeff = 1
        # pconn = pconn * pconn_coeff

        gt_param = get_gt_3dparams()
        param = params_dict['params'][setting-params_dict['setting_0']]
        # gtJ = 0.096; gtg = 0.384/ 0.096
        if generate_GTdata:
            J = gt_param[0] #  # excitatory synaptic weight in mV, w^{αβ} in the paper
            g = gt_param[1] #   # inhibition-to-excitation ratio, -g*J is the weight for inhibitory signals
        else:
            J = param[0] # 0.096 #  # excitatory synaptic weight in mV, w^{αβ} in the paper
            g = param[1] # 0.384/ 0.096 #   # inhibition-to-excitation ratio, -g*J is the weight for inhibitory signals
        J_syn = np.outer(np.ones_like(pops_prop), np.where(pops_prop == -1, -g*J, J))
        J_syn = J_syn * pconn # * np.random.uniform(0.5, 1.5, (len(pops), len(pops)))


        # gtmu = 36.; gtau_m = 20; gtV_th = 15.
        if generate_GTdata:
            mu = gt_param[2] * np.ones(len(pops)) # # V_rest + I_external * R
            tau_m = gt_param[3] * np.ones(len(pops)) #  # membrane time constant
            V_th = gt_param[4] * np.ones(len(pops)) #  # baseline threshold (non-accumulating part)
        else:
            mu = param[2] * np.ones(len(pops)) # np.random.uniform(gtmu*args.lb, gtmu*args.ub, len(pops)) # 36. * np.ones(len(pops)) # # V_rest + I_external * R
            tau_m = param[3] * np.ones(len(pops)) # np.random.uniform(gtau_m*args.lb, gtau_m*args.ub, len(pops)) # 20 * np.ones(len(pops)) #  # membrane time constant
            V_th = param[4] * np.ones(len(pops)) # np.random.uniform(gtV_th*args.lb, gtV_th*args.ub, len(pops)) # 15. * np.ones(len(pops)) #  # baseline threshold (non-accumulating part)

        # tau_sfa_exc = [1000.] # [np.random.uniform(80, 1500)] #  # threshold adaptation time constants of excitatory neurons
        # tau_sfa_inh = [1000.] # [np.random.uniform(80, 1500)] #  # threshold adaptation time constants of inhibitory neurons
        # J_sfa_exc = [100.] # [np.random.uniform(80, 1500)]  #  # adaptation strength: size of feedback kernel theta (= area under exponential) in mV*ms
        # J_sfa_inh = [100.] #  [np.random.uniform(80, 1500)] #  # in mV*ms
        # tau_theta = np.array([tau_sfa_exc if prop == 1 else tau_sfa_inh for prop in pops_prop])
        # J_theta = np.array([J_sfa_exc if prop == 1 else J_sfa_inh for prop in pops_prop])

        J_syn_list = J_syn.tolist()
        mu_list = mu.tolist()
        tau_m_list = tau_m.tolist()
        V_th_list = V_th.tolist()
        J_theta_list = J_theta.tolist()
        tau_theta_list = tau_theta.tolist()

        for seed_num in range(1,1+args.seed_N):
            # if os.path.exists(f"{data_folder}/{_prefix}_{setting}-{seed_num}.json"):
            #     continue
            A_N, Abar, elapsed_time, t = mesoscopic(pops=pops, 
                                  pops_prop=pops_prop, 
                                  connect_mat=J_syn, 
                                  mu=mu, 
                                  tau_m=tau_m, 
                                  V_th=V_th, 
                                  J_theta=J_theta, 
                                  tau_theta=tau_theta,
                                  pconn=pconn,
                                  adapt=not args.not_adapt,
                                  seed=seed_num,
                                  t_end=args.sim_t_end)
            print(elapsed_time)
            A_N_list = A_N.tolist()
            Abar_list = Abar.tolist()
            t_list = t.tolist()
            data_dict = {"setting": setting, 
                    "seed_num": seed_num, 
                    "J_syn": J_syn_list, 
                    "mu": mu_list, 
                    "tau_m": tau_m_list, 
                    "V_th": V_th_list, 
                    "J_theta": J_theta_list, 
                    "tau_theta": tau_theta_list,
                    "A_N": A_N_list,
                    "Abar": Abar_list,
                    "t": t_list,
                    "elapsed_time": elapsed_time
                    }
            json_str = json.dumps(data_dict)

            filename = f"{data_folder}/{_prefix}_{setting}-{seed_num}.json"
            with open(filename, 'w') as json_file:
                json_file.write(json_str)

            if args.plot_image:
                if seed_num <= 2:
                    plt.figure()
                    for popi in [0,2]:
                        plt.plot(t, gaussian_filter1d(Abar[:,popi],10) * 1000)  # plot instantaneous population rates (in Hz)
                    plt.ylabel(r'$\bar A$ [Hz]')
                    plt.xlabel('time [ms]')
                    plt.title('Population activities (mesoscopic sim.)')
                    plt.savefig(f"{data_folder}/{_prefix}_{setting}-{seed_num}.png")
                    # plt.show()


if __name__ == '__main__':
    exp_params = experiment_params()
    simulate_3ddata(exp_params)


''' Depracated
folder_name = "pop3_data_with_adapt_test"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for setting in range(10001, 11001):
    print(f"current setting is {setting}")

    pops = np.array([400, 200, 400])
    pops_prop = np.array([1, -1, 1]) # np.random.choice([1, -1], size=len(pops)) # 1: excitatory, -1: inhibitory

    pconn = np.array([[1, 1, 0],
          [1, 1, 1],
          [0, 1, 1]]) # np.random.randint(0, 2, (len(pops), len(pops)))
    np.fill_diagonal(pconn, 1)

    J = 0.096 # np.random.uniform(0.06, 0.3)  # excitatory synaptic weight in mV, w^{αβ} in the paper
    g = 0.384/ 0.096 # np.random.uniform(3, 6)   # inhibition-to-excitation ratio, -g*J is the weight for inhibitory signals
    J_syn = np.outer(np.ones_like(pops_prop), np.where(pops_prop == -1, -g*J, J))
    J_syn = J_syn * pconn # * np.random.uniform(0.5, 1.5, (len(pops), len(pops)))

    pconn_coeff = 1
    pconn = pconn * pconn_coeff

    mu = 36. * np.ones(len(pops)) # np.random.uniform(20, 60, len(pops)) # V_rest + I_external * R
    tau_m = 20 * np.ones(len(pops)) # np.random.uniform(10, 40, len(pops))  # membrane time constant
    V_th = 15. * np.ones(len(pops)) # np.random.uniform(10, 30, len(pops))  # baseline threshold (non-accumulating part)

    tau_sfa_exc =[1000.] # [np.random.uniform(80, 1500)]  # threshold adaptation time constants of excitatory neurons
    tau_sfa_inh = [1000.] # [np.random.uniform(80, 1500)]  # threshold adaptation time constants of inhibitory neurons
    J_sfa_exc = [100.] # [np.random.uniform(80, 1500)]   # adaptation strength: size of feedback kernel theta (= area under exponential) in mV*ms
    J_sfa_inh = [100.] # [np.random.uniform(80, 1500)]   # in mV*ms
    tau_theta = np.array([tau_sfa_exc if prop == 1 else tau_sfa_inh for prop in pops_prop])
    J_theta = np.array([J_sfa_exc if prop == 1 else J_sfa_inh for prop in pops_prop])

    J_syn_list = J_syn.tolist()
    mu_list = mu.tolist()
    tau_m_list = tau_m.tolist()
    V_th_list = V_th.tolist()
    J_theta_list = J_theta.tolist()
    tau_theta_list = tau_theta.tolist()

    for seed_num in range(1, 21):
        A_N, Abar, elapsed_time, t = mesoscopic(pops=pops, 
                              pops_prop=pops_prop, 
                              connect_mat=J_syn, 
                              mu=mu, 
                              tau_m=tau_m, 
                              V_th=V_th, 
                              J_theta=J_theta, 
                              tau_theta=tau_theta,
                              pconn=pconn,
                              adapt=True,
                              seed=seed_num)
        A_N_list = A_N.tolist()
        Abar_list = Abar.tolist()
        t_list = t.tolist()
        data_dict = {"setting": setting, 
                "seed_num": seed_num, 
                "J_syn": J_syn_list, 
                "mu": mu_list, 
                "tau_m": tau_m_list, 
                "V_th": V_th_list, 
                "J_theta": J_theta_list, 
                "tau_theta": tau_theta_list,
                "A_N": A_N_list,
                "Abar": Abar_list,
                "t": t_list,
                "elapsed_time": elapsed_time
                }
        json_str = json.dumps(data_dict)
        filename = f"{folder_name}/data_{setting}-{seed_num}.json"
        with open(filename, 'w') as json_file:
            json_file.write(json_str)

        plt.figure()
        for popi in [0,2]:
            plt.plot(t, gaussian_filter1d(Abar[:,popi],10) * 1000)  # plot instantaneous population rates (in Hz)
        plt.ylabel(r'$\bar A$ [Hz]')
        plt.xlabel('time [ms]')
        plt.title('Population activities (mesoscopic sim.)')
        print(len(A_N))
        print(elapsed_time)

'''
