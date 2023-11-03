import numpy as np
import matplotlib.pyplot as plt
import nest
import time
import json
import os

from Meso_CollectData_func_pop3 import mesoscopic

folder_name = "pop3_data_with_adapt_test"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for setting in range(10001, 11001):
    print(f"current setting is {setting}")

    pops = np.array([400, 200, 400])
    pops_prop = np.random.choice([1, -1], size=len(pops)) # 1: excitatory, -1: inhibitory

    pconn = np.random.randint(0, 2, (len(pops), len(pops)))
    np.fill_diagonal(pconn, 1)

    J = np.random.uniform(0.06, 0.3)  # excitatory synaptic weight in mV, w^{αβ} in the paper
    g = np.random.uniform(3, 6)   # inhibition-to-excitation ratio, -g*J is the weight for inhibitory signals
    J_syn = np.outer(np.ones_like(pops_prop), np.where(pops_prop == -1, -g*J, J))
    J_syn = J_syn * pconn * np.random.uniform(0.5, 1.5, (len(pops), len(pops)))

    pconn_coeff = 0.6
    pconn = pconn * pconn_coeff

    mu = np.random.uniform(20, 60, len(pops)) # V_rest + I_external * R
    tau_m = np.random.uniform(10, 40, len(pops))  # membrane time constant
    V_th = np.random.uniform(10, 30, len(pops))  # baseline threshold (non-accumulating part)

    tau_sfa_exc = [np.random.uniform(80, 1500)]  # threshold adaptation time constants of excitatory neurons
    tau_sfa_inh = [np.random.uniform(80, 1500)]  # threshold adaptation time constants of inhibitory neurons
    J_sfa_exc = [np.random.uniform(80, 1500)]   # adaptation strength: size of feedback kernel theta (= area under exponential) in mV*ms
    J_sfa_inh = [np.random.uniform(80, 1500)]   # in mV*ms
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
