'''
Here are functions for: 
(1) simualting mesoscopic models;
(2) using the model to generate data.

Author: Huimiao Chen
Date: Aug 13, 2023
'''

import nest
import json
import time
import numpy as np
import matplotlib.pyplot as plt

def meso_model(dt = 0.5, # resolution time interval
               dt_rec = 1.0, # record time interval
               t_end = 60000.0, # simulation time
               N = np.array([400, 200, 400]), # population size
               pops_prop = np.array([1, -1, 1]), # pop property, 1: excitatory, -1: inhibitory
               t_ref = 4.0 * np.ones(3), # absolute refractory period
               tau_m = 20 * np.ones(3),  # membrane time constant
               mu = 36.0 * np.ones(3),  # constant base current mu=R*(I0+Vrest), R*I0 is contant inputs
               c = 10.0 * np.ones(3),  # base rate of exponential link function
               Delta_u = 2.5 * np.ones(3),  # softness of exponential link function
               V_reset = 0.0 * np.ones(3),  # Reset potential
               V_th = 15.0 * np.ones(3),  # baseline threshold (non-accumulating part)
               tau_theta = np.array([[1000.0], [1000.0], [1000.0]]), #adaptation time constants, in ms
               J_theta = np.array([[100.0], [100.0], [100.0]]), # adaptation stregth, in mV*ms
               pconn = np.array([[1, 1, 0],
                                [1, 1, 1],
                                [0, 1, 1]]), # connectivity matrix, probability of connections
               delay = 1.0 * np.ones((3, 3)), # every two populations have a delay constant
               J_syn = np.array([[ 0.096, -0.384, 0.0], 
                                [0.096, -0.384, 0.096], 
                                [0.0, -0.384, 0.096]]), # synaptic weight matrix, positive: excitatory, negative: inhibitory
               step = [[20.] for i in range(3)],  # external inputs, jump size of mu in mV
               tstep = np.array(np.array([[30000.] for i in range(3)])),  # times of jumps of external inputs
               tau_ex = 3.0,  # synaptic time constants of excitatory connections, in ms
               tau_in = 6.0,  # synaptic time constants of inhibitory connections, in ms
               adapt = True, # if using adaptaion of thresholds, usually True
               seed = 1, # random seed
               ):
    M = len(N)
    
    start_time = time.time()

    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.resolution = dt
    nest.print_time = True
    nest.local_num_threads = 1

    t0 = nest.biological_time

    nest_pops = nest.Create('gif_pop_psc_exp', M)

    C_m = 250.  # irrelevant value for membrane capacity, cancels out in simulation
    g_L = C_m / tau_m

    if adapt: # usually dapat = True
        q_sfa_array = J_theta / tau_theta # [J_theta]= mV*ms -> [q_sfa]=mV
        # print("Adpat is True.")
    else:
        q_sfa_array = np.zeros_like(J_theta / tau_theta)
        # print("Adpat is False.")

    params = [
        {
        'C_m': C_m,
        'I_e': mu[i] * g_L[i],
        'lambda_0': c[i],  # in Hz!
        'Delta_V': Delta_u[i],
        'tau_m': tau_m[i],
        'tau_sfa': tau_theta[i],
        'q_sfa': q_sfa_array[i],  
        'V_T_star': V_th[i],
        'V_reset': V_reset[i],
        'len_kernel': -1,  
        # integer, refractory effects are accounted for up to len_kernel time steps)
        # -1 triggers automatic history size
        'N': N[i],
        't_ref': t_ref[i],
        'tau_syn_ex': max([tau_ex, dt]),
        'tau_syn_in': max([tau_in, dt]),
        'E_L': 0.0,
        }
        for i in range(M)
    ]
    nest_pops.set(params)

    # connect the populations
    g_syn = np.ones_like(J_syn)  # synaptic conductance
    for i, prop in enumerate(pops_prop):
        if prop == 1:
            g_syn[:, i] = C_m / tau_ex
        else:
            g_syn[:, i] = C_m / tau_in
    for i in range(M):
        for j in range(M):
            nest.Connect(
                nest_pops[j], 
                nest_pops[i],
                syn_spec={'weight': J_syn[i, j] * g_syn[i, j] * pconn[i, j], 
                          'delay': delay[i, j]},
            )

    # monitor the output using a multimeter, this only records with dt_rec!
    nest_mm = nest.Create('multimeter')
    nest_mm.set(record_from=['n_events', 'mean'], interval=dt_rec)
    nest.Connect(nest_mm, nest_pops)

    # monitor the output using a spike recorder
    nest_sr = []
    for i in range(M):
        nest_sr.append(nest.Create('spike_recorder'))
        nest_sr[i].time_in_steps = True
        nest.Connect(nest_pops[i], nest_sr[i], syn_spec={'weight': 1.0, 'delay': dt})

    # set initial value (at t0+dt) of step current generator to zero
    tstep = np.hstack((dt * np.ones((M, 1)), tstep))
    step = np.hstack((np.zeros((M, 1)), step))

    # create the step current devices
    nest_stepcurrent = nest.Create('step_current_generator', M)
    # set the parameters for the step currents
    for i in range(M):
        nest_stepcurrent[i].set(amplitude_times=tstep[i] + t0,
                                amplitude_values=step[i] * g_L[i],
                                origin=t0,
                                stop=t_end)
        pop_ = nest_pops[i]
        nest.Connect(nest_stepcurrent[i], pop_, syn_spec={'weight': 1.0, 'delay': dt})

    # begin simulation for output
    nest.rng_seed = seed

    t = np.arange(0., t_end, dt_rec)
    A_N = np.ones((t.size, M)) * np.nan
    Abar = np.ones_like(A_N) * np.nan

    # simulate 1 step longer to make sure all t are simulated
    nest.Simulate(t_end + dt)
    data_mm = nest_mm.events
    for i, nest_i in enumerate(nest_pops):
        a_i = data_mm['mean'][data_mm['senders'] == nest_i.global_id]
        a = a_i / N[i] / dt
        min_len = np.min([len(a), len(Abar)])
        Abar[:min_len, i] = a[:min_len]

        data_sr = nest_sr[i].get('events', 'times')
        data_sr = data_sr * dt - t0
        bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
        A = np.histogram(data_sr, bins=bins)[0] / float(N[i]) / dt_rec
        A_N[:, i] = A

    end_time = time.time()
    elapsed_time = end_time - start_time

    return A_N, Abar, elapsed_time, t


if __name__ == '__main__':
    pops = np.array([400, 200, 400])
    pops_prop = np.array([1, -1, 1]) # 1: excitatory, -1: inhibitory

    pconn = np.array([[1, 1, 0],
            [1, 1, 1],
            [0, 1, 1]])
    J = 0.096  # excitatory synaptic weight in mV
    g = 0.384/0.096   # inhibition-to-excitation ratio, -g*J is the weight for inhibitory signals
    J_syn = np.outer(np.ones_like(pops_prop), np.where(pops_prop == -1, -g*J, J))
    J_syn = J_syn * pconn

    pconn_coeff = 1.
    pconn = pconn * pconn_coeff

    mu = 36. * np.ones(len(pops)) # V_rest + I_external * R
    tau_m = 20. * np.ones(len(pops))  # membrane time constant
    V_th = 15. * np.ones(len(pops))  # baseline threshold (non-accumulating part)

    tau_sfa_exc = [1000.]  # threshold adaptation time constants of excitatory neurons
    tau_sfa_inh = [1000.]  # threshold adaptation time constants of inhibitory neurons
    J_sfa_exc = [100.]   # adaptation strength: size of feedback kernel theta (= area under exponential) in mV*ms
    J_sfa_inh = [100.]   # in mV*ms
    tau_theta = np.array([tau_sfa_exc if prop == 1 else tau_sfa_inh for prop in pops_prop])
    J_theta = np.array([J_sfa_exc if prop == 1 else J_sfa_inh for prop in pops_prop])


