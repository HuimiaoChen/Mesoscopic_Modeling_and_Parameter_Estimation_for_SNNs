'''
This file is part of the new version code of SNN stimulation and parameter estimation.

The functions related to modeling are included here.

Function list
1) meso_model: the mesoscopic model;
2) micro_model: the microscopic model.

Author: Huimiao Chen
Date: Aug 13 to Oct 15, 2023
'''

import nest
import time
import numpy as np

from funcs_of_data_processing import plot_neuro_activities


# Below is the function of the mesoscopic model. 
# The default parameters are corresponding  to 3-population winner-takes-all bistability phenomenon.
def meso_model(dt = 0.5, # resolution time interval
               dt_rec = 1.0, # record time interval
               t_end = 60000.0, # simulation time
               N = np.array([400, 200, 400]), # population size
               pops_prop = np.array([1, -1, 1]), # pop property, 1: excitatory, -1: inhibitory
               # We can know pops_prop from J_syn unless there is a population unconnected to any other populations (its
               # column is with all zero elements). But here we still have pops_prop as an input so that no extra 
               # calculation is needed in the function; pops_prop is used to generate g_syn in the following code. 
               t_ref = 4.0 * np.ones(3), # absolute refractory period
               tau_m = 20 * np.ones(3),  # membrane time constant
               mu = 36.0 * np.ones(3),  # membrane base potential, V_rest + I_external * R
               # In original version (also the examples from official web), V_rest is not an input parameter, and it just
               # set 'E_L' (a param in nest, resting potential) as 0. Here, I include V_rest as an input (the third from the
               # end) whose defalut is 0, and let 'E_l' be V_rest. Then in the following code (see params = [...]), 'I_e' 
               # becomes 'I_e': (mu[i] - V_rest) * g_L[i] (originally 'I_e': mu[i] * g_L[i]).
               # Basically, V_rest and I_external * R have the same effect on the results because they are added up as one
               # term in the differential equation. That is, if mu is fixed, then changing V_rest should not affetc the 
               # results.
               c = 10.0 * np.ones(3),  # base rate of exponential link function
               Delta_u = 2.5 * np.ones(3),  # softness of exponential link function
               V_reset = 0.0 * np.ones(3),  # Reset potential 
               # here, reset potential is euqal to resting potential E_L (E_L=0 in the following code)
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
               # each column corresponds to a pre-cell and each row corresponds to a post-cell, that is,
               # the column index is for pre-cell and the row index is for post-cell.
               # Here, if connectd, we have w^EE = w^IE and w^EI = w^II; note that they can be different.
               # 
               step = [[20.] for i in range(3)],  # external inputs, jump size of mu in mV
               tstep = np.array([[30000.] for i in range(3)]),  # times of jumps of external inputs
               tau_ex = 3.0,  # synaptic time constants of excitatory connections, in ms, i.e., presynapse is excitatory
               tau_in = 6.0,  # synaptic time constants of inhibitory connections, in ms, i.e., presynapse is inhibitory
               # here, all excitatory neurons are supposed to have the same time constant, so are inhibitory neurons.
               # Actually, excitatory/inhibitory neurons can have different time constants if from different populations.
               V_rest = 0.0,
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
        print("Adpat is True.")
    else:
        q_sfa_array = np.zeros_like(J_theta / tau_theta)
        print("Adpat is False.")

    params = [
        {
        'C_m': C_m,
        'I_e': (mu[i] - V_rest) * g_L[i], # constant input current
        'lambda_0': c[i],  # in Hz!
        'Delta_V': Delta_u[i],
        'tau_m': tau_m[i],
        'tau_sfa': tau_theta[i],
        'q_sfa': q_sfa_array[i],  
        'V_T_star': V_th[i],
        'V_reset': V_reset[i],
        'len_kernel': -1,  
        # integer, refractory effects are accounted for up to len_kernel time steps)
        # -1 triggers automatic history size.
        # I think this corresponds to the content in chapter "Population equations for a finite history"
        # in paper "Towards a theory of cortical columns: From spiking neurons to interacting neural 
        # populations of finite size".
        'N': N[i],
        't_ref': t_ref[i],
        'tau_syn_ex': max([tau_ex, dt]),
        'tau_syn_in': max([tau_in, dt]),
        'E_L': V_rest, # resting potential
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


# Below is the function of the microscopic model. 
# The default parameters are corresponding  to 3-population winner-takes-all bistability phenomenon.
def micro_model(dt = 0.5, # resolution time interval
               dt_rec = 1.0, # record time interval
               t_end = 60000.0, # simulation time
               N = np.array([400, 200, 400]), # population size
               pops_prop = np.array([1, -1, 1]), # pop property, 1: excitatory, -1: inhibitory
               # We can know pops_prop from J_syn unless there is a population unconnected to any other populations (its
               # column is with all zero elements). But here we still have pops_prop as an input so that no extra 
               # calculation is needed in the function; pops_prop is used to generate g_syn in the following code. 
               t_ref = 4.0 * np.ones(3), # absolute refractory period
               tau_m = 20 * np.ones(3),  # membrane time constant
               mu = 36.0 * np.ones(3),  # membrane base potential, V_rest + I_external * R
               # In original version (also the examples from official web), V_rest is not an input parameter, and it just
               # set 'E_L' (a param in nest, resting potential) as 0. Here, I include V_rest as an input (the third from the
               # end) whose defalut is 0, and let 'E_l' be V_rest. Then in the following code (see params = [...]), 'I_e' 
               # becomes 'I_e': (mu[i] - V_rest) * g_L[i] (originally 'I_e': mu[i] * g_L[i]).
               # Basically, V_rest and I_external * R have the same effect on the results because they are added up as one
               # term in the differential equation. That is, if mu is fixed, then changing V_rest should not affetc the 
               # results.
               c = 10.0 * np.ones(3),  # base rate of exponential link function
               Delta_u = 2.5 * np.ones(3),  # softness of exponential link function
               V_reset = 0.0 * np.ones(3),  # Reset potential 
               # here, reset potential is euqal to resting potential E_L (E_L=0 in the following code)
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
               # each column corresponds to a pre-cell and each row corresponds to a post-cell, that is,
               # the column index is for pre-cell and the row index is for post-cell.
               # Here, if connectd, we have w^EE = w^IE and w^EI = w^II; note that they can be different.
               # 
               step = [[20.] for i in range(3)],  # external inputs, jump size of mu in mV
               tstep = np.array([[30000.] for i in range(3)]),  # times of jumps of external inputs
               tau_ex = 3.0,  # synaptic time constants of excitatory connections, in ms, i.e., presynapse is excitatory
               tau_in = 6.0,  # synaptic time constants of inhibitory connections, in ms, i.e., presynapse is inhibitory
               # here, all excitatory neurons are supposed to have the same time constant, so are inhibitory neurons.
               # Actually, excitatory/inhibitory neurons can have different time constants if from different populations.
               V_rest = 0.0,
               adapt = True, # if using adaptaion of thresholds, usually True
               seed = 1, # random seed
               ):
    M = len(N) # numbers of populations

    start_time = time.time()

    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.resolution = dt
    nest.print_time = True
    nest.local_num_threads = 1

    t0 = nest.biological_time

    nest_pops = []
    for k in range(M):
        nest_pops.append(nest.Create('gif_psc_exp', N[k]))

    C_m = 250.  # irrelevant value for membrane capacity, cancels out in simulation
    g_L = C_m / tau_m

    if adapt:
        q_sfa_array = J_theta / tau_theta # [J_theta]= mV*ms -> [q_sfa]=mV
        print("Adpat is True.")
    else:
        q_sfa_array = np.zeros_like(J_theta / tau_theta)
        print("Adpat is False.")

    # set single neuron properties
    for i in range(M):
        nest_pops[i].set(C_m=C_m,
                I_e=mu[i] * g_L[i],
                lambda_0=c[i],
                Delta_V=Delta_u[i],
                g_L=g_L[i],
                tau_sfa=tau_theta[i],
                q_sfa=q_sfa_array[i],
                V_T_star=V_th[i],
                V_reset=V_reset[i],
                t_ref=t_ref[i],
                tau_syn_ex=max([tau_ex, dt]),
                tau_syn_in=max([tau_in, dt]),
                E_L=V_rest, # resting potential
                V_m=0.) # initial membrane potential 

    # connect the populations
    g_syn = np.ones_like(J_syn)  # synaptic conductance
    for i, prop in enumerate(pops_prop):
        if prop == 1:
            g_syn[:, i] = C_m / tau_ex
        else:
            g_syn[:, i] = C_m / tau_in
        
    # connect the populations
    for i, nest_i in enumerate(nest_pops):
        for j, nest_j in enumerate(nest_pops):
            if np.allclose(pconn[i, j], 1.):
                conn_spec = {'rule': 'all_to_all'}
            else:
                conn_spec = {
                    'rule': 'fixed_indegree', 'indegree': int(pconn[i, j] * N[j])}

            nest.Connect(nest_j, nest_i,
                        conn_spec,
                        syn_spec={'weight': J_syn[i, j] * g_syn[i, j],
                                  'delay': delay[i, j]})

    # monitor the output using a multimeter and a spike recorder
    nest_sr = []
    for i, nest_i in enumerate(nest_pops):
        nest_sr.append(nest.Create('spike_recorder'))
        nest_sr[i].time_in_steps = True

        # record all spikes from population to compute population activity
        nest.Connect(nest_i, nest_sr[i], syn_spec={'weight': 1., 'delay': dt})

    # Nrecord = [5 for i in range(M)]    # for each population "i" the first Nrecord[i] neurons are recorded
    Nrecord = [10 for i in range(M)]    # record more neurons, possibly more accurate; seems no help
    # Nrecord = [int(_/2) for _ in N]    # record more neurons, possibly more accurate; seems no help
    nest_mm_Vm = []
    for i, nest_i in enumerate(nest_pops):
        nest_mm_Vm.append(nest.Create('multimeter'))
        nest_mm_Vm[i].set(record_from=['V_m'], interval=dt_rec)
        if Nrecord[i] != 0:
            nest.Connect(nest_mm_Vm[i], nest_i[:Nrecord[i]], syn_spec={'weight': 1., 'delay': dt})

    # set initial value (at t0+dt) of step current generator to zero
    tstep = np.hstack((dt * np.ones((M, 1)), tstep))
    step = np.hstack((np.zeros((M, 1)), step))

    # create the step current devices if they do not exist already
    nest_stepcurrent = nest.Create('step_current_generator', M)
    # set the parameters for the step currents
    for i in range(M):
        nest_stepcurrent[i].set(amplitude_times=tstep[i] + t0,
                                amplitude_values=step[i] * g_L[i],
                                origin=t0,
                                stop=t_end)
        nest_stepcurrent[i].set(amplitude_times=tstep[i] + t0,
                                amplitude_values=step[i] * g_L[i],
                                origin=t0,
                                stop=t_end)
        # optionally a stopping time may be added by: 'stop': sim_T + t0
        pop_ = nest_pops[i]
        nest.Connect(nest_stepcurrent[i], pop_, syn_spec={'weight': 1., 'delay': dt})

    # start the microscopic simulation
    nest.rng_seed = seed

    t = np.arange(0., t_end, dt_rec)
    A_N = np.ones((t.size, M)) * np.nan

    # simulate 1 step longer to make sure all t are simulated
    nest.Simulate(t_end + dt)

    for i in range(len(nest_pops)):
        data_sr = nest_sr[i].get('events', 'times') * dt - t0
        bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
        A = np.histogram(data_sr, bins=bins)[0] / float(N[i]) / dt_rec
        A_N[:, i] = A * 1000  # in Hz

    end_time = time.time()
    elapsed_time = end_time - start_time

    return A_N, elapsed_time, t


if __name__ == '__main__':

    # set parameters
    pops = np.array([400, 200, 400])
    pops_prop = np.array([1, -1, 1]) # 1: excitatory, -1: inhibitory

    pconn = np.array([[1, 1, 0],
            [1, 1, 1],
            [0, 1, 1]])
    J = 0.096  # excitatory synaptic weight in mV
    g = 0.384/0.096   # inhibition-to-excitation ratio, -g*J is the weight for inhibitory signals
    J_syn = np.outer(np.ones_like(pops_prop), np.where(pops_prop == -1, -g*J, J))
    J_syn = J_syn * pconn

    pconn_coeff = 1. # probability of connection
    pconn = pconn * pconn_coeff # from whether connected to connected probabilities

    mu = 36. * np.ones(len(pops)) # V_rest + I_external * R
    tau_m = 20. * np.ones(len(pops))  # membrane time constant
    V_th = 15. * np.ones(len(pops))  # baseline threshold (non-accumulating part)

    tau_sfa_exc = [1000.]  # threshold adaptation time constants of excitatory neurons
    tau_sfa_inh = [1000.]  # threshold adaptation time constants of inhibitory neurons
    J_sfa_exc = [100.]   # adaptation strength: size of feedback kernel theta (= area under exponential) in mV*ms
    J_sfa_inh = [100.]   # in mV*ms
    tau_theta = np.array([tau_sfa_exc if prop == 1 else tau_sfa_inh for prop in pops_prop])
    J_theta = np.array([J_sfa_exc if prop == 1 else J_sfa_inh for prop in pops_prop])


    # run mesoscopic model
    A_N, Abar, elapsed_time, t = meso_model(N=pops,
                                            pops_prop=pops_prop,
                                            pconn=pconn,
                                            J_syn=J_syn,
                                            mu=mu,
                                            tau_m=tau_m,
                                            V_th=V_th,
                                            tau_theta=tau_theta,
                                            J_theta=J_theta,
                                            )
    
    results = [A_N, Abar, elapsed_time, t]
    
    plot_neuro_activities(results, "meso", save=True)


    # run microscopic model
    A_N, elapsed_time, t = micro_model(N=pops,
                                        pops_prop=pops_prop,
                                        pconn=pconn,
                                        J_syn=J_syn,
                                        mu=mu,
                                        tau_m=tau_m,
                                        V_th=V_th,
                                        tau_theta=tau_theta,
                                        J_theta=J_theta,
                                        )
    
    results = [A_N, elapsed_time, t]
    
    plot_neuro_activities(results, "micro", save=True)
    