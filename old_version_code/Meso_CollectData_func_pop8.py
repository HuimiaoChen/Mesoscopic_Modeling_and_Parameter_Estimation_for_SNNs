# Loading the necessary modules:
import numpy as np
import matplotlib.pyplot as plt
import nest
import time
import json

def mesoscopic(pops, pops_prop, connect_mat, mu, tau_m, V_th, 
               J_theta, tau_theta, pconn, adapt=True, seed=1):
    # simulation time interval and record time interval
    dt = 0.5
    dt_rec = 1.

    # simulation time
    t_end = 20000.

    # populations and their neuron numbers
    N = pops
    M = len(N) # numbers of populations

    # neuronal parameters
    t_ref = 2. * np.ones(M)  # absolute refractory period
    V_reset = 0. * np.ones(M)    # Reset potential

    # exponential link function for the conditional intensity (also called hazard rate, escape rate or conditional rate)
    c = 10. * np.ones(M)     # base rate of exponential link function
    Delta_u = 5. * np.ones(M)   # softness of exponential link function

    # connectivity
    # pconn = pconn_coeff * np.ones((M, M)) # probability of connections
    delay = 1.5 * np.ones((M, M)) # every two populations have a delay constant
    J_syn = connect_mat # synaptic weights

    # step current input
    step = [[0., 0.] for i in range(M)]  # jump size of mu in mV
    tstep = np.array([[60., 90.] for i in range(M)])  # times of jumps
    step[2] = [19., 0.]
    step[3] = [11.964, 0.]
    step[6] = [9.896, 0.]
    step[7] = [3.788, 0.]

    # synaptic time constants of excitatory and inhibitory connections, tau_s in the paper
    # for calculating post-synaptic currents caused by each spike of presynaptic neurons
    tau_ex = 0.5  # in ms,
    tau_in = 0.5  # in ms

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

    if adapt:
        q_sfa_array = J_theta / tau_theta # [J_theta]= mV*ms -> [q_sfa]=mV
        print("Adpat is True.")
    else:
        q_sfa_array = np.zeros_like(J_theta / tau_theta)
        print("Adpat is False.")

    params = [{
        'C_m': C_m,
        'I_e': mu[i] * g_L[i],
        'lambda_0': c[i],  # in Hz!
        'Delta_V': Delta_u[i],
        'tau_m': tau_m[i],
        'tau_sfa': tau_theta[i],
        'q_sfa': q_sfa_array[i],  
        'V_T_star': V_th[i],
        'V_reset': V_reset[i],
        'len_kernel': -1,  # -1 triggers automatic history size
        'N': N[i],
        't_ref': t_ref[i],
        'tau_syn_ex': max([tau_ex, dt]),
        'tau_syn_in': max([tau_in, dt]),
        'E_L': 0.
    } for i in range(M)]
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
            nest.Connect(nest_pops[j], nest_pops[i],
                        syn_spec={'weight': J_syn[i, j] * g_syn[i, j] * pconn[i, j],
                                  'delay': delay[i, j]})

    # monitor the output using a multimeter, this only records with dt_rec!
    nest_mm = nest.Create('multimeter')
    nest_mm.set(record_from=['n_events', 'mean'], interval=dt_rec)
    nest.Connect(nest_mm, nest_pops)

    # monitor the output using a spike recorder
    nest_sr = []
    for i in range(M):
        nest_sr.append(nest.Create('spike_recorder'))
        nest_sr[i].time_in_steps = True
        nest.Connect(nest_pops[i], nest_sr[i], syn_spec={'weight': 1., 'delay': dt})

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
        nest.Connect(nest_stepcurrent[i], pop_, syn_spec={'weight': 1., 'delay': dt})

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