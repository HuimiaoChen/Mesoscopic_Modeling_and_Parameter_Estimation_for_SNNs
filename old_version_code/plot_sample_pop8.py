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

## sample 2
pops = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
pops_prop = np.array([1, 1, 1, -1, 1, -1, 1, 1])

labels = [
    np.array([[ 0.0516045 ,  0.05720543,  0.06084777, -0.        ,  0.        ,
        -0.32204016,  0.        ,  0.10790706],
       [ 0.        ,  0.10836798,  0.12072769, -0.        ,  0.        ,
        -0.        ,  0.09407883,  0.07225284],
       [ 0.        ,  0.07630352,  0.07955077, -0.35183934,  0.11150024,
        -0.        ,  0.05920743,  0.10459157],
       [ 0.06133567,  0.07528349,  0.        , -0.45268619,  0.0857895 ,
        -0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -0.        ,  0.04547897,
        -0.        ,  0.        ,  0.        ],
       [ 0.1220797 ,  0.        ,  0.        , -0.45933959,  0.        ,
        -0.45796471,  0.08864084,  0.        ],
       [ 0.06942852,  0.        ,  0.05952505, -0.19352493,  0.12214448,
        -0.3697075 ,  0.12425616,  0.04313796],
       [ 0.11261817,  0.07974789,  0.10270948, -0.        ,  0.06472168,
        -0.        ,  0.        ,  0.06500893]]),
    np.array([43.72796151, 48.49630203, 49.12333911, 46.4466335 , 57.00290153,
       55.24674576, 55.04336712, 55.99558428]),
    np.array([35.51168267, 17.44347701, 24.95948162, 29.29519197, 26.66679224,
       11.85925934, 24.74424161, 27.86041134]),
    np.array([19.89523273, 12.79930693, 24.18124386, 26.43159331, 21.54451978,
       18.83328877, 20.23009722, 18.43618004]),
    np.array([[1152.53595202],
       [1152.53595202],
       [1152.53595202],
       [  99.34984217],
       [1152.53595202],
       [  99.34984217],
       [1152.53595202],
       [1152.53595202]]),
    np.array([[1146.80287747],
       [1146.80287747],
       [1146.80287747],
       [ 651.40254886],
       [1146.80287747],
       [ 651.40254886],
       [1146.80287747],
       [1146.80287747]])
]
outputs = [
    np.array([[-0.29830006,  0.13725553, -0.04982918, -0.23339447, -0.05263552,
        -0.5390272 , -0.13476285, -0.11326584],
       [ 0.09849138, -0.11540221, -0.15543403, -0.09527891, -0.0584644 ,
        -0.6280002 , -0.03857533,  0.22197315],
       [ 0.06840274,  0.13466316, -0.2883627 , -0.14694259,  0.18637204,
        -0.16179621, -0.18379678, -0.1482398 ],
       [-0.15292707,  0.09149227,  0.02541979, -0.463426  ,  0.02494115,
        -0.3856435 ,  0.02582808,  0.11836892],
       [-0.03009411,  0.05819826, -0.01347397, -0.15961856, -0.30356926,
        -0.23024252,  0.07799223,  0.02511766],
       [-0.06130965, -0.14346851, -0.06757445, -0.1787715 ,  0.00827284,
        -0.58278227,  0.0487883 ,  0.27748844],
       [ 0.01957539, -0.22447674, -0.27879745, -0.1413327 ,  0.01801507,
        -0.15536605, -0.15199538, -0.00998754],
       [-0.09490723, -0.13296348, -0.03854386, -0.31497562,  0.06161088,
        -0.31780866, -0.00587337,  0.09975605]], dtype=np.float32),
    np.array([49.85455 , 47.780357, 48.4585  , 42.75139 , 48.118263, 55.190746,
       54.767643, 58.310253], dtype=np.float32),
    np.array([39.885475, 24.795326, 25.222504, 32.39124 , 27.760277, 12.867373,
       26.318752, 29.035992], dtype=np.float32),
    np.array([20.78475 , 10.845005, 21.9704  , 23.940414, 19.027822, 18.654942,
       16.403366, 16.978064], dtype=np.float32),
    np.array([[ 970.408  ],
       [1270.9257 ],
       [ 764.437  ],
       [ 473.47745],
       [1075.9495 ],
       [ 317.5207 ],
       [1180.8855 ],
       [1042.6996 ]], dtype=np.float32),
    np.array([[1311.5981 ],
       [1301.0137 ],
       [1140.8992 ],
       [ 845.8237 ],
       [ 926.2055 ],
       [ 815.45044],
       [1348.8698 ],
       [1116.717  ]], dtype=np.float32)
]

J_syn, mu, tau_m, V_th, J_theta, tau_theta = labels
# _, mu, tau_m, V_th, J_theta, tau_theta = outputs

pconn = np.where(labels[0] != 0, 1, 0)
pconn_coeff = 0.6
pconn = pconn * pconn_coeff

seed_num = 1
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


t = t/1000

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Define the size of labels and ticks
labelsize = 18
ticksize = 16

# Define the legend names
legends = ["pop 1", "pop 2", "pop 3"]

plt.figure(figsize=(10*0.8, 8*0.8))  # Make the figure larger
for popi, legend in zip([0,1,2], legends): #range(len(pops)):
    plt.plot(t[:20000], gaussian_filter1d(Abar[:20000,popi],10) * 1000, label=legend)  # plot instantaneous population rates (in Hz)
plt.ylabel('Population activities (Hz)', fontsize=labelsize)
plt.xlabel('Time (s)', fontsize=labelsize)

# Increase the size of the ticks
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

# Add legend
plt.legend(fontsize=labelsize)

# Save the figure to a PDF file
plt.savefig('pop8_sample_2_label.pdf', format='pdf')
# plt.savefig('pop8_sample_2_estimation.pdf', format='pdf')

# Define the legend names
legends = ["pop 1", "pop 2", "pop 3"]

plt.figure(figsize=(10*0.8, 8*0.8))  # Make the figure larger
for popi, legend in zip([4,6,7], legends): #range(len(pops)):
    plt.plot(t[:20000], gaussian_filter1d(A_N[:20000,popi],10) * 1000, label=legend)  # plot instantaneous population rates (in Hz)
plt.ylabel('Population activities (Hz)', fontsize=labelsize)
plt.xlabel('Time (s)', fontsize=labelsize)

# Increase the size of the ticks
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

# Add legend
plt.legend(fontsize=labelsize)

# Save the figure to a PDF file
plt.savefig('pop8_sample_2_label_1.pdf', format='pdf')
# plt.savefig('pop8_sample_2_estimation.pdf', format='pdf')


###### estimation

## sample 2
pops = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
pops_prop = np.array([1, 1, 1, -1, 1, -1, 1, 1])

labels = [
    np.array([[ 0.0516045 ,  0.05720543,  0.06084777, -0.        ,  0.        ,
        -0.32204016,  0.        ,  0.10790706],
       [ 0.        ,  0.10836798,  0.12072769, -0.        ,  0.        ,
        -0.        ,  0.09407883,  0.07225284],
       [ 0.        ,  0.07630352,  0.07955077, -0.35183934,  0.11150024,
        -0.        ,  0.05920743,  0.10459157],
       [ 0.06133567,  0.07528349,  0.        , -0.45268619,  0.0857895 ,
        -0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -0.        ,  0.04547897,
        -0.        ,  0.        ,  0.        ],
       [ 0.1220797 ,  0.        ,  0.        , -0.45933959,  0.        ,
        -0.45796471,  0.08864084,  0.        ],
       [ 0.06942852,  0.        ,  0.05952505, -0.19352493,  0.12214448,
        -0.3697075 ,  0.12425616,  0.04313796],
       [ 0.11261817,  0.07974789,  0.10270948, -0.        ,  0.06472168,
        -0.        ,  0.        ,  0.06500893]]),
    np.array([43.72796151, 48.49630203, 49.12333911, 46.4466335 , 57.00290153,
       55.24674576, 55.04336712, 55.99558428]),
    np.array([35.51168267, 17.44347701, 24.95948162, 29.29519197, 26.66679224,
       11.85925934, 24.74424161, 27.86041134]),
    np.array([19.89523273, 12.79930693, 24.18124386, 26.43159331, 21.54451978,
       18.83328877, 20.23009722, 18.43618004]),
    np.array([[1152.53595202],
       [1152.53595202],
       [1152.53595202],
       [  99.34984217],
       [1152.53595202],
       [  99.34984217],
       [1152.53595202],
       [1152.53595202]]),
    np.array([[1146.80287747],
       [1146.80287747],
       [1146.80287747],
       [ 651.40254886],
       [1146.80287747],
       [ 651.40254886],
       [1146.80287747],
       [1146.80287747]])
]
outputs = [
    np.array([[-0.29830006,  0.13725553, -0.04982918, -0.23339447, -0.05263552,
        -0.5390272 , -0.13476285, -0.11326584],
       [ 0.09849138, -0.11540221, -0.15543403, -0.09527891, -0.0584644 ,
        -0.6280002 , -0.03857533,  0.22197315],
       [ 0.06840274,  0.13466316, -0.2883627 , -0.14694259,  0.18637204,
        -0.16179621, -0.18379678, -0.1482398 ],
       [-0.15292707,  0.09149227,  0.02541979, -0.463426  ,  0.02494115,
        -0.3856435 ,  0.02582808,  0.11836892],
       [-0.03009411,  0.05819826, -0.01347397, -0.15961856, -0.30356926,
        -0.23024252,  0.07799223,  0.02511766],
       [-0.06130965, -0.14346851, -0.06757445, -0.1787715 ,  0.00827284,
        -0.58278227,  0.0487883 ,  0.27748844],
       [ 0.01957539, -0.22447674, -0.27879745, -0.1413327 ,  0.01801507,
        -0.15536605, -0.15199538, -0.00998754],
       [-0.09490723, -0.13296348, -0.03854386, -0.31497562,  0.06161088,
        -0.31780866, -0.00587337,  0.09975605]], dtype=np.float32),
    np.array([49.85455 , 47.780357, 48.4585  , 42.75139 , 48.118263, 55.190746,
       54.767643, 58.310253], dtype=np.float32),
    np.array([39.885475, 24.795326, 25.222504, 32.39124 , 27.760277, 12.867373,
       26.318752, 29.035992], dtype=np.float32),
    np.array([20.78475 , 10.845005, 21.9704  , 23.940414, 19.027822, 18.654942,
       16.403366, 16.978064], dtype=np.float32),
    np.array([[ 970.408  ],
       [1270.9257 ],
       [ 764.437  ],
       [ 473.47745],
       [1075.9495 ],
       [ 317.5207 ],
       [1180.8855 ],
       [1042.6996 ]], dtype=np.float32),
    np.array([[1311.5981 ],
       [1301.0137 ],
       [1140.8992 ],
       [ 845.8237 ],
       [ 926.2055 ],
       [ 815.45044],
       [1348.8698 ],
       [1116.717  ]], dtype=np.float32)
]

J_syn, mu, tau_m, V_th, J_theta, tau_theta = labels
_, mu, tau_m, V_th, J_theta, tau_theta = outputs

pconn = np.where(labels[0] != 0, 1, 0)
pconn_coeff = 0.6
pconn = pconn * pconn_coeff

seed_num = 1
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


t = t/1000

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Define the size of labels and ticks
labelsize = 18
ticksize = 16

# Define the legend names
legends = ["pop 1", "pop 2", "pop 3"]

plt.figure(figsize=(10*0.8, 8*0.8))  # Make the figure larger
for popi, legend in zip([0,1,2], legends): #range(len(pops)):
    plt.plot(t[:20000], gaussian_filter1d(Abar[:20000,popi],10) * 1000, label=legend)  # plot instantaneous population rates (in Hz)
plt.ylabel('Population activities (Hz)', fontsize=labelsize)
plt.xlabel('Time (s)', fontsize=labelsize)

# Increase the size of the ticks
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

# Add legend
plt.legend(fontsize=labelsize)

# Save the figure to a PDF file
# plt.savefig('pop8_sample_2_label.pdf', format='pdf')
plt.savefig('pop8_sample_2_estimation.pdf', format='pdf')

# Define the legend names
legends = ["pop 1", "pop 2", "pop 3"]

plt.figure(figsize=(10*0.8, 8*0.8))  # Make the figure larger
for popi, legend in zip([4,6,7], legends): #range(len(pops)):
    plt.plot(t[:20000], gaussian_filter1d(A_N[:20000,popi],10) * 1000, label=legend)  # plot instantaneous population rates (in Hz)
plt.ylabel('Population activities (Hz)', fontsize=labelsize)
plt.xlabel('Time (s)', fontsize=labelsize)

# Increase the size of the ticks
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

# Add legend
plt.legend(fontsize=labelsize)

# Save the figure to a PDF file
# plt.savefig('pop8_sample_2_label.pdf', format='pdf')
plt.savefig('pop8_sample_2_estimation_1.pdf', format='pdf')
