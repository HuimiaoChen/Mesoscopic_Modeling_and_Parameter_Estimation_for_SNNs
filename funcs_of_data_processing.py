'''
This file is part of the new version code of SNN stimulation and parameter estimation.

The functions related to data processing and visualization are included here.

Function list
1) save_neuro_activitie: save neuro activities of micro or meso models;
2) load_neuro_activities: load neuro activities of micro or meso models;
3) plot_neuro_activities: save and plot neuro activities of micro or meso models.

Author: Huimiao Chen
Date: Aug 13 to Oct 15, 2023
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import datetime


# save neuro activities of micro or meso models
def save_neuro_activities(results, # A_N, Abar, elapsed_time, t for "meso"
                                   # A_N, elapsed_time, t for "micro" 
                          type, # "meso" or "micro"
                          filename, # data file name to save
                          ):
    if type == "meso":
        A_N, Abar, elapsed_time, t = results

        # Save the data to a JSON file
        data_to_save = {
            "A_N": A_N.tolist(),
            "Abar": Abar.tolist(),
            "elapsed_time": elapsed_time,
            "t": t.tolist()
        }
        with open(filename, 'w') as json_file:
            json.dump(data_to_save, json_file)

    if type == "micro":
        A_N, elapsed_time, t = results

        # Save the data to a JSON file
        data_to_save = {
            "A_N": A_N.tolist(),
            "elapsed_time": elapsed_time,
            "t": t.tolist()
        }
        with open(filename, 'w') as json_file:
            json.dump(data_to_save, json_file)

    return


# load neuro activities of micro or meso models
def load_neuro_activities(filename, # data file name to load
                          type, # "meso" or "micro"
                          ):
    if type == "meso":
         # Load the data from the JSON file
        with open(filename, 'r') as json_file:
            loaded_data = json.load(json_file)
        A_N = np.array(loaded_data["A_N"])
        Abar = np.array(loaded_data["Abar"])
        elapsed_time = loaded_data["elapsed_time"]
        t = np.array(loaded_data["t"])

        return A_N, Abar, elapsed_time, t
    
    if type == "micro":
         # Load the data from the JSON file
        with open(filename, 'r') as json_file:
            loaded_data = json.load(json_file)
        A_N = np.array(loaded_data["A_N"])
        elapsed_time = loaded_data["elapsed_time"]
        t = np.array(loaded_data["t"])

        return A_N, elapsed_time, t


# save and plot neuro activities of micro or meso models
def plot_neuro_activities(results, # A_N, Abar, elapsed_time, t for "meso"
                                   # A_N, elapsed_time, t for "micro" 
                          type, # "meso" or "micro"
                          save=True, # whether saving data (plotted figures are by default saved)
                          ):
    '''
    (1) save data into json files if save=True;
    (2) plot raw neural activities and save;
    (3) plot Guassian filter smoothed neural activities and save;
    (4) plot moving average smoothed activities and save;
    '''

    if type == "meso":
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y%m%d-%H%M%S")
        
        filename = f'data_{type}_{time_string}.json'

        if save:
            # Save the data to a JSON file
            save_neuro_activities(results, type, filename)

            # Load the data from the JSON file
            A_N, Abar, elapsed_time, t = load_neuro_activities(filename, type)

            print(f"The simulation time of this data is {elapsed_time}")
        else:
            A_N, Abar, elapsed_time, t = results

            print(f"The simulation time of this data is {elapsed_time}")

        # plot raw data
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, A_N * 1000)  # plot population activities (in Hz)
        plt.ylabel(r"$A_N$ [Hz]")
        plt.title("Population activities (raw, mesoscopic model)")
        plt.subplot(2, 1, 2)
        plt.plot(t, Abar * 1000)  # plot instantaneous population rates (in Hz)
        plt.ylabel(r"$\bar A$ [Hz]")
        plt.xlabel("time [ms]")
        plt.savefig(f"Population activities (raw, mesoscopic model)_{time_string}.pdf", format="pdf")
        plt.show()

        # data processing method 1: Guassian filter -- using Guassian kernel convolution for smoothing
        pop_num = A_N.shape[1]
        A_N_GuassianConv = np.zeros_like(A_N)
        Abar_GuassianConv = np.zeros_like(Abar)
        sigma = 10 # standard deviation for Gaussian kernel
        for popi in range(pop_num):
            A_N_GuassianConv[:, popi] = gaussian_filter1d(A_N[:, popi], sigma)
            Abar_GuassianConv[:, popi] = gaussian_filter1d(Abar[:, popi], sigma)
        # plot processed data
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, A_N_GuassianConv * 1000)  # plot population activities (in Hz)
        plt.ylabel(r"$A_N$ [Hz]")
        plt.title("Population activities (Guassian filter, mesoscopic model)")
        plt.subplot(2, 1, 2)
        plt.plot(t, Abar_GuassianConv * 1000)  # plot instantaneous population rates (in Hz)
        plt.ylabel(r"$\bar A$ [Hz]")
        plt.xlabel("time [ms]")
        plt.savefig(f"Population activities (Guassian filter, mesoscopic model)_{time_string}.pdf", format="pdf")
        plt.show()

        # data processing method 1: moving average smooth
        pop_num = A_N.shape[1]
        timePoint_num = A_N.shape[0]
        averaging_window = 100 # moving average window
        # given dt_rec = 1 ms, averaging_window = 100 means "low-pass-filtered by a moving average of 100 ms" 
        A_N_MovingAve = np.zeros((timePoint_num + averaging_window - 1, pop_num))
        Abar_MovingAve = np.zeros((timePoint_num + averaging_window - 1, pop_num))
        for popi in range(pop_num):
            A_N_MovingAve[:, popi] = np.pad(A_N[:, popi], (averaging_window-1, 0), 'symmetric')
            Abar_MovingAve[:, popi] = np.pad(Abar[:, popi], (averaging_window-1, 0), 'symmetric')
            # mode 'symmetric' in np.pad is same as mode 'reflect' (default mode) in scipy.ndimage.gaussian_filter1d
            # reflect mode means (d c b a | a b c d | d c b a).
            # The "mode" parameter determines how the input array is extended beyond its boundaries.
        # plot processed data
        for popi in range(pop_num):
            for t_ in range(timePoint_num):
                t_ = - (t_ + 1)
                A_N_MovingAve[t_, popi] = np.mean(A_N_MovingAve[t_-averaging_window:t_,popi])
                Abar_MovingAve[t_, popi] = np.mean(Abar_MovingAve[t_-averaging_window:t_,popi])
        A_N_MovingAve = A_N_MovingAve[averaging_window-1:, :]
        Abar_MovingAve = Abar_MovingAve[averaging_window-1:, :]
        # plot processed data
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, A_N_MovingAve * 1000)  # plot population activities (in Hz)
        plt.ylabel(r"$A_N$ [Hz]")
        plt.title("Population activities (moving average, mesoscopic model)")
        plt.subplot(2, 1, 2)
        plt.plot(t, Abar_MovingAve * 1000)  # plot instantaneous population rates (in Hz)
        plt.ylabel(r"$\bar A$ [Hz]")
        plt.xlabel("time [ms]")
        plt.savefig(f"Population activities (moving average, mesoscopic model)_{time_string}.pdf", format="pdf")
        plt.show()

    if type == "micro":
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y%m%d-%H%M%S")
        
        filename = f'data_{type}_{time_string}.json'

        if save:
            # Save the data to a JSON file
            save_neuro_activities(results, type, filename)

            # Load the data from the JSON file
            A_N, elapsed_time, t = load_neuro_activities(filename, type)

            print(f"The simulation time of this data is {elapsed_time}")
        else:
            A_N, elapsed_time, t = results

            print(f"The simulation time of this data is {elapsed_time}")

        # plot raw data
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(t, A_N * 1000)  # plot population activities (in Hz)
        plt.ylabel(r"$A_N$ [Hz]")
        plt.xlabel("time [ms]")
        plt.title("Population activities (raw, microscopic model)")
        plt.savefig(f"Population activities (raw, microscopic model)_{time_string}.pdf", format="pdf")
        plt.show()

        # data processing method 1: Guassian filter -- using Guassian kernel convolution for smoothing
        pop_num = A_N.shape[1]
        A_N_GuassianConv = np.zeros_like(A_N)
        sigma = 10 # standard deviation for Gaussian kernel
        for popi in range(pop_num):
            A_N_GuassianConv[:, popi] = gaussian_filter1d(A_N[:, popi], sigma)
        # plot processed data
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(t, A_N_GuassianConv * 1000)  # plot population activities (in Hz)
        plt.ylabel(r"$A_N$ [Hz]")
        plt.xlabel("time [ms]")
        plt.title("Population activities (Guassian filter, microscopic model)")
        plt.savefig(f"Population activities (Guassian filter, microscopic model)_{time_string}.pdf", format="pdf")
        plt.show()

        # data processing method 1: moving average smooth
        pop_num = A_N.shape[1]
        timePoint_num = A_N.shape[0]
        averaging_window = 100 # moving average window
        # given dt_rec = 1 ms, averaging_window = 100 means "low-pass-filtered by a moving average of 100 ms" 
        A_N_MovingAve = np.zeros((timePoint_num + averaging_window - 1, pop_num))
        for popi in range(pop_num):
            A_N_MovingAve[:, popi] = np.pad(A_N[:, popi], (averaging_window-1, 0), 'symmetric')
            # mode 'symmetric' in np.pad is same as mode 'reflect' (default mode) in scipy.ndimage.gaussian_filter1d
            # reflect mode means (d c b a | a b c d | d c b a).
            # The "mode" parameter determines how the input array is extended beyond its boundaries.
        # plot processed data
        for popi in range(pop_num):
            for t_ in range(timePoint_num):
                t_ = - (t_ + 1)
                A_N_MovingAve[t_, popi] = np.mean(A_N_MovingAve[t_-averaging_window:t_,popi])
        A_N_MovingAve = A_N_MovingAve[averaging_window-1:, :]
        # plot processed data
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(t, A_N_MovingAve * 1000)  # plot population activities (in Hz)
        plt.ylabel(r"$A_N$ [Hz]")
        plt.title("Population activities (moving average, microscopic model)")
        plt.xlabel("time [ms]")
        plt.savefig(f"Population activities (moving average, microscopic model)_{time_string}.pdf", format="pdf")
        plt.show()
