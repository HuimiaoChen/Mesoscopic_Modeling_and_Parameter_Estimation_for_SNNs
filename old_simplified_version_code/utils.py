import numpy as np
import matplotlib.pyplot as plt
import os, sys, pdb, json, time


import argparse
from scipy.ndimage import gaussian_filter1d
from itertools import product

class experiment_params():
    def __init__(self):
        self.parse_args()
        return     

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lb', type=float, default=0.1)
        parser.add_argument('--ub', type=float, default=2.0)
        parser.add_argument('--phase', type=int, default=1)
        parser.add_argument('--setting_0', type=int, default=10001)
        parser.add_argument('--setting_N', type=int, default=10000)
        parser.add_argument('--sample_N_perparam', type=int, default=5)
        parser.add_argument('--sim_t_end', type=int, default=20000)
        parser.add_argument('--seed_N', type=int, default=1)
        parser.add_argument('--folder_path', type=str, default='pop3_data_with_adapt_test')
        parser.add_argument('--not_adapt', action='store_true')
        parser.add_argument('--plot_image', action='store_true')
        self.args = parser.parse_args()
        return self.args

    @staticmethod
    def return_prefix(adapt, lb, ub, phase):
        # adapt = not self.args.not_adapt
        prefix = f"lb{lb}ub{ub}"+ ("" if adapt else "_noadapt") + f"_phase{phase}"
        return prefix

    def return_folders(self):
        folder_path = self.args.folder_path
        prefix = experiment_params.return_prefix(not self.args.not_adapt, self.args.lb, self.args.ub, self.args.phase)

        data_folder = f"{folder_path}/{prefix}"
        print(f"data_folder: {data_folder}")
        ### Cannot be run simultaneously
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)


        post_data_folder = f"{folder_path}/{prefix}_post_processed"
        if not os.path.exists(post_data_folder):
            os.makedirs(post_data_folder)


        trained_model_folder = f"{folder_path}/{prefix}_trained_models"
        if not os.path.exists(trained_model_folder):
            os.makedirs(trained_model_folder)

        self.raw_data_folder = data_folder
        self.data_folder = post_data_folder
        self.trained_model_folder = trained_model_folder

        return data_folder, post_data_folder, trained_model_folder, folder_path, prefix

def flat_2dlist(list2d):
    return np.array(list2d).reshape((-1)).tolist()

def find_bestGMM(X, n_components_range):
    lowest_bic = np.infty
    best_gmm = None

    # Iterate over different numbers of components to find the best GMM
    for n_components in n_components_range:
        # Fit a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        # Calculate BIC for the current GMM
        bic = gmm.bic(X)
        
        # Check if the current model has lower BIC
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    print("Best number of components:", best_gmm.n_components)
    print("BIC of the best model:", lowest_bic)

    return best_gmm, lowest_bic

def get_gt_3dparams():
    gtJ = 0.096; gtg = 0.384/ 0.096
    gtmu = 36.; gtau_m = 20; gtV_th = 15.
    gt_param = np.array([gtJ, gtg, gtmu, gtau_m, gtV_th])

    return gt_param

def get_gt_3dotherparams():
    pops = np.array([400, 200, 400])
    pops_prop = np.array([1, -1, 1]) # np.random.choice([1, -1], size=len(pops)) # # 1: excitatory, -1: inhibitory
    pconn = np.array([[1, 1, 0],
                  [1, 1, 1],
                  [0, 1, 1]]) # np.random.randint(0, 2, (len(pops), len(pops)))
    tau_sfa_exc = [1000.] # [np.random.uniform(80, 1500)] #  # threshold adaptation time constants of excitatory neurons
    tau_sfa_inh = [1000.] # [np.random.uniform(80, 1500)] #  # threshold adaptation time constants of inhibitory neurons
    J_sfa_exc = [100.] # [np.random.uniform(80, 1500)]  #  # adaptation strength: size of feedback kernel theta (= area under exponential) in mV*ms
    J_sfa_inh = [100.] #  [np.random.uniform(80, 1500)] #  # in mV*ms
    tau_theta = np.array([tau_sfa_exc if prop == 1 else tau_sfa_inh for prop in pops_prop])
    J_theta = np.array([J_sfa_exc if prop == 1 else J_sfa_inh for prop in pops_prop])

    return pops, pops_prop, pconn, tau_sfa_exc, tau_sfa_inh, J_sfa_exc, J_sfa_inh, tau_theta, J_theta


def sample_3dparams_uniform(lb, ub, N_perparam,
                            data_folder, setting_0):
    gt_param = get_gt_3dparams()
    # N_perparam = int(np.ceil(np.power(args.setting_N, 1/5)))
    params = product(*[np.linspace(param*lb, param*ub, N_perparam, endpoint=True) for param in gt_param])
    params = np.array([p for p in params])

    np.savez(f"{data_folder}/params.npz", params=params, N_perparam=N_perparam, setting_0=setting_0)

    return params


def plot_losses(training_losses, validation_losses, GT_losses, save_file):
    plt.figure()
    for i, (loss, line_style, loss_name) in enumerate(zip([training_losses, validation_losses, GT_losses], ['bo-', 'go-', 'ro-'], ['Training Loss', 'Validation Loss', 'GT Loss'])):
        plt.plot([o[0] for o in loss], [o[1] for o in loss], line_style, label=loss_name)
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_file)
    # plt.show()

def plot_spectrum(time_domain_data, save_file, sampling_rate=1000, smooth_std=50):
    power_spectrums = []
    smoothed_pss = []
    plt.figure()
    for data in time_domain_data:
        # Compute the FFT
        fft_result = np.fft.fft(data)

        # Calculate the power spectrum (square of the magnitude of FFT coefficients)
        power_spectrum = np.abs(fft_result) ** 2
        smoothed_ps = gaussian_filter1d(power_spectrum, smooth_std)
        power_spectrums.append(power_spectrum)
        smoothed_pss.append(smoothed_ps)

        freq_axis = np.fft.fftfreq(len(data), 1 / sampling_rate)

        # Plot the power spectrum
        plt.plot(freq_axis[freq_axis>0], power_spectrum[freq_axis>0])
        plt.plot(freq_axis[freq_axis>0], smoothed_ps[freq_axis>0])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.xscale('log');
    plt.yscale('log');
    plt.title('Power Spectrum')
    plt.grid()
    plt.savefig(save_file)
    # plt.show()

    return np.array(power_spectrums), np.array(smoothed_pss)











