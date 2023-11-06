import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiplicativeLR

from Models_class_pop3 import MultiTaskNet_C, MultiTaskNet_L
from Meso_CollectData_func_pop3 import mesoscopic
from utils import *

import pdb


class NeuroDataset(Dataset):
    def __init__(self, inputs, outputs, 
        inp_m=None, inp_s=None, out_m=None, out_s=None):
        self.inputs = inputs
        self.outputs = outputs
        if inp_m is None:
            self.inputs_mean = np.mean(inputs, 0)
        else:
            self.inputs_mean = inp_m
        if inp_s is None:
            self.inputs_std= np.std(inputs, 0)
            self.inputs_std[self.inputs_std==0] = 1
        else:
            self.inputs_std = inp_s
        if out_m is None:
            self.outputs_mean = np.mean(outputs, 0)
        else:
            self.outputs_mean = out_m
        if out_s is None:
            self.outputs_std= np.std(outputs, 0)
            self.outputs_std[self.outputs_std==0] = 1
        else:
            self.outputs_std = out_s

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = torch.from_numpy((self.inputs[idx]-self.inputs_mean)/self.inputs_std).float()
        labels = torch.from_numpy((self.outputs[idx]-self.outputs_mean)/self.outputs_std).float()

        return inputs, labels

def train_3dmodel(exp_params, lr=3e-4, epochs=500, batch_size=1024):
    _, data_folder, trained_model_folder, folder_path, prefix = exp_params.return_folders()
    args = exp_params.parse_args()

    GTdata_folder = f"{folder_path}/{experiment_params.return_prefix(True, 1.0, 1.0, 0)}_post_processed"

    all_data = np.load(f"{data_folder}/frs.npz")
    all_inputs = np.concatenate((all_data['mean_frs'],all_data['std_frs']), axis=1)
    all_outputs = all_data['params']

    GT_data = np.load(f"{GTdata_folder}/frs.npz")
    GT_inputs = np.concatenate((GT_data['mean_frs'],GT_data['std_frs']), axis=1)
    GT_outputs = GT_data['params']

    # Split your data into train and validation sets
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(all_inputs, all_outputs, test_size=0.02, random_state=42)
    print(f"In total: {len(train_inputs)} training files; {len(val_inputs)} val files")

    # Pass these values when creating your datasets
    train_dataloader = DataLoader(NeuroDataset(train_inputs, train_outputs), 
                                  batch_size=batch_size, shuffle=True)
    inp_m, inp_s, out_m, out_s = train_dataloader.dataset.inputs_mean, train_dataloader.dataset.inputs_std, train_dataloader.dataset.outputs_mean, train_dataloader.dataset.outputs_std
    val_dataloader = DataLoader(NeuroDataset(val_inputs, val_outputs,
                                    inp_m=inp_m, inp_s=inp_s, out_m=out_m, out_s=out_s), 
                                batch_size=batch_size, shuffle=False)
    GT_dataloader = DataLoader(NeuroDataset(GT_inputs, GT_outputs,
                                    inp_m=inp_m, inp_s=inp_s, out_m=out_m, out_s=out_s), 
                                batch_size=50, shuffle=False)
        
    pops, pops_prop, pconn, tau_sfa_exc, tau_sfa_inh, J_sfa_exc, J_sfa_inh, tau_theta, J_theta = get_gt_3dotherparams() 


    # Initialize your model and optimizer
    # model = MultiTaskNet_C()
    model = MultiTaskNet_L()
    optimizer = Adam(model.parameters(), lr)
    # scheduler = StepLR(optimizer, step_size=10, gamma=np.sqrt(0.1))
    # lmbda = lambda epoch: 0.98
    # scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = model.to(device)

    best_val_loss = float('inf')
    train_losses = []; val_losses = []; GT_losses = []
    # Training loop
    for epoch in range(epochs):  # 100 epochs, adjust as needed
        model.train()
        train_loss = 0.0
        for A_N, labels in train_dataloader:
            # A_N is a tensor whose first dimension is batchsize;
            # while labels is a list of length 6 (6 is the task number)
            
            # Convert numpy arrays to PyTorch tensors and move them to the correct device
            A_N = A_N.float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(A_N)
            # outputs is a tuple of length 6 (6 is the task number)

            # Compute loss
            loss = F.mse_loss(outputs, labels)
            # loss = sum(F.mse_loss(output, label) for output, label in zip(outputs, labels))
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_dataloader)
        train_losses.append((epoch+1, train_loss))
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}")
        
        # scheduler.step()

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for A_N, labels in val_dataloader:
                A_N = A_N.float().to(device)
                labels = labels.float().to(device)
                outputs = model(A_N)
                val_loss += F.mse_loss(outputs, labels).item()
        
        val_loss /= len(val_dataloader)
        val_losses.append((epoch+1, val_loss))
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            with open(f"{trained_model_folder}/model_pop3_lr{lr}.pth", "wb") as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss

            GT_loss = 0.0
            for A_N, labels in GT_dataloader:
                A_N = A_N.float().to(device)
                labels = labels.float().to(device)
                outputs = model(A_N)
                GT_loss += F.mse_loss(outputs, labels).item()
        
            GT_loss /= len(GT_dataloader)
            GT_losses.append((epoch+1, GT_loss))
            print(f"Epoch {epoch+1}, GT Loss: {GT_loss}")
            params = outputs.mean(0).cpu().detach().numpy() * out_s + out_m
            gt_params = labels.mean(0).cpu().detach().numpy() * out_s + out_m
            print(f"fit: {np.around(params, decimals=1)}")
            print(f"GT: {np.around(gt_params, decimals=1)}")

            np.save(f"{trained_model_folder}/model_pop3_fitParams_lr{lr}_{epoch+1}.npy", params)


            plot_losses(train_losses, val_losses, GT_losses, 
                        f"{trained_model_folder}/loss_curve_lr{lr}.png")   
            plt.close()

            J_syn = np.outer(np.ones_like(pops_prop), np.where(pops_prop == -1, params[1], params[0]))
            J_syn = J_syn * pconn # * np.random.uniform(0.5, 1.5, (len(pops), len(pops)))
            A_N, Abar, elapsed_time, t = mesoscopic(pops=pops, 
                                  pops_prop=pops_prop, 
                                  connect_mat=J_syn, 
                                  mu=params[2]*np.ones(len(pops)), 
                                  tau_m=params[3]*np.ones(len(pops)), 
                                  V_th=params[4]*np.ones(len(pops)), 
                                  J_theta=J_theta, 
                                  tau_theta=tau_theta,
                                  pconn=pconn,
                                  adapt=True,
                                  seed=1,
                                  t_end=30000)

            plt.figure()
            for popi in [0,2]:
                plt.plot(t, gaussian_filter1d(Abar[:,popi],10) * 1000)  # plot instantaneous population rates (in Hz)
            plt.ylabel(r'$\bar A$ [Hz]')
            plt.xlabel('time [ms]')
            plt.title('Population activities (mesoscopic sim.)')
            plt.savefig(f"{trained_model_folder}/model_pop3_lr{lr}_{epoch+1}.png")
            # plt.show()
        
if __name__ == '__main__':
    exp_params = experiment_params()
    train_3dmodel(exp_params)

