import os
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from Models_class_pop8 import MultiTaskNet_T, MultiTaskNet_S, MultiTaskNet_M, MultiTaskNet_L, MultiTaskNet_X

# Assume your json files are in the "data_folder"
data_folder = 'pop8_data_with_adapt'
filenames = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder) if fn.endswith('.json')]
filenames = random.sample(filenames, 5000) # 4000

# Split your data into train and validation sets
train_filenames, val_filenames = train_test_split(filenames, test_size=0.02, random_state=42)

class NeuroDataset(Dataset):
    def __init__(self, filenames, max_values_A_N, max_values_labels):
        self.filenames = filenames
        self.max_values_A_N = max_values_A_N
        self.max_values_labels = max_values_labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with open(self.filenames[idx], 'r') as json_file:
            data_dict = json.loads(json_file.read())

        A_N = np.array(data_dict["A_N"])
        A_N_reshaped = A_N.reshape(-1, 40, 8)  # Reshape to (500, 40, 8), original (20000, 8)
        A_N_avg = A_N_reshaped.mean(axis=1)  # Average over the second dimension, resulting in a (500, 8) array

        # Normalize A_N_avg to [-1, 1] for each channel
        A_N_avg = A_N_avg / self.max_values_A_N[np.newaxis, :] if np.any(self.max_values_A_N > 0) else A_N_avg
        
        # Convert A_N_avg to tensor
        A_N_avg = torch.from_numpy(A_N_avg).float()
        
        labels_raw = (
            np.array(data_dict["J_syn"]),
            np.array(data_dict["mu"]),
            np.array(data_dict["tau_m"]),
            np.array(data_dict["V_th"]),
            np.array(data_dict["J_theta"]),
            np.array(data_dict["tau_theta"])
        )

        # Normalize labels to [-1, 1]
        labels = [torch.from_numpy(l / max_val) if max_val > 0 else torch.from_numpy(l) 
                    for l, max_val in zip(labels_raw, self.max_values_labels)]

        return A_N_avg, labels

# Give the max values as reference (which are not necessarily max values in data)
max_values_A_N = np.array([0.40100324, 0.42095903, 0.40015286, 0.40059317, 
                           0.40120619, 0.40443662, 0.3810976, 0.40001696]) 
max_values_labels = np.array([1.98439324, 59.93106451, 39.98236293, 
                              29.99720433, 1477.90163064, 1494.3190043])

# Pass these values when creating your datasets
train_dataloader = DataLoader(NeuroDataset(train_filenames, max_values_A_N, max_values_labels), 
                              batch_size=32, shuffle=True)
val_dataloader = DataLoader(NeuroDataset(val_filenames, max_values_A_N, max_values_labels), 
                            batch_size=32, shuffle=False)

# # Create data loaders
# train_dataloader = DataLoader(NeuroDataset(train_filenames), batch_size=32, shuffle=True)
# val_dataloader = DataLoader(NeuroDataset(val_filenames), batch_size=32, shuffle=False)

# Initialize your model and optimizer
model = MultiTaskNet_X()
optimizer = Adam(model.parameters(), lr=0.004) # , lr=0.003
scheduler = StepLR(optimizer, step_size=20, gamma=0.4)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device)

best_val_loss = float('inf')
train_loss_list = []
val_loss_list = []

# Training loop
for epoch in range(100):  # 100 epochs, adjust as needed
    model.train()
    train_loss = 0.0
    for A_N, labels in train_dataloader:
        # A_N is a tensor whose first dimension is batchsize;
        # while labels is a list of length 6 (6 is the task number)
        
        # Convert numpy arrays to PyTorch tensors and move them to the correct device
        A_N = A_N.float().to(device)
        labels = [l.float().to(device) for l in labels]

        # Forward pass
        outputs = model(A_N)
        # outputs is a tuple of length 6 (6 is the task number)

        # Compute loss
        loss = sum(F.mse_loss(output, label) for output, label in zip(outputs, labels))
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch+1}, Training Loss: {train_loss}")
    train_loss_list.append(train_loss)
    
    scheduler.step()

    # Evaluation on the validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for A_N, labels in val_dataloader:
            A_N = A_N.float().to(device)
            labels = [l.float().to(device) for l in labels]
            outputs = model(A_N)
            val_loss += sum(F.mse_loss(output, label) for output, label in zip(outputs, labels)).item()
    
    val_loss /= len(val_dataloader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    val_loss_list.append(val_loss)

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        with open("./trained_models/model_pop8_best_X.pth", "wb") as f:
            torch.save(model.state_dict(), f)
        best_val_loss = val_loss

# Save the loss lists into a dictionary
loss_dict = {'train_loss': train_loss_list, 'val_loss': val_loss_list}

# Write the dictionary to a json file
with open('./trained_models/pop8_X_loss_values.json', 'w') as f:
    json.dump(loss_dict, f)

