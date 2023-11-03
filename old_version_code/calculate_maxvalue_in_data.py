import os
import json
import numpy as np

def calculate_max_values_pop3(filenames):
    max_values_A_N = np.zeros(3)  # Initialize to a zero array of size 8
    max_values_labels = np.array([0, 0, 0, 0, 0, 0])
    
    for filename in filenames:
        with open(filename, 'r') as json_file:
            data_dict = json.loads(json_file.read())

        A_N = np.array(data_dict["A_N"])
        A_N_reshaped = A_N.reshape(-1, 120, 3)
        A_N_avg = A_N_reshaped.mean(axis=1)

        max_values_A_N = np.maximum(max_values_A_N, np.max(np.abs(A_N_avg), axis=0))
        
        labels_raw = (
            np.array(data_dict["J_syn"]),
            np.array(data_dict["mu"]),
            np.array(data_dict["tau_m"]),
            np.array(data_dict["V_th"]),
            np.array(data_dict["J_theta"]),
            np.array(data_dict["tau_theta"])
        )
        
        max_values_labels = np.maximum(max_values_labels, np.array([np.max(np.abs(l)) for l in labels_raw]))
    
    return max_values_A_N, max_values_labels

def calculate_max_values_pop8(filenames):
    max_values_A_N = np.zeros(8)  # Initialize to a zero array of size 8
    max_values_labels = np.array([0, 0, 0, 0, 0, 0])
    
    for filename in filenames:
        with open(filename, 'r') as json_file:
            data_dict = json.loads(json_file.read())

        A_N = np.array(data_dict["A_N"])
        A_N_reshaped = A_N.reshape(-1, 40, 8)
        A_N_avg = A_N_reshaped.mean(axis=1)

        max_values_A_N = np.maximum(max_values_A_N, np.max(np.abs(A_N_avg), axis=0))
        
        labels_raw = (
            np.array(data_dict["J_syn"]),
            np.array(data_dict["mu"]),
            np.array(data_dict["tau_m"]),
            np.array(data_dict["V_th"]),
            np.array(data_dict["J_theta"]),
            np.array(data_dict["tau_theta"])
        )
        
        max_values_labels = np.maximum(max_values_labels, np.array([np.max(np.abs(l)) for l in labels_raw]))
    
    return max_values_A_N, max_values_labels

# pop3
# Assume your json files are in the "data_folder"
data_folder = 'pop3_data_with_adapt'
filenames = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder) if fn.endswith('.json')]
filenames = filenames[:100]

# Calculate the max values using given filenames
max_values_A_N, max_values_labels = calculate_max_values_pop3(filenames)
print(max_values_A_N, max_values_labels)

# # pop8
# # Assume your json files are in the "data_folder"
# data_folder = 'pop8_data_with_adapt'
# filenames = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder) if fn.endswith('.json')]
# filenames = filenames[:100]

# # Calculate the max values using given filenames
# max_values_A_N, max_values_labels = calculate_max_values_pop8(filenames)
# print(max_values_A_N, max_values_labels)