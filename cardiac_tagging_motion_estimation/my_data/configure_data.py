
import torch
import os
import numpy as np
import json

data_dir = './cardiac_tagging_motion_estimation/my_data'

dataset_name = 'dataset_train_100'

# Load the data
data = torch.load(os.path.join(data_dir, dataset_name + '.pt'))

# convert to simple np array

images = np.zeros((len(data), 20, 256, 256))

for i,case in enumerate(data):
    images[i] = np.array(case['images'])

print(images.shape)




output_dir = './cardiac_tagging_motion_estimation/data_new'
os.makedirs(output_dir, exist_ok=True)

np.savez_compressed(os.path.join(output_dir, dataset_name + '.npz'), images)

# load it again to check

data = np.load(os.path.join(output_dir, dataset_name + '.npz'))



"""
output_dir = os.path.join(data_dir, 'npz_files')
os.makedirs(output_dir, exist_ok=True)

data = images

# Save each case as a separate .npz file
for i in range(data.shape[0]):
    case_data = data[i]  # Shape [20, 256, 256]
    npz_filename = os.path.join(output_dir, f'validation_cine_{i:05d}.npz')
    np.savez_compressed(npz_filename, case_data)

# Create a dictionary to map each case to its .npz file
data_config = {
    "validation": {
        "group_0": {
            f"case_{i:05d}": {
                "cine": [os.path.join(output_dir, f'validation_cine_{i:05d}.npz')],
                "tag": [os.path.join(output_dir, f'validation_cine_{i:05d}.npz')]
            } for i in range(data.shape[0])  
        }
    }
}

# Save the configuration to a JSON file
config_file = os.path.join(output_dir, 'Cardiac_ME_val_config.json')
with open(config_file, 'w') as f:
    json.dump(data_config, f, indent=4)"""