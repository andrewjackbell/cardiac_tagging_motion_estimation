import sys
import os, json
import numpy as np
import torch

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Parent of current directory

# Add root to path
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.nn.functional import grid_sample

from data.preprocess_my import preprocess_tensors

import torch
from ME_nets.LagrangianMotionEstimationNet import Lagrangian_motion_estimate_net
from ME_nets.LagrangianMotionEstimationNet import SpatialTransformer
from data_set.load_data_for_cine_ME import add_np_data, get_np_data_as_groupids,load_np_datagroups, DataType, \
    load_Dataset
import SimpleITK as sitk
from ext.warping import warp_image, warp_points, normalise_coords, denormalise_coords

# seeding
torch.manual_seed(0)
np.random.seed(0)



# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    w_dict = torch.load(weights)
    model.load_state_dict(w_dict, strict=True)
    return model


def create_synced_animations(fig, axs, data_list, fps=5):
    """
    data_list: List of tuples (points_list, points_shape, colors, background_frames)
    for each subplot
    """
    n_frames = data_list[0][3].shape[0]  # Get frames from first background
    interval = 1000 // fps
    
    # Store all animation elements
    all_bg_images = []
    all_lines_by_set = []
    
    # Initialize each subplot
    for ax, (points_list, points_shape, colors, background_frames) in zip(axs, data_list):
        u_samples, v_samples, _ = points_shape
        
        # Pre-calculate indices
        h_indices = [(j, j+1) for j in range(points_list[0].shape[1] - 1) 
                     if (j + 1) % v_samples != 0]
        v_indices = [(j, j+v_samples) for j in range(points_list[0].shape[1] - v_samples) 
                     if (j % v_samples) < 10]
        
        # Setup background
        bg_image = ax.imshow(background_frames[0], cmap='gray', zorder=0)
        all_bg_images.append(bg_image)
        
        # Setup lines
        subplot_lines = []
        for idx, _ in enumerate(points_list):
            h_lines = [ax.plot([], [], '-', color=colors[idx], zorder=1)[0] 
                      for _ in range(len(h_indices))]
            v_lines = [ax.plot([], [], '-', color=colors[idx], zorder=1)[0] 
                      for _ in range(len(v_indices))]
            subplot_lines.append((h_lines, v_lines, h_indices, v_indices))
        all_lines_by_set.append((points_list, subplot_lines))
    
    def update(frame):
        all_artists = []
        
        # Update each subplot
        for idx, ((points_list, subplot_lines), bg_image) in enumerate(zip(all_lines_by_set, all_bg_images)):
            # Update background
            bg_image.set_array(data_list[idx][3][frame])
            all_artists.append(bg_image)
            
            # Update lines
            for points_tensor, (h_lines, v_lines, h_indices, v_indices) in zip(points_list, subplot_lines):
                points = points_tensor[frame]
                
                for line, (i, j) in zip(h_lines, h_indices):
                    line.set_data([points[i, 0], points[j, 0]], 
                                [points[i, 1], points[j, 1]])
                    all_artists.append(line)
                
                for line, (i, j) in zip(v_lines, v_indices):
                    line.set_data([points[i, 0], points[j, 0]], 
                                [points[i, 1], points[j, 1]])
                    all_artists.append(line)
        
        return all_artists
    
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                interval=interval, blit=True)
    return ani

def test_Cardiac_Tagging_ME_net(net, \
                                data_dir, \
                                model_path, \
                                dst_root, \
                                case = 'proposed'):

    dataset_name = 'dataset_test_100.npz'

    dataset = np.load(os.path.join(data_dir,dataset_name))

    test_images = dataset['images']
    test_points = dataset['points']
    cases = dataset['case']
    slices = dataset['slice']
    es_frame_numbers = dataset['es_frame']

    test_images = torch.tensor(test_images)
    test_points = torch.tensor(test_points)

    test_images, test_points = preprocess_tensors(test_images, test_points, crop=True)

    print("Test images shape: ", test_images.shape)
    print(f"Max: {test_images.max()}, Min: {test_images.min()}")
    print("type: ", test_images.dtype)


    print("-"*10)
    print("Test points shape: ", test_points.shape)
    print(f"Max: {test_points.max()}, Min: {test_points.min()}")
    print("type: ", test_points.dtype)



    val_dataset = load_Dataset(test_images)
    val_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    
    model = 'model_newish_40.pth'

    ME_model = load_dec_weights(net, model_path + model)
    ME_model = ME_model.to(device)
    ME_model.eval()

    errors = []
    es_errors = []
    predictions = []

    for i, data in enumerate(test_set_loader):
        # cine0, tag = data

        tagged_cine = data.to(device).float()
        points_tensor = test_points[i].unsqueeze(0).to(device)
        tagged_cine_short = tagged_cine[:, :, ::]  # remove the first two frames
        points_tensor_short = points_tensor[:, :, ::]  # remove the first two frames
        es_frame_n = es_frame_numbers[i]
        es_frame_index = es_frame_n - 1 
        es_flow_index = es_frame_n - 2

        x = tagged_cine[:, 1:, ::]  # other frames except the 1st frame
        y = tagged_cine[:, :19, ::]  # 1st frame also is the reference frame
        shape = x.shape  # batch_size, seq_length, height, width
        seq_length = shape[1]
        height = shape[2]
        width = shape[3]
        x = x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
        y = y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # forward pass
        with torch.no_grad():
            val_registered_cine1, val_registered_cine2, val_registered_cine_lag, val_flow_param, \
            val_deformation_matrix, val_deformation_matrix_neg, val_deformation_matrix_lag = net(y, x)

        ed_points_repeat = points_tensor_short[:, 0, ::].repeat(seq_length, 1, 1)
        warped_points_all = warp_points(ed_points_repeat, val_deformation_matrix_lag, device=device)
        warped_points_es = warped_points_all[es_flow_index].cpu()

        predictions.append(warped_points_all)
        

        points_short_denorm = denormalise_coords(points_tensor_short, width, range="-1-1").cpu()
        ed_gt = points_short_denorm[0, 0, ::]
        es_gt = points_short_denorm[0, es_frame_index, ::].numpy()

        #rmse = torch.sqrt(torch.mean((warped_points_es - es_gt) ** 2))
        rmse = torch.sqrt(torch.mean((warped_points_es - es_gt) ** 2))
        
        es_errors.append(rmse)


    def prepare_case_data(case_idx, val_dataset, predictions, test_points, width):
        """Helper function to prepare data for a single case"""
        cine = val_dataset[case_idx][1:]
        predicted_points = predictions[case_idx]
        gt_points = test_points[case_idx][1:]
        gt_points_denorm = denormalise_coords(gt_points, width, range="-1-1").cpu()
        
        return (
            [gt_points_denorm.squeeze().to('cpu'), predicted_points.cpu()],  # points_list
            (7, 7, 17),  # points_shape
            ['green', 'blue'],  # colors
            cine.squeeze().cpu()  # background_frames
        )

    # Calculate statistics
    mean_errors = np.mean(es_errors)
    sdev_errors = np.std(es_errors)
    print(f"Mean error: {mean_errors}")
    print(f"Standard deviation of errors: {sdev_errors}")

    # Select cases
    cases = {
        'Worst': np.argmax(es_errors),
        'Best': np.argmin(es_errors),
        'Average': np.argsort(es_errors)[len(es_errors)//2],
        'Random': np.random.randint(len(es_errors))
    }

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    # Prepare data for all cases
    data_list = [
        prepare_case_data(case_idx, val_dataset, predictions, test_points, width)
        for case_idx in cases.values()
    ]

    # Create synced animations
    ani = create_synced_animations(fig, axs, data_list, fps=30)

    # Set titles
    for ax, (case_name, case_idx) in zip(axs, cases.items()):
        ax.set_title(f'{case_name} Case, green=GT, blue=Predicted\nRMSE: {es_errors[case_idx]:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    data_dir = './cardiac_tagging_motion_estimation/data/'
 
    # proposed model
    vol_size = (128, 128)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    test_model_path = './cardiac_tagging_motion_estimation/model/'
    dst_root = './cardiac_tagging_motion_estimation/results/'
    if not os.path.exists(dst_root): os.mkdir(dst_root)
    test_Cardiac_Tagging_ME_net(net=net,
                             data_dir=data_dir,
                             model_path= test_model_path,
                             dst_root=dst_root,
                             case = 'proposed')










