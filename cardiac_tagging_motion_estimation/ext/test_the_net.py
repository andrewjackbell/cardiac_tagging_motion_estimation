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

def plotting_function_grid(fig, ax, points_list, points_shape, colors, background_frames=None):
    '''
    Takes a single axis and plots an animation on it. 
    The animation consists of a scatter plot of the points moving over time, with an optional background movie (sequence of images).

    Parameters:
    - fig: matplotlib figure object.
    - ax: matplotlib axis object.
    - points_list: List of torch tensors of shape (n_frames, n_points, 2) where the last dimension is the x and y coordinates of the points.
    - points_shape: Tuple (u_samples, v_samples, _) representing the shape of the points grid.
    - colors: List of colors (e.g., ['orange', 'blue', 'green']) for each set of points in points_list.
    - background_frames: torch tensor of shape (n_frames, H, W) where H and W are the height and width of the images.

    Returns the animation object, which needs to be preserved to prevent garbage collection.
    '''
    ims = []
    n_frames = points_list[0].shape[0]
    u_samples, v_samples, _ = points_shape

    for i in range(n_frames):
        all_lines = []
        
        for idx, points_tensor in enumerate(points_list):
            points = points_tensor[i]
            interval = v_samples
            color = colors[idx]  # Get the color for this set of points

            # Join up adjacent points horizontally
            for j in range(points.shape[0] - 1):
                if (j + 1) % interval != 0:  # Ensure not to join the last point of one interval with the first of the next
                    line, = ax.plot(points[j:j+2, 0], points[j:j+2, 1], '-', color=color)
                    all_lines.append(line)

            # Join up adjacent points vertically
            for j in range(points.shape[0] - interval):
                if (j % interval) < 10:
                    line, = ax.plot(points[j:j+interval+1:interval, 0], points[j:j+interval+1:interval, 1], '-', color=color)
                    all_lines.append(line)

        if background_frames is not None:
            im = ax.imshow(background_frames[i], cmap='gray')
            ims.append([im] + all_lines)
        else:
            ims.append(all_lines)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=0)
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
    
    model = 'end_model.pth'

    ME_model = load_dec_weights(net, model_path + model)
    ME_model = ME_model.to(device)
    ME_model.eval()

    errors = []
    es_errors = []

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
        

        points_short_denorm = denormalise_coords(points_tensor_short, width, range="-1-1").cpu()
        ed_gt = points_short_denorm[0, 0, ::].numpy()
        es_gt = points_short_denorm[0, es_frame_index, ::].numpy()

        mse = np.mean((warped_points_es.numpy() - es_gt) ** 2)
        es_errors.append(mse)

        plt.figure()

        """plt.scatter(warped_points_es[:, 0], warped_points_es[:, 1], c='r', label='Predicted')
        plt.scatter(es_gt[:, 0], es_gt[:, 1], c='b', label='Ground Truth')
        plt.title(f'Predicted vs Ground Truth Points for case {cases[i]} slice {slices[i]}, es_frame {es_frame_n}')
        plt.legend()
        plt.show()"""




    """    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    ani1 = plotting_function_grid(fig, axs[0], [points_short_denorm.squeeze()[:-1].to('cpu'),torch.tensor(warped_points_all).cpu()], (7, 7, 17), ['green', 'blue'], background_frames=tagged_cine_short.squeeze().cpu())
    axs[0].set_title('Original Points')
    axs[0].axis('off')
    plt.show()
    plt.close(fig)"""

    mean_errors = np.mean(es_errors)
    print(f"Mean error: {mean_errors}")




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










