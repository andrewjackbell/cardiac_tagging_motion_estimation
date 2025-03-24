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


# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    w_dict = torch.load(weights)
    model.load_state_dict(w_dict, strict=True)
    return model

def test_Cardiac_Tagging_ME_net(net, \
                                data_dir, \
                                model_path, \
                                dst_root, \
                                case = 'proposed'):

    dataset_name = 'dataset_test_100.npz'

    test_images = np.load(os.path.join(data_dir,dataset_name))['images']
    test_points = np.load(os.path.join(data_dir,dataset_name))['points']

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

    for i, data in enumerate(test_set_loader):
        # cine0, tag = dataB
        tagged_cine = data
        points_tensor = test_points[i].to(device)

        # wrap input data in a Variable object
        cine0 = tagged_cine.to(device)

        #cine1 = tagged_cine[:, 2:, ::]  # no grid frame

        # wrap input data in a Variable object
        img = cine0.cuda()
        img = img.float()

        x = img[:, 1:, ::]  # other frames except the 1st frame
        y = img[:, 0:19, ::]  # 1st frame also is the reference frame
        shape = x.shape  # batch_size, seq_length, height, width
        seq_length = shape[1]
        height = shape[2]
        width = shape[3]
        x = x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # y = y.repeat(1, seq_length, 1, 1)  # repeat the ES frame to match other frames contained in a Cine
        y = y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        z = cine0[:, 0:1, ::]  # Tag grid frame also is the reference frame
        z = z.repeat(1, seq_length, 1, 1)  # repeat the ES frame to match other frames contained in a Cine
        z = z.contiguous()
        z = z.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # forward pass
        with torch.no_grad():
            val_registered_cine1, val_registered_cine2, val_registered_cine_lag, val_flow_param, \
            val_deformation_matrix, val_deformation_matrix_neg, val_deformation_matrix_lag = net(y, x)

        # warp the points using the lagrangian flow

        flow = val_deformation_matrix_lag
        es_flow = flow[5]

        ed_points = points_tensor[0]
        ed_points_pixel = (ed_points +1) * 128/2
        es_gt_points = points_tensor[5]
        es_gt_points_pixel = (es_gt_points +1) * 128/2

    
        grid = ed_points # shape [N, 2]
        grid = grid.view(1, -1, 1, 2)  # Reshape for grid_sample: [1, N, 1, 2]

        sampled_flow_pixel = grid_sample(
            es_flow.unsqueeze(0),  # [1, 2, H, W]
            grid,
            mode='bilinear',
            padding_mode='border',  
            align_corners=True
        ).squeeze().T  # [N, 2]

        es_pred_points_pixel = ed_points_pixel + sampled_flow_pixel

        rmse = torch.sqrt(torch.mean((es_gt_points_pixel - es_pred_points_pixel) ** 2))
        errors.append(rmse.item())


    print(f"Mean RMSE: {np.mean(errors)}") 
    print(f"Max RMSE: {np.max(errors)}")
    print(f"Min RMSE: {np.min(errors)}")

    # on the last iteration, convert and plot
    ed_frame = tagged_cine[0, 0, :, :].cpu().numpy()
    es_frame = tagged_cine[0, 5, :, :].cpu().numpy()
    es_pred_points_pixel = es_pred_points_pixel.cpu().numpy()
    ed_points_pixel = ed_points_pixel.cpu().numpy()
    es_gt_points_pixel = es_gt_points_pixel.cpu().numpy()


    plt.subplot(2, 2, 1)
    plt.imshow(ed_frame, cmap='gray')
    plt.scatter(ed_points_pixel[:, 0], ed_points_pixel[:, 1], c='r', s=5)
    plt.title("End-diastole")
    plt.subplot(2, 2, 3)
    plt.imshow(es_frame, cmap='gray')
    plt.scatter(es_gt_points_pixel[:, 0], es_gt_points_pixel[:, 1], c='r', s=5)
    plt.title("End-systole GT")
    plt.subplot(2, 2, 4)
    plt.imshow(es_frame, cmap='gray')
    plt.scatter(es_pred_points_pixel[:, 0], es_pred_points_pixel[:, 1], c='r', s=5)
    plt.title("End-systole Pred")
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










