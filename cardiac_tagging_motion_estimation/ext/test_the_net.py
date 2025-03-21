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

    for i, data in enumerate(test_set_loader):
        # cine0, tag = dataB
        tagged_cine = data
        points_tensor = test_points[i]

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

        # visualise the results

        """print("Shapes:")
        print("val_registered_cine1: ", val_registered_cine1.shape) # [frames, 1, H, W]
        print("val_registered_cine2: ", val_registered_cine2.shape) # [frames, 1, H, W]
        print("val_registered_cine_lag: ", val_registered_cine_lag.shape) # [frames, 1, H, W]
        print("val_flow_param: ", val_flow_param.shape) # [frames, 4, H, W]
        print("val_deformation_matrix: ", val_deformation_matrix.shape) # [frames, 2, H, W]
        print("val_deformation_matrix_neg: ", val_deformation_matrix_neg.shape) # [frames, 2, H, W]
        print("val_deformation_matrix_lag: ", val_deformation_matrix_lag.shape) # [frames, 2, H, W]"""




        # warp the first frame by the 6th frame lagrangian deformation field to yield predicted 6th frame

        from torch.nn.functional import grid_sample

        # first frame points
        first_frame_points = points_tensor[0]
        print(first_frame_points.shape) # 168,2

        first_frame_points = first_frame_points.unsqueeze(0) # add batch dim
        first_frame_points = first_frame_points.unsqueeze(2) # add fake width dim

        print(first_frame_points.shape) # 1,168,1,2 (THIS WILL BE THE GRID)
        
        # lagrangian deformation field at the 6th frame (ED)

        deformation_field = val_deformation_matrix_neg[0]
        print(deformation_field.shape) # 2,128,128 (THIS WILL BE THE INPUT)
        deformation_field = deformation_field.unsqueeze(0) # add batch dim # 1,2,128,128

        # normalize the deformation field to [-1, 1]
        deformation_field = deformation_field / 128.0
        deformation_field = deformation_field *2 - 1

        output = grid_sample(deformation_field.cpu(), first_frame_points.cpu(), mode='bilinear', padding_mode='zeros')
        print(output.shape) # 1,2,168,1

        output = output.squeeze(0) # remove batch dim
        output = output.squeeze(2) # remove fake width dim

        print(output.shape) # 168,2

      
        # plot the points
        plt.scatter(output[:,0], output[:,1], c='r', s=10)
        plt.show()

        """# warp the points by the estimated deformation field 

        points_warped = torch.zeros_like(points_tensor)
        
        points_imspace = ((points_tensor + 1) / 2) * 128
        points_imspace = points_imspace.int()

        reference_frame = 0
        for j in range(val_deformation_matrix_lag.shape[0]):
            for k in range(points_imspace.shape[1]):
                ref_x = points_imspace[reference_frame, k, 0]
                ref_y = points_imspace[reference_frame, k, 1]

                def_x = val_deformation_matrix_lag[j, 0, ref_y, ref_x]
                def_y = val_deformation_matrix_lag[j, 1, ref_y, ref_x]

                points_warped[j, k, 0] = ref_x + def_x
                points_warped[j, k, 1] = ref_y + def_y


        # show the warped points on the cine, alongside the ground truth points

        for j in range(points_warped.shape[0]): # j frames
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(tagged_cine[0, j, :, :].cpu().numpy(), cmap='gray')
            ax.scatter(points_imspace[j, :, 0], points_imspace[j, :, 1], c='r', s=10, label='GT')
            ax.scatter(points_warped[j, :, 0], points_warped[j, :, 1], c='b', s=10, label='Warped')
            ax.legend()


            plt.show()"""


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










