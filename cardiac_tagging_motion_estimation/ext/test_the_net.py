import sys
import os, json
import numpy as np

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Parent of current directory

# Add root to path
sys.path.append(root_dir)

import matplotlib.pyplot as plt


import torch
from ME_nets.LagrangianMotionEstimationNet import Lagrangian_motion_estimate_net
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

    tag_set = 'dataset_test_80.npz'

    validation_set = np.load(os.path.join(data_dir,tag_set))['arr_0']

    val_dataset = load_Dataset(validation_set)
    val_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    if case == 'proposed':
        model = '/end_model.pth'
    else:
        model = '/baseline_model.pth'
    ME_model = load_dec_weights(net, model_path + model)
    ME_model = ME_model.to(device)
    ME_model.eval()

    for i, data in enumerate(test_set_loader):
        # cine0, tag = data
        tagged_cine = data

        # wrap input data in a Variable object
        cine0 = tagged_cine.to(device)

        cine1 = tagged_cine[:, 2:, ::]  # no grid frame

        # wrap input data in a Variable object
        img = cine1.cuda()
        img = img.float()

        x = img[:, 1:, ::]  # other frames except the 1st frame
        y = img[:, 0:17, ::]  # 1st frame also is the reference frame
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

        print("Shapes:")
        print("val_registered_cine1: ", val_registered_cine1.shape) # [B, 1, H, W]
        print("val_registered_cine2: ", val_registered_cine2.shape) # [B, 1, H, W]
        print("val_registered_cine_lag: ", val_registered_cine_lag.shape) # [B, 1, H, W]
        print("val_flow_param: ", val_flow_param.shape) # [B, 4, H, W]
        print("val_deformation_matrix: ", val_deformation_matrix.shape) # [B, 2, H, W]
        print("val_deformation_matrix_neg: ", val_deformation_matrix_neg.shape) # [B, 2, H, W]
        print("val_deformation_matrix_lag: ", val_deformation_matrix_lag.shape) # [B, 2, H, W]

        # compare registered cines with the ground truth

        # cine1

        cine1 = cine1.cpu()
        val_registered_cine1 = val_registered_cine1.cpu()
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(val_registered_cine1[8, 0, :, :], cmap='gray')
        plt.title('Registered Cine1')
        plt.subplot(1, 2, 2)
        plt.imshow(cine1[0, 8, :, :], cmap='gray')
        plt.title('Ground Truth Cine1')

        mse = ((val_registered_cine1[8, 0, :, :] - cine1[0, 8, :, :]) ** 2).mean()
        print("MSE Cine1: ", mse)

        plt.show()

        


if __name__ == '__main__':

    data_dir = './cardiac_tagging_motion_estimation/data/'
 
    # proposed model
    vol_size = (256, 256)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    test_model_path = './cardiac_tagging_motion_estimation/model/my'
    dst_root = './cardiac_tagging_motion_estimation/results/'
    if not os.path.exists(dst_root): os.mkdir(dst_root)
    test_Cardiac_Tagging_ME_net(net=net,
                             data_dir=data_dir,
                             model_path= test_model_path,
                             dst_root=dst_root,
                             case = 'proposed')










