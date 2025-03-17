import sys
import os, time

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Parent of current directory

# Add root to path
sys.path.append(root_dir)


import torch
import torch.optim as optim
from ME_nets.LagrangianMotionEstimationNet import Lagrangian_motion_estimate_net
from losses.train_loss import VM_diffeo_loss, NCC
import numpy as np
from data_set.load_data_for_cine_ME import add_np_data, get_np_data_as_groupids,load_np_datagroups, DataType, \
    load_Dataset

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

def train_Cardiac_Tagging_ME_net(net, \
                                 data_dir, \
                                 batch_size, \
                                 n_epochs, \
                                 learning_rate, \
                                 model_path, \
                                 kl_loss, \
                                 recon_loss, \
                                 smoothing_loss):
    net.train()
    net.cuda()
    net = net.float()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # training start time
    training_start_time = time.time()


    # load training data

    train_set = 'dataset_train_100.npz'
    train_tags = np.load(os.path.join(data_dir,train_set))['arr_0']
    train_dataset = load_Dataset(train_tags)
    training_set_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_set = 'dataset_test_80.npz'
    val_tags = np.load(os.path.join(data_dir,val_set))['arr_0']
    val_dataset = load_Dataset(val_tags)
    val_batch_size = 1
    val_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)

    train_loss_dict = []
    val_loss_dict = []

    epoch_loss = 0
    for epoch in range(n_epochs):
        # print training log
        print("epochs = ", (epoch))
        print("." * 50)

        train_n_batches = len(training_set_loader)
        batch_loss = 0
        for i, batch in enumerate(training_set_loader):
            tag0 = batch
            tag = tag0[:, 2:, ::] # no grid frame

            tag = tag.to(device)
            img = tag.cuda()
            img = img.float()

            x = img[:, 1:, ::]  # other frames except the 1st frame
            y = img[:, 0:17, ::]  # 1st frame also is the reference frame
            shape = x.shape  # batch_size, seq_length, height, width
            batch_size = shape[0]
            seq_length = shape[1]
            height = shape[2]
            width = shape[3]
            x = x.contiguous()
            x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
            y = y.contiguous()
            y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            # set the param gradients as zero
            optimizer.zero_grad()
            # forward pass, backward pass and optimization
            registered_cine1, registered_cine2, registered_cine1_lag, flow_param,  \
            deformation_matrix, deformation_matrix_neg, deformation_matrix_lag = net(y, x)

            train_smoothing_loss = smoothing_loss(deformation_matrix)
            train_smoothing_loss_neg = smoothing_loss(deformation_matrix_neg)
            train_smoothing_loss_lag = smoothing_loss(deformation_matrix_lag)

            a = 5
            b = 1
            training_loss = kl_loss(x, flow_param) + 0.5 * recon_loss(x, registered_cine1) + \
                            0.5 * recon_loss(y, registered_cine2) + 0.5 * recon_loss(x, registered_cine1_lag) + \
                            a * train_smoothing_loss + a * train_smoothing_loss_neg + b * train_smoothing_loss_lag

            training_loss.backward()
            optimizer.step()
            # statistic
            epoch_loss += training_loss.item()


        epoch_loss = epoch_loss / train_n_batches
        train_loss_dict.append(epoch_loss)
        np.savetxt(os.path.join(model_path, 'train_loss.txt'), train_loss_dict, fmt='%.6f')
        print("training loss: {:.6f} ".format(epoch_loss))

        if (epoch) % 1 == 0:
            torch.save(net.state_dict(),
                       os.path.join(model_path, '{:d}_{:.4f}_model.pth'.format((epoch), epoch_loss)))

        # when the epoch is over do a pass on the validation set
        total_val_loss = 0
        val_n_batches = len(val_set_loader)

        for i, batch in enumerate(val_set_loader):
            tag0 = batch
            tag = tag0[:, 2:, ::]  # no grid frame
            val_batch_num_0 = tag0.shape
            val_batch_num = val_batch_num_0[0]
            tag = tag.to(device)
            img = tag.cuda()
            img = img.float()

            x = img[:, 1:, ::]  # other frames except the 1st frame
            y = img[:, 0:17, ::]  # 1st frame also is the reference frame
            shape = x.shape  # batch_size, seq_length, height, width
            batch_size = shape[0]
            seq_length = shape[1]
            height = shape[2]
            width = shape[3]
            x = x.contiguous()
            x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            y = y.contiguous()
            y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            # forward pass
            val_registered_cine1, val_registered_cine2, val_registered_cine1_lag, \
            val_flow_param, val_deformation_matrix, val_deformation_matrix_neg, val_deformation_matrix_lag = net(y, x)

            val_smoothing_loss = smoothing_loss(val_deformation_matrix)
            val_smoothing_loss_neg = smoothing_loss(val_deformation_matrix_neg)
            val_smoothing_loss_lag = smoothing_loss(val_deformation_matrix_lag)

            a = 5
            b = 1
            val_loss = kl_loss(x, val_flow_param) + 0.5*recon_loss(x, val_registered_cine1) + \
                       0.5*recon_loss(y, val_registered_cine2) + 0.5*recon_loss(x, val_registered_cine1_lag) + \
                       a * val_smoothing_loss + a * val_smoothing_loss_neg + b * val_smoothing_loss_lag

            val_loss = val_loss / val_batch_num
            total_val_loss += val_loss.item()

        val_epoch_loss = total_val_loss / val_n_batches
        val_loss_dict.append(val_epoch_loss)
        np.savetxt(os.path.join(model_path, 'val_loss.txt'), val_loss_dict, fmt='%.6f')

        print("validation loss: {:.6f} ".format(val_epoch_loss))

    torch.save(net.state_dict(), os.path.join(model_path, 'end_model.pth'))
    print("Training finished! It took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
 
    data_dir = './cardiac_tagging_motion_estimation/data'
    training_model_path = './cardiac_tagging_motion_estimation/model/'

    if not os.path.exists(training_model_path):
        os.mkdir(training_model_path)
    n_epochs = 10
    learning_rate = 5e-4
    batch_size = 1
    print("......HYPER-PARAMETERS 4 TRAINING......")
    print("batch size = ", batch_size)
    print("learning rate = ", learning_rate)
    print("." * 30)

    # proposed model
    vol_size = (256, 256)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    loss_class = VM_diffeo_loss(image_sigma=0.02, prior_lambda=10, flow_vol_shape=vol_size).cuda()
    my_ncc_loss = NCC()

    train_Cardiac_Tagging_ME_net(net=net,
                         data_dir=data_dir,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         model_path=training_model_path,
                         kl_loss=loss_class.kl_loss,
                         recon_loss=my_ncc_loss,
                         smoothing_loss = loss_class.gradient_loss
                         )








