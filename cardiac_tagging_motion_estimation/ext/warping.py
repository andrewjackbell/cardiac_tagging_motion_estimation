import numpy as np
import torch.nn.functional as f
import torch
import matplotlib.pyplot as plt



def normalise_coords(tensor, max, range="0-1"):
    """
    Coords should be a tensor of any shape, but last dimension should be the number of dimensions in image space (D)
    Divisors should be a tensor of shape (D)
    Range should be string, either "0-1" or "-1-1"
    """

    coords = tensor / max

    if range == "0-1":   
        return coords
    elif range == "-1-1":
        return 2 * coords - 1
    
def denormalise_coords(tensor, max, range="0-1"):
    """
    Coords should be a tensor of any shape, but last dimension should be the number of dimensions in image space (D)
    Divisors should be a tensor of shape (D)
    Range should be string, either "0-1" or "-1-1"
    """

    if range == "0-1":   
        return tensor * max
    elif range == "-1-1":
        return (tensor + 1) / 2 * max
    
def warp_image(image, flow):
    """
    Warp image using flow field
    :param image: image to warp - shape should be (B, C, H, W)
    :param flow: flow field - shape should be (B, 2, H, W)
    :return: warped image - shape (B, C, H, W)
    """
    h, w = flow.shape[2:]

    origin_locations = torch.meshgrid([torch.arange(0, s) for s in (h, w)])
    origin_locations = torch.stack(origin_locations)  
    origin_locations = torch.unsqueeze(origin_locations, 0)  
    origin_locations = origin_locations.type(torch.FloatTensor)

    new_locations = origin_locations + flow 
    new_locations_norm = normalise_coords(new_locations, w, range="-1-1") 

    # flip the new locs to match frame orientation
    new_locations_flipped = new_locations_norm.flip(1)
    
    # Add permute to match grid_sample expected format
    new_locations_flipped = new_locations_flipped.permute(0, 2, 3, 1)

    output = f.grid_sample(input=image, grid=new_locations_flipped, mode='bilinear')

    return output
    

def warp_points(points, flow):
    """
    Warp points using flow field
    :param points: points to warp - shape should be (B, N_points, 2)
    :param flow: flow field - shape should be (B, 2, H, W)
    :return: warped points - shape (B, N_points, 2)
    
    """

    h, w = flow.shape[2:]

    origin_locations = torch.meshgrid([torch.arange(0, s) for s in (h, w)])
    origin_locations = torch.stack(origin_locations)  # 
    origin_locations = torch.unsqueeze(origin_locations, 0)  # add batch
    origin_locations = origin_locations.type(torch.FloatTensor)

    new_locations = origin_locations - flow # shape (B, 2, H, W) 
    new_locations_norm = normalise_coords(new_locations, w, range="-1-1") # shape (B, 2, H, W)

    # flip the new locs to match frame orientation
    new_locations_flipped = new_locations_norm.flip(1)

    points_t = points.unsqueeze(2) # shape (B, N_points, 1, 2)

    output = f.grid_sample(input=new_locations_flipped, grid=points_t, mode='bilinear') # shape (B, N_points, 1, 2)

    output = output.squeeze(2) # shape (B, N_points, 2)

    output_denorm = denormalise_coords(output, w, range="-1-1") # shape (B, N_points, 2)

    return output_denorm


# test the functions

if __name__=="__main__":

    thing = np.load('cardiac_tagging_motion_estimation/results/flow_stuff2.npz')

    es_flow = thing['es_flow']
    es_frame = thing['es_frame']
    ed_frame = thing['ed_frame']
    ed_points = thing['ed_points']
    es_points = thing['es_points']


    print(es_points.shape)
    print(ed_points.shape)
    print(es_flow.shape)
    print(es_frame.shape)
    print(ed_frame.shape)

    flow_prep = torch.tensor(es_flow).unsqueeze(0)
    ed_points_prep = torch.tensor(ed_points).unsqueeze(0)

    output = warp_points(ed_points_prep, flow_prep)

    output = output.squeeze(0)

    ed_points_denorm = denormalise_coords(ed_points, 128, range="-1-1")
    es_gt_denorm = denormalise_coords(es_points, 128, range="-1-1")

    plt.subplot(2,2, 1)
    plt.imshow(ed_frame, cmap='gray')
    plt.scatter(ed_points_denorm[:, 0], ed_points_denorm[:, 1], c='r', s=5)
    plt.title('ED Frame')

    plt.subplot(2,2, 2)
    plt.imshow(es_frame, cmap='gray')
    plt.scatter(es_gt_denorm[:, 0], es_gt_denorm[:, 1], c='b', s=5)
    plt.scatter(output[0, :], output[1, :], c='g', s=5)
    plt.title('ES Frame with Flow')

    plt.show()

    # now try warping the image

    ed_frame_prep = torch.tensor(ed_frame).unsqueeze(0).unsqueeze(0)

    output = warp_image(ed_frame_prep, flow_prep)

    output = output.squeeze(0).squeeze(0).numpy()

    plt.subplot(2,2,1)

    plt.imshow(ed_frame, cmap='gray')
    plt.title('ED Frame')
    plt.subplot(2,2,2)
    plt.imshow(output, cmap='gray')

    plt.title('PRed ES Frame')
    plt.subplot(2,2,3)
    plt.imshow(es_frame, cmap='gray')
    plt.title('ES Frame')
    plt.show()

