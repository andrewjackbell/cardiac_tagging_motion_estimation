import torch

def preprocess_tensors(images, points, crop=False):
    """
    Preprocesses image and point arrays directly.
    Args:
        images: tensor of shape [N, F, H, W]
        points: tensor of shape [N, F, n_points, 2]
        crop: boolean, whether to crop images using bounding box
    Returns:
        processed_images: normalized [0,1] tensor
        processed_points: normalized [-1,1] tensor
    """
    if crop:
        processed_images = []
        processed_points = []
        # Process each case separately when cropping
        for img, pts in zip(images, points):
            cropped_img, cropped_pts = crop_one_case(img, pts)
            processed_images.append(cropped_img / 255.0)
            processed_points.append((2*cropped_pts / 128) - 1)
        return torch.stack(processed_images), torch.stack(processed_points)
    else:
        # Process full arrays at once when not cropping
        processed_images = images / 255.0
        processed_points = (2*points / 256) - 1
        return processed_images, processed_points
    

def crop_image_tensor(images, bb, output_size=(128, 128)):
    """
    Crop the input image tensor to the bounding box specified by bb.
    The cropped image is then resized to the output_size using bicubic interpolation.

    images: (n_frames, height, width)
    bb: (4) tensor with the bounding box values
    output_size: (2) tuple with the output size
    """
    
    n_frames = images.shape[0]
    cropped_images = torch.zeros(n_frames, *output_size)
    for i in range(n_frames):
        x1, y1, x2, y2 = bb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image_section = images[i, y1:y2, x1:x2]
        # convert to float
        image_section = image_section.float()
        result = torch.nn.functional.interpolate(image_section.unsqueeze(0).unsqueeze(0), size=output_size, mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
        cropped_images[i] = torch.clamp(result, 0, 255)

    return cropped_images

def move_points(points, bb, output_size=(128, 128)):
    '''
    The points are moved to the new coordinate system defined by the bounding box. 
    '''

    x1, y1, x2, y2 = bb
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    width, height = output_size

    moved_points = points - torch.tensor([x1, y1]) # move to origin
    moved_points = moved_points / torch.tensor([x2 - x1, y2 - y1]) # scale to 1
    moved_points = moved_points * torch.tensor([width, height]) # scale to new size

    return moved_points


def crop_one_case(images, points):

    bounding_box = find_bounding_box(points[0]) # only use first frame for bb

    cropped_images = crop_image_tensor(images, bounding_box, (128, 128))
    moved_points = move_points(points, bounding_box, (128, 128))

    return cropped_images, moved_points



   

def find_bounding_box(points, margin=0, mutliplier=1.6):
    '''
    Given a set of points with x,y coordinates, find the bounding box that contains all the points.
    Add a margin to the bounding box if required. And then scale by a multiplier.

    Returns the bounding box in the format (min_x, min_y, max_x, max_y)
    '''

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    # scale the bounding box by a multiplier

    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2

    min_x = x_center - (x_center - min_x) * mutliplier
    max_x = x_center + (max_x - x_center) * mutliplier
    min_y = y_center - (y_center - min_y) * mutliplier
    max_y = y_center + (max_y - y_center) * mutliplier


    return min_x, min_y, max_x, max_y
