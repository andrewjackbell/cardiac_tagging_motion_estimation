import cv2
import numpy as np
import math

def apply_colormap(tensor, image_type):
    """
    Converts a tensor into a displayable image.
    - Displacement maps use red-blue colormap (negative = blue, positive = red).
    - Normal images are shown as grayscale.
    
    Args:
        tensor (numpy array): Input tensor (H x W).
        image_type (str): "displacement" for red-blue colormap, "normal" for grayscale.
    
    Returns:
        numpy array: Processed image in BGR format for OpenCV.
    """
    if image_type == "displacement":
        min_val, max_val = np.min(tensor), np.max(tensor)
        if min_val == max_val:  # Avoid division by zero
            return np.zeros((*tensor.shape, 3), dtype=np.uint8)

        # Normalize displacement values to [0,1] while preserving sign
        tensor = (tensor - min_val) / (max_val - min_val)

        # Apply Red-Blue colormap
        color_mapped = np.zeros((*tensor.shape, 3), dtype=np.uint8)
        color_mapped[..., 0] = (1 - tensor) * 255  # Blue for negative
        color_mapped[..., 2] = tensor * 255  # Red for positive
        return color_mapped
    
    elif image_type == "normal":
        # Normalize grayscale images to [0,255]
        tensor = np.clip(tensor, 0, 1) * 255
        return np.stack([tensor] * 3, axis=-1).astype(np.uint8)  # Convert to BGR

    else:
        raise ValueError("Invalid image_type. Use 'displacement' or 'normal'.")

def play_videos(tensors, image_types, fps=30):
    """
    Plays multiple cine images (3D tensors) as a looping video, using correct colormap per type.
    
    Args:
        tensors (list of numpy arrays): List of 3D tensors (Frames x Height x Width).
        image_types (list of str): List of "displacement" or "normal" for each tensor.
        fps (int): Frames per second for playback.
    """
    assert len(tensors) > 0, "At least one tensor must be provided."
    assert len(tensors) == len(image_types), "Each tensor must have a corresponding image type."

    num_frames = tensors[0].shape[0]
    assert all(tensor.shape[0] == num_frames for tensor in tensors), "All tensors must have the same number of frames."
    
    num_videos = len(tensors)
    grid_size = math.ceil(math.sqrt(num_videos))  # Compute a square grid size
    frame_height, frame_width = tensors[0].shape[1], tensors[0].shape[2]

    while True:  # Loop indefinitely
        for i in range(num_frames):
            frames = [apply_colormap(tensor[i], img_type) for tensor, img_type in zip(tensors, image_types)]
            
            # Arrange frames in a grid
            rows = []
            for r in range(grid_size):
                row_frames = frames[r * grid_size:(r + 1) * grid_size]
                while len(row_frames) < grid_size:  # Pad with black if needed
                    row_frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
                rows.append(np.hstack(row_frames))  # Stack horizontally
            
            grid_frame = np.vstack(rows)  # Stack vertically
            
            cv2.imshow("Cine Image Playback (Press 'q' to exit)", grid_frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return  # Exit
