import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import torch


def create_point_animation(points, cine_images=None, connections=None, fps=30):
    """
    Create a high-quality animation of moving points with optional background and connections.
    
    Args:
        points: Tensor of shape [num_frames, num_points, 2] (in pixel coordinates)
        cine_images: Optional [num_frames, H, W] numpy array for background
        connections: List of point index pairs to connect with lines
        fps: Frames per second for animation
    """
    # Convert to numpy if needed
    points_np = points.cpu().numpy() if torch.is_tensor(points) else points
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.invert_yaxis()  # Match image coordinates
    
    # Initialize artists
    point_plot = ax.scatter([], [], c='red', s=50, edgecolor='white')
    
    # Add connections if specified
    line_collection = None
    if connections:
        lines = [[points_np[0,i], points_np[0,j]] for i,j in connections]
        line_collection = LineCollection(lines, colors='cyan', linewidths=1)
        ax.add_collection(line_collection)
    
    # Add background if provided
    bg_image = None
    if cine_images is not None:
        bg_image = ax.imshow(cine_images[0], cmap='gray', alpha=0.7)
    
    def update(frame):
        # Update points
        point_plot.set_offsets(points_np[frame])
        
        # Update connections
        if line_collection:
            lines = [[points_np[frame,i], points_np[frame,j]] for i,j in connections]
            line_collection.set_segments(lines)
        
        # Update background
        if bg_image:
            bg_image.set_array(cine_images[frame])
        
        return point_plot, line_collection, bg_image
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(points_np),
        interval=1000/fps, blit=True
    )
    
    plt.close()
    return ani