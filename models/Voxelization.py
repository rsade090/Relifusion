import numpy as np
import torch

def voxelize(point_cloud, voxel_size=(0.2, 0.2, 0.2), grid_size=(100, 100, 10)):
    """
    Voxelizes a 3D point cloud into a structured grid.

    Args:
        point_cloud (numpy.ndarray): Input point cloud of shape (N, 3), where N is the number of points.
        voxel_size (tuple): Size of each voxel in meters (x, y, z).
        grid_size (tuple): Number of voxels in (x, y, z) dimensions.

    Returns:
        voxels (torch.Tensor): Voxelized representation (D x H x W).
    """
    # Initialize the grid
    grid = np.zeros(grid_size, dtype=np.float32)
    
    # Normalize points to fit in the grid
    pc_min = np.min(point_cloud, axis=0)
    pc_max = np.max(point_cloud, axis=0)
    
    pc_normalized = (point_cloud - pc_min) / (pc_max - pc_min + 1e-6)
    pc_scaled = np.floor(pc_normalized * (np.array(grid_size) - 1)).astype(int)
    
    for x, y, z in pc_scaled:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
            grid[x, y, z] += 1  # Add the number of points in the voxel

    # Normalize voxel values
    grid = grid / np.max(grid) if np.max(grid) > 0 else grid
    return torch.tensor(grid).unsqueeze(0)  # Add channel dimension
