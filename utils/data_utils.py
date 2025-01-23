import numpy as np
import torch
import os
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

# Utility functions for data processing

def voxelize_point_cloud(point_cloud, voxel_size=(0.075, 0.075, 0.2), point_range=(-50, -50, -3, 50, 50, 1.5)):
    """
    Voxelize a LiDAR point cloud.

    :param point_cloud: (N, 4) array representing LiDAR points (x, y, z, intensity).
    :param voxel_size: Tuple representing the size of each voxel.
    :param point_range: Tuple specifying the 3D bounding box to voxelize.
    :return: Voxel grid as a torch.Tensor.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_range
    voxel_x, voxel_y, voxel_z = voxel_size

    # Filter points within the range
    mask = (
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
        (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
    )
    points = point_cloud[mask]

    # Compute voxel indices
    indices = np.floor((points[:, :3] - np.array([x_min, y_min, z_min])) / voxel_size).astype(int)

    # Create a voxel grid
    voxel_shape = (
        int((x_max - x_min) / voxel_x),
        int((y_max - y_min) / voxel_y),
        int((z_max - z_min) / voxel_z)
    )
    voxel_grid = np.zeros(voxel_shape, dtype=np.float32)

    for idx in indices:
        voxel_grid[tuple(idx[:3])] += 1

    return torch.tensor(voxel_grid, dtype=torch.float32)


def project_lidar_to_image(lidar_points, camera_intrinsics, image_shape):
    """
    Project LiDAR points onto a camera image plane.

    :param lidar_points: (N, 3) array of LiDAR points (x, y, z).
    :param camera_intrinsics: (3, 3) array representing camera intrinsic matrix.
    :param image_shape: Tuple of (height, width) of the image.
    :return: Projected points as a numpy array.
    """
    # Homogeneous coordinates
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    projections = view_points(lidar_hom.T, camera_intrinsics, normalize=True).T

    # Filter points within image bounds
    mask = (
        (projections[:, 0] >= 0) & (projections[:, 0] < image_shape[1]) &
        (projections[:, 1] >= 0) & (projections[:, 1] < image_shape[0]) &
        (projections[:, 2] > 0)
    )
    return projections[mask]


def load_lidar_file(lidar_path):
    """
    Load a LiDAR point cloud from a file.

    :param lidar_path: Path to the LiDAR file.
    :return: LiDAR point cloud as a numpy array.
    """
    lidar_pointcloud = LidarPointCloud.from_file(lidar_path)
    return lidar_pointcloud.points.T


def normalize_image(image):
    """
    Normalize an image to [0, 1].

    :param image: Input image as a numpy array.
    :return: Normalized image as a numpy array.
    """
    return image / 255.0


if __name__ == "__main__":
    # Example usage
    example_lidar_path = "path/to/lidar/file.bin"
    example_image_shape = (448, 800)
    example_intrinsics = np.array([
        [1260, 0, 400],
        [0, 1260, 225],
        [0, 0, 1]
    ])

    lidar_data = load_lidar_file(example_lidar_path)
    voxel_grid = voxelize_point_cloud(lidar_data)
    projected_points = project_lidar_to_image(lidar_data[:, :3], example_intrinsics, example_image_shape)

    print("Voxel grid shape:", voxel_grid.shape)
    print("Projected points shape:", projected_points.shape)
