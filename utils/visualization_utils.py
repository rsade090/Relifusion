import matplotlib.pyplot as plt
import numpy as np
import torch

# Visualization utilities

def plot_lidar_points(lidar_points):
    """
    Plot LiDAR points in a 2D bird's-eye view.

    :param lidar_points: (N, 3) array of LiDAR points (x, y, z).
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(lidar_points[:, 0], lidar_points[:, 1], c=lidar_points[:, 2], s=1, cmap="viridis")
    plt.colorbar(label="Height (Z)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LiDAR Bird's-Eye View")
    plt.axis("equal")
    plt.show()


def overlay_lidar_on_image(image, lidar_projections):
    """
    Overlay LiDAR projections on a camera image.

    :param image: (H, W, 3) array representing the camera image.
    :param lidar_projections: (N, 2) array of projected LiDAR points (u, v).
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.scatter(lidar_projections[:, 0], lidar_projections[:, 1], c="red", s=1, label="LiDAR Points")
    plt.legend()
    plt.axis("off")
    plt.title("LiDAR Points Overlayed on Image")
    plt.show()


def plot_feature_map(feature_map, title="Feature Map", cmap="viridis"):
    """
    Visualize a single-channel feature map as a 2D image.

    :param feature_map: (H, W) array representing the feature map.
    :param title: Title of the plot.
    :param cmap: Colormap to use for visualization.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(feature_map, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_multiple_feature_maps(feature_maps, cols=4, cmap="viridis"):
    """
    Visualize multiple feature maps.

    :param feature_maps: (N, H, W) array of feature maps.
    :param cols: Number of columns in the grid.
    :param cmap: Colormap to use for visualization.
    """
    rows = (len(feature_maps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))

    for i, ax in enumerate(axes.flat):
        if i < len(feature_maps):
            ax.imshow(feature_maps[i], cmap=cmap)
            ax.set_title(f"Feature Map {i + 1}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    example_lidar_points = np.random.rand(1000, 3) * 50 - 25
    example_image = np.random.rand(448, 800, 3)
    example_projections = np.random.rand(500, 2) * [800, 448]
    example_feature_map = np.random.rand(64, 128)
    example_feature_maps = np.random.rand(8, 64, 128)

    plot_lidar_points(example_lidar_points)
    overlay_lidar_on_image(example_image, example_projections)
    plot_feature_map(example_feature_map)
    plot_multiple_feature_maps(example_feature_maps)
