import torch
import torch.nn as nn

class VoxelNetBackbone(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        """
        A VoxelNet backbone for processing LiDAR point clouds.

        :param input_dim: Number of input channels (e.g., x, y, z, intensity).
        :param hidden_dim: Dimension of hidden layers.
        """
        super(VoxelNetBackbone, self).__init__()

        # Voxel feature encoding (VFE)
        self.vfe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 3D convolutional middle layers
        self.middle_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU()
        )

        # Feature reduction layer
        self.fc = nn.Linear(hidden_dim * 4, 256)  # Output feature dimension set to 256 

    def forward(self, voxel_data):
        """
        Forward pass for the VoxelNet backbone.

        :param voxel_data: Input LiDAR voxel data.
        :return: Processed features in BEV format.
        """
        # Voxel feature encoding
        voxel_features = self.vfe(voxel_data)

        # Reshape for 3D convolutions
        voxel_features = voxel_features.view(voxel_features.size(0), -1, 1, 1, 1)

        # Apply 3D convolutions
        features = self.middle_conv(voxel_features)

        # Global pooling and feature reduction
        features = torch.mean(features, dim=(2, 3, 4))  # Global average pooling
        features = self.fc(features)

        return features

# Example usage
if __name__ == "__main__":
    model = VoxelNetBackbone()
    example_input = torch.rand(1, 4, 32, 32, 32)  # Batch size 1, 4 channels, 32x32x32 voxels
    output = model(example_input)
    print("Output shape:", output.shape)
