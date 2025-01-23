import torch
import torch.nn as nn

class ConvMixerLayer(nn.Module):
    def __init__(self, dim, depth, kernel_size):
        """
        Single ConvMixer layer.
        
        :param dim: Feature dimension.
        :param depth: Depth of the mixer layers.
        :param kernel_size: Convolution kernel size.
        """
        super(ConvMixerLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.gelu(self.bn(self.depthwise_conv(x))) + x
        x = self.gelu(self.bn(self.pointwise_conv(x)))
        return x

class ConvMixerBackbone(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=256, depth=8, kernel_size=9):
        """
        ConvMixer backbone for processing multi-view camera images.

        :param input_channels: Number of input channels (e.g., RGB: 3).
        :param hidden_dim: Feature dimension.
        :param depth: Number of ConvMixer layers.
        :param kernel_size: Kernel size for depthwise convolutions.
        """
        super(ConvMixerBackbone, self).__init__()

        # Initial patch embedding layer
        self.patch_embed = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )

        # ConvMixer layers
        self.conv_mixer_layers = nn.Sequential(
            *[ConvMixerLayer(hidden_dim, depth, kernel_size) for _ in range(depth)]
        )

        # BEV transformation
        self.bev_transform = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Feature reduction
        self.fc = nn.Linear(hidden_dim * 28 * 50, 512)  # Assuming input images resized to 448x800

    def forward(self, camera_images):
        """
        Forward pass for the ConvMixer backbone.

        :param camera_images: Input images from multi-view cameras (batch size x 3 x H x W).
        :return: Processed features projected into BEV space.
        """
        # Patch embedding
        x = self.patch_embed(camera_images)

        # ConvMixer layers
        x = self.conv_mixer_layers(x)

        # BEV transformation
        bev_features = self.bev_transform(x)

        # Flatten and reduce dimensions
        batch_size = bev_features.size(0)
        bev_features = bev_features.view(batch_size, -1)  # Flatten
        bev_features = self.fc(bev_features)

        return bev_features

# Example usage
if __name__ == "__main__":
    model = ConvMixerBackbone()
    example_input = torch.rand(1, 3, 448, 800)  # Batch size 1, RGB image of size 448x800
    output = model(example_input)
    print("Output shape:", output.shape)
