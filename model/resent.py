import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, output_dim=256):
        """
        A ResNet backbone for processing camera images from NuScenes data.

        :param pretrained: Whether to use a pretrained ResNet model.
        :param output_dim: Dimension of the output feature vector.
        """
        super(ResNetBackbone, self).__init__()

        # Load ResNet18
        self.resnet = resnet18(pretrained=pretrained)

        # Modify the fully connected layer to output the desired dimension
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, output_dim)

        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass for the ResNet backbone.

        :param x: Input camera images (B, C, H, W).
        :return: Extracted feature vectors.
        """
        features = self.resnet(x)
        projected_features = self.feature_proj(features)
        return projected_features

