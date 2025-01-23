import torch
import torch.nn as nn
from VoxelNet import VoxelNetBackbone
from ConvMixer import ConvMixerBackbone

class SpatioTemporalFeatureAggregation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatioTemporalFeatureAggregation, self).__init__()
        self.spatial_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, spatial_features, temporal_features):
        # Spatial attention
        spatial_features, _ = self.spatial_attention(spatial_features, spatial_features, spatial_features)
        
        # Temporal attention
        temporal_features, _ = self.temporal_attention(temporal_features, temporal_features, temporal_features)
        
        # Residual connection and layer normalization
        temporal_features = self.norm(temporal_features + self.mlp(temporal_features))
        return spatial_features, temporal_features


class ReliabilityModule(nn.Module):
    def __init__(self, feature_dim):
        super(ReliabilityModule, self).__init__()
        # Contrastive Module for feature alignment
        self.contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.ReLU()
        )
        
        # Confidence Module for reliability scoring
        self.confidence_module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, lidar_features, camera_features):
        # Align features using the Contrastive Module
        lidar_aligned = self.contrastive_module(lidar_features)
        camera_aligned = self.contrastive_module(camera_features)

        # Compute reliability scores using the Confidence Module
        lidar_reliability = self.confidence_module(lidar_aligned)
        camera_reliability = self.confidence_module(camera_aligned)
        
        return lidar_reliability, camera_reliability


class ConfidenceWeightedMutualCrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ConfidenceWeightedMutualCrossAttention, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, lidar_features, camera_features, lidar_confidence, camera_confidence):
        # Apply cross-attention
        lidar_query = self.query_proj(lidar_features) * lidar_confidence
        camera_key = self.key_proj(camera_features)
        camera_value = self.value_proj(camera_features)
        
        cross_attention = torch.softmax(torch.matmul(lidar_query, camera_key.transpose(-2, -1)), dim=-1)
        fused_features = torch.matmul(cross_attention, camera_value)
        return fused_features


class ReliFusion(nn.Module):
    def __init__(self, lidar_input_dim=4, camera_input_channels=3, hidden_dim=256):
        super(ReliFusion, self).__init__()
        self.lidar_backbone = VoxelNetBackbone(input_dim=lidar_input_dim, hidden_dim=hidden_dim)
        self.camera_backbone = ConvMixerBackbone(input_channels=camera_input_channels, hidden_dim=hidden_dim)
        self.stfa = SpatioTemporalFeatureAggregation(hidden_dim, hidden_dim)
        self.reliability_module = ReliabilityModule(hidden_dim)
        self.cw_mca = ConfidenceWeightedMutualCrossAttention(hidden_dim)

    def forward(self, lidar_input, camera_input):
        # Extract features using backbones
        lidar_features = self.lidar_backbone(lidar_input)
        camera_features = self.camera_backbone(camera_input)

        # Spatio-temporal feature aggregation
        spatial_features, temporal_features = self.stfa(lidar_features, camera_features)

        # Compute reliability scores
        lidar_confidence, camera_confidence = self.reliability_module(spatial_features, temporal_features)

        # Perform cross-attention
        fused_features = self.cw_mca(spatial_features, temporal_features, lidar_confidence, camera_confidence)

        return fused_features

# Example usage
if __name__ == "__main__":
    model = ReliFusion()
    lidar_input = torch.rand(1, 4, 32, 32, 32)  # Example LiDAR input
    camera_input = torch.rand(1, 3, 448, 800)  # Example camera input
    output = model(lidar_input, camera_input)
    print("Output shape:", output.shape)
