import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, lidar_features, camera_features):
        # Compute reliability scores for each modality
        lidar_reliability = self.fc(lidar_features)
        camera_reliability = self.fc(camera_features)
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
    def __init__(self, input_dim, hidden_dim):
        super(ReliFusion, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1)
        )
        self.stfa = SpatioTemporalFeatureAggregation(input_dim, hidden_dim)
        self.reliability_module = ReliabilityModule(hidden_dim)
        self.cw_mca = ConfidenceWeightedMutualCrossAttention(hidden_dim)

    def forward(self, lidar_input, camera_input):
        # Extract features
        lidar_features = self.feature_extractor(lidar_input)
        camera_features = self.feature_extractor(camera_input)

        # Spatio-temporal feature aggregation
        spatial_features, temporal_features = self.stfa(lidar_features, camera_features)

        # Compute reliability scores
        lidar_confidence, camera_confidence = self.reliability_module(spatial_features, temporal_features)

        # Perform cross-attention
        fused_features = self.cw_mca(spatial_features, temporal_features, lidar_confidence, camera_confidence)

        return fused_features
