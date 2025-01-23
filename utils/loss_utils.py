import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        """
        Multi-task loss combining detection, contrastive, temporal, and confidence losses.

        :param weights: List of weights for each loss component [λ1, λ2, λ3, λ4].
        """
        super(MultiTaskLoss, self).__init__()
        self.weights = weights if weights is not None else [1.0, 0.1, 0.2, 0.05]

        self.detection_loss = nn.MSELoss()
        self.contrastive_loss = nn.CrossEntropyLoss()
        self.temporal_loss = nn.MSELoss()
        self.confidence_loss = nn.BCELoss()

    def forward(self, detection_output, detection_target, contrastive_output, contrastive_target,
                temporal_output, temporal_target, confidence_output, confidence_target):
        """
        Compute the total multi-task loss.

        :param detection_output: Output of the detection head.
        :param detection_target: Ground truth for detection.
        :param contrastive_output: Output for contrastive learning.
        :param contrastive_target: Ground truth for contrastive learning.
        :param temporal_output: Temporal consistency output.
        :param temporal_target: Ground truth for temporal consistency.
        :param confidence_output: Confidence scores.
        :param confidence_target: Ground truth for confidence scores.
        :return: Combined loss.
        """
        loss_detection = self.detection_loss(detection_output, detection_target)
        loss_contrastive = self.contrastive_loss(contrastive_output, contrastive_target)
        loss_temporal = self.temporal_loss(temporal_output, temporal_target)
        loss_confidence = self.confidence_loss(confidence_output, confidence_target)

        total_loss = (
            self.weights[0] * loss_detection +
            self.weights[1] * loss_contrastive +
            self.weights[2] * loss_temporal +
            self.weights[3] * loss_confidence
        )

        return total_loss, {
            "detection_loss": loss_detection.item(),
            "contrastive_loss": loss_contrastive.item(),
            "temporal_loss": loss_temporal.item(),
            "confidence_loss": loss_confidence.item()
        }
