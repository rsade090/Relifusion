# ReliFusion: Reliability-Driven LiDAR-Camera Fusion for Robust 3D Object Detection

## Overview
This repository contains the implementation of ReliFusion, a novel LiDAR-camera fusion framework designed to improve the robustness of 3D object detection in autonomous driving scenarios. ReliFusion addresses the challenges of sensor malfunctions and degraded data by dynamically adapting to sensor reliability.

### Key Features:
1. **Spatio-Temporal Feature Aggregation (STFA):** Captures spatial and temporal dependencies across frames for stable predictions.
2. **Reliability Module:** Uses Cross-Modality Contrastive Learning (CMCL) to quantify the reliability of sensor inputs.
3. **Confidence-Weighted Mutual Cross-Attention (CW-MCA):** Dynamically balances LiDAR and camera contributions based on reliability scores.
4. State-of-the-art performance on the **nuScenes** dataset, demonstrating significant improvements in accuracy and robustness.

---

## Installation
To set up the environment, clone this repository and install the dependencies in requirements.txt file.
For Loading Nuscenes Datset we have used MMDetection3d github repo related files.
