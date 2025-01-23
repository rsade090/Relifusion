import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import torch
from torch.utils.data import Dataset

class NuScenesDataset(Dataset):
    def __init__(self, root_dir, version='v1.0-mini', split='train', t=5, transforms=None):
        """
        Initialize the NuScenes dataset.

        :param root_dir: Path to the NuScenes dataset directory.
        :param version: Dataset version (e.g., 'v1.0-mini', 'v1.0-trainval').
        :param split: Data split ('train', 'val', 'test').
        :param t: Number of consecutive time steps for training mode.
        :param transforms: Transformations to apply to the data.
        """
        self.nusc = NuScenes(version=version, dataroot=root_dir, verbose=True)
        self.split = split
        self.t = t
        self.transforms = transforms

        # Get scene splits
        self.scenes = create_splits_scenes()[split]
        self.samples = [s for s in self.nusc.sample if self.nusc.get('scene', s['scene_token'])['name'] in self.scenes]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        :param idx: Index of the sample to retrieve.
        :return: A dictionary with LiDAR and camera data.
        """
        if self.split == 'train':
            return self._get_train_sample(idx)
        else:
            return self._get_test_sample(idx)

    def _get_train_sample(self, idx):
        """
        Get a training sample with t consecutive time steps.

        :param idx: Index of the starting sample.
        :return: A dictionary with 6 camera views and LiDAR data for t time steps.
        """
        start_sample = self.samples[idx]
        t_samples = self._get_consecutive_samples(start_sample, self.t)

        lidar_data = []
        camera_data = {f'CAM_{view}': [] for view in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']}

        for sample in t_samples:
            # Load LiDAR data
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_file = self.nusc.get('sample_data', lidar_token)['filename']
            lidar_pointcloud = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_file))
            lidar_data.append(torch.tensor(lidar_pointcloud.points[:3, :].T, dtype=torch.float32))

            # Load camera data
            for view in camera_data.keys():
                cam_token = sample['data'][view]
                cam_file = self.nusc.get('sample_data', cam_token)['filename']
                camera_data[view].append(cam_file)

        return {
            'lidar': lidar_data,
            'camera': camera_data
        }

    def _get_test_sample(self, idx):
        """
        Get a test sample with a single time step.

        :param idx: Index of the sample.
        :return: A dictionary with 6 camera views and LiDAR data for time t.
        """
        sample = self.samples[idx]

        # Load LiDAR data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_file = self.nusc.get('sample_data', lidar_token)['filename']
        lidar_pointcloud = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_file))
        lidar_data = torch.tensor(lidar_pointcloud.points[:3, :].T, dtype=torch.float32)

        # Load camera data
        camera_data = {f'CAM_{view}': None for view in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']}
        for view in camera_data.keys():
            cam_token = sample['data'][view]
            cam_file = self.nusc.get('sample_data', cam_token)['filename']
            camera_data[view] = cam_file

        return {
            'lidar': lidar_data,
            'camera': camera_data
        }

    def _get_consecutive_samples(self, start_sample, t):
        """
        Retrieve t consecutive samples starting from start_sample.

        :param start_sample: The initial sample token.
        :param t: Number of consecutive samples to retrieve.
        :return: A list of consecutive samples.
        """
        samples = []
        current_sample = start_sample
        for _ in range(t):
            samples.append(current_sample)
            if current_sample['next'] == "":  # End of the sequence
                break
            current_sample = self.nusc.get('sample', current_sample['next'])
        return samples
