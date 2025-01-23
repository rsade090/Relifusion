import os
import numpy as np
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import torch

class NuScenesDataset(Dataset):
    def __init__(self, root_dir, version='v1.0-mini', split='train', num_frames=5, transforms=None):
        """
        Dataset class for NuScenes data.

        :param root_dir: Path to the NuScenes dataset root.
        :param version: Dataset version (e.g., 'v1.0-mini').
        :param split: Data split ('train', 'val', 'test').
        :param num_frames: Number of consecutive frames to use (only for training).
        :param transforms: Optional transforms to apply to the data.
        """
        self.nusc = NuScenes(version=version, dataroot=root_dir, verbose=True)
        self.split = split
        self.num_frames = num_frames
        self.transforms = transforms

        # Filter scenes by split
        self.scenes = [s for s in self.nusc.scene if s['name'].startswith(split)]
        self.samples = [s for s in self.nusc.sample if self.nusc.get('scene', s['scene_token'])['name'] in [sc['name'] for sc in self.scenes]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a data sample.

        :param idx: Index of the sample.
        :return: A dictionary containing LiDAR and camera data.
        """
        sample = self.samples[idx]

        if self.split == 'train':
            return self._get_train_sample(sample)
        else:
            return self._get_test_sample(sample)

    def _get_train_sample(self, sample):
        """
        Retrieve training data, including multiple frames.

        :param sample: The starting sample.
        :return: A dictionary containing LiDAR and camera data for multiple frames.
        """
        frames = self._get_consecutive_frames(sample)
        lidar_data = []
        camera_data = {f'CAM_{view}': [] for view in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']}

        for frame in frames:
            lidar_token = frame['data']['LIDAR_TOP']
            lidar_data.append(self._load_lidar_data(lidar_token))

            for view in camera_data.keys():
                camera_token = frame['data'][view]
                camera_data[view].append(self._load_camera_data(camera_token))

        return {'lidar': lidar_data, 'camera': camera_data}

    def _get_test_sample(self, sample):
        """
        Retrieve testing data, including a single frame.

        :param sample: The sample to retrieve.
        :return: A dictionary containing LiDAR and camera data.
        """
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self._load_lidar_data(lidar_token)

        camera_data = {f'CAM_{view}': None for view in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']}
        for view in camera_data.keys():
            camera_token = sample['data'][view]
            camera_data[view] = self._load_camera_data(camera_token)

        return {'lidar': lidar_data, 'camera': camera_data}

    def _get_consecutive_frames(self, sample):
        """
        Retrieve consecutive frames starting from the given sample.

        :param sample: Starting sample.
        :return: A list of consecutive frames.
        """
        frames = []
        current_sample = sample
        for _ in range(self.num_frames):
            frames.append(current_sample)
            if not current_sample['next']:
                break
            current_sample = self.nusc.get('sample', current_sample['next'])
        return frames

    def _load_lidar_data(self, lidar_token):
        """
        Load LiDAR point cloud from the given token.

        :param lidar_token: Token for the LiDAR data.
        :return: LiDAR points as a numpy array.
        """
        lidar_path = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', lidar_token)['filename'])
        lidar_pointcloud = LidarPointCloud.from_file(lidar_path)
        return torch.tensor(lidar_pointcloud.points.T, dtype=torch.float32)

    def _load_camera_data(self, camera_token):
        """
        Load camera image from the given token.

        :param camera_token: Token for the camera data.
        :return: Image as a numpy array.
        """
        cam_path = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', camera_token)['filename'])
        image = mmcv.imread(cam_path)
        if self.transforms:
            image = self.transforms(image)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

# Example usage
if __name__ == "__main__":
    dataset = NuScenesDataset(root_dir="/path/to/nuscenes", version="v1.0-mini", split="train", num_frames=5)
    sample = dataset[0]
    print("Sample keys:", sample.keys())
