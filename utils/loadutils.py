import os
import random
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from functools import partial

# parser = argparse.ArgumentParser()

# # TODO: define args in function arguments, and remove args from script

# # parser.add_argument("--seed", type=int, required=True)
# # parser.add_argument("--dpath", type=str, required=True)
# # parser.add_argument("--model_out_path", type=str, required=True)
# parser.add_argument("--activities", type=str, required=True)
# # parser.add_argument("--secs", type=float, required=True)
# # parser.add_argument("--split", type=float, required=True)
# # parser.add_argument("--batch_size", type=int, required=True)
# # parser.add_argument("--depth", type=int, required=True)
# # parser.add_argument("--epochs", type=int, required=True)
# # parser.add_argument("--lr", type=float, required=True)
# # parser.add_argument("--factor", type=float, required=True)
# # parser.add_argument("--patience", type=int, required=True)
# # parser.add_argument("--cv_folds", type=int, required=True)
# # parser.add_argument("--early_stop_patience", type=int, required=True) 
# # parser.add_argument("--modelname", type=str, required=True)
# args = parser.parse_args()

# activities = json.loads(args.activities)

video_joint_names = [
    'pelvis', 'left_hip', 'right_hip', 'torso', 'left_knee', 'right_knee', 'neck', 'left_ankle', 
    'right_ankle', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 
    'right_heel', 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky_knuckle', 'right_pinky_knuckle', 
    'left_middle_tip', 'right_middle_tip', 'left_index_knuckle', 'right_index_knuckle', 'left_thumb_tip', 'right_thumb_tip'
]

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)            # Python random module
    np.random.seed(seed)         # NumPy
    torch.manual_seed(seed)      # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cache_data(filepath, data):
    """Save data to disk as a PyTorch tensor."""
    torch.save(data, filepath)

def load_cached_data(filepath):
    """Load cached data if it exists."""
    if os.path.exists(filepath):
        return torch.load(filepath, weights_only=True)
    return None

class VIDIMU(Dataset):
    def __init__(self, data_dir, time_in_seconds, activities, ins='VIDIMU', out='act'):
        self.data_dir = data_dir
        self.time_in_seconds = time_in_seconds
        self.imu_sampling_rate = 50
        self.video_sampling_rate = 30
        self.activities = activities
        self.ins = ins
        self.out = out

        # Window sizes
        self.imu_window_size = int(self.time_in_seconds * self.imu_sampling_rate)
        self.video_window_size = int(self.time_in_seconds * self.video_sampling_rate)
        
        # Data file paths
        print("Fetching data files...")
        self.imu_files = self._get_data_files(extension=".mot", prefix="ik_")
        self.video_files = self._get_data_files(extension=".csv", prefix="S")
        print(f"Found {len(self.imu_files)} IMU files and {len(self.video_files)} video files")

        # Initialize data lists
        self.all_imu_windows = []
        self.all_video_windows = []
        self.all_activity_labels = []

        # Load and preprocess data
        self._load_data()  # This now generates fixed-size windows

    def _get_data_files(self, extension, prefix, activity=None):
        data_files = []
        for subject in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject)
            if os.path.isdir(subject_path):
                for activity_file in os.listdir(subject_path):
                    if activity_file.startswith(prefix) and activity_file.endswith(extension):
                        if activity is None or any(act in activity_file for act in self.activities):
                            data_files.append(os.path.join(subject_path, activity_file))
        return data_files

    def _get_activity_from_filename(self, filename):
        return os.path.basename(filename).split('_')[1]

    def _get_activity_index(self, activity_code):
        return self.activities.index(activity_code)

    def _load_data(self):
        print("Loading and preprocessing data...")
        for imu_file, video_file in zip(self.imu_files, self.video_files):
            # Load IMU data with explicit check for cache
            imu_cache_path = f"{imu_file}.pt"
            imu_data = load_cached_data(imu_cache_path)
            if imu_data is None:
                imu_data = pd.read_csv(
                    imu_file, skiprows=self._find_start_of_data(imu_file), sep=r'\s+', dtype=np.float32
                ).iloc[:, 1:].values
                imu_data = torch.tensor(imu_data, dtype=torch.float32)
                cache_data(imu_cache_path, imu_data)

            # Load video data with explicit check for cache
            video_cache_path = f"{video_file}.pt"
            video_data = load_cached_data(video_cache_path)
            if video_data is None:
                video_data = pd.read_csv(video_file, dtype=np.float32).iloc[:, 1:]
                video_data.columns = video_data.columns.str.strip()  # Strip whitespace from column names
                root_positions = video_data[['pelvis_x', 'pelvis_y', 'pelvis_z']].values

                # Normalize each joint relative to pelvis
                for joint in video_joint_names:
                    if joint != 'pelvis':
                        for axis in ['x', 'y', 'z']:
                            col = f'{joint}_{axis}'
                            if col in video_data.columns:
                                video_data[col] -= root_positions[:, 'xyz'.index(axis)]

                video_data = torch.tensor(video_data.values, dtype=torch.float32)
                cache_data(video_cache_path, video_data)

            # Generate and store fixed-size windows
            imu_windows = [
                imu_data[i:i + self.imu_window_size]
                for i in range(0, len(imu_data) - self.imu_window_size + 1, self.imu_window_size)
            ]
            video_windows = [
                video_data[i:i + self.video_window_size]
                for i in range(0, len(video_data) - self.video_window_size + 1, self.video_window_size)
            ]

            num_windows = min(len(imu_windows), len(video_windows))  # Ensure alignment
            self.all_imu_windows.extend(imu_windows[:num_windows])
            self.all_video_windows.extend(video_windows[:num_windows])
            activity_label = self._get_activity_index(self._get_activity_from_filename(video_file))
            self.all_activity_labels.extend([activity_label] * num_windows)

        # Convert lists to tensors and ensure uniform size
        self.all_imu_windows = torch.stack(self.all_imu_windows).permute(0, 2, 1)
        self.all_video_windows = torch.stack(self.all_video_windows).permute(0, 2, 1)
        self.all_activity_labels = torch.tensor(self.all_activity_labels, dtype=torch.long)

    def _find_start_of_data(self, filepath):
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if 'endheader' in line:
                    return i + 1
        return 0

    def __len__(self):
        return len(self.all_imu_windows)

    def __getitem__(self, idx):
        imu_window = self.all_imu_windows[idx]#.squeeze()
        video_window = self.all_video_windows[idx]#.squeeze()
        activity_label = self.all_activity_labels[idx]
        
        if self.ins == 'VIDIMU' and self.out == 'act':
            return video_window, imu_window, activity_label
        elif self.ins == 'VID' and self.out == 'act':
            return video_window, activity_label
        elif self.ins == 'IMU' and self.out == 'act':
            return imu_window, activity_label
        elif self.ins == 'VIDIMU' and self.out == 'IMU':
            return video_window, imu_window
        else:
            raise ValueError("Invalid ins/out combination")

def collate_fn(batch, ins, out, imu_min, imu_max, video_min, video_max):
    video_batch, imu_batch, labels = [], [], []
    for sample in batch:
        if ins == 'VIDIMU' and out == 'act':
            video, imu, label = sample
            imu_batch.append((imu - imu_min) / (imu_max - imu_min + 1e-8))
            video_batch.append((video - video_min) / (video_max - video_min + 1e-8))
            labels.append(label)
        elif ins == 'VID' and out == 'act':
            video, label = sample
            video_batch.append((video - video_min) / (video_max - video_min + 1e-8))
            labels.append(label)
        elif ins == 'IMU' and out == 'act':
            imu, label = sample
            imu_batch.append((imu - imu_min) / (imu_max - imu_min + 1e-8))
            labels.append(label)

    if ins == 'VIDIMU' and out == 'act':
        return torch.stack(video_batch), torch.stack(imu_batch), torch.tensor(labels)
    elif ins == 'VID' and out == 'act':
        return torch.stack(video_batch), torch.tensor(labels)
    elif ins == 'IMU' and out == 'act':
        return torch.stack(imu_batch), torch.tensor(labels)

def compute_scaling_params(dataset, train_indices, ins):

    imu_min, imu_max, video_min, video_max = None, None, None, None
    with torch.no_grad():

        if ins in ['IMU']:
            imu_data = torch.stack([dataset[i][0] for i in train_indices])
            imu_min = imu_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
            imu_max = imu_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0] 
        if ins in ['VID']:
            video_data = torch.stack([dataset[i][0] for i in train_indices])
            video_min = video_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
            video_max = video_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0]
        if ins in ['VIDIMU']:  
            video_data = torch.stack([dataset[i][0] for i in train_indices])  
            imu_data = torch.stack([dataset[i][1] for i in train_indices])
            imu_min = imu_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
            imu_max = imu_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0] 
            video_min = video_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
            video_max = video_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0]


    return imu_min, imu_max, video_min, video_max

def loaders_cv(dataset, fold, train_idx, test_idx, ins, out, batch_size, cv_folds, seed):
        print(f"Starting fold {fold + 1}/{cv_folds}")
        set_random_seed(seed + fold)

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Compute min-max scaling parameters on train set
        imu_min, imu_max, video_min, video_max = compute_scaling_params(dataset, train_idx, ins)

        # Partially bound collate function for scaling
        collate = partial(collate_fn, ins=ins, out=out, imu_min=imu_min, imu_max=imu_max, video_min=video_min, video_max=video_max)

        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        return train_loader, test_loader

# def standardize_vidimu(dpath, time_in_seconds=3.34, split=0.8, batch_size=32, activities=activities, ins='VIDIMU', out='act'):
def standardize_vidimu(dpath, time_in_seconds, split, batch_size, activities, ins, out):
    dataset = VIDIMU(data_dir=dpath, time_in_seconds=time_in_seconds, activities=activities, ins=ins, out=out)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=1 - split, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    with torch.no_grad():
        if ins in ['VIDIMU', 'IMU']:
            imu_data = torch.stack([dataset[i][1 if ins == 'VIDIMU' else 0] for i in train_idx])
            imu_min = imu_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
            imu_max = imu_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0]
            print(f"IMU min shape: {imu_min.shape}, IMU max shape: {imu_max.shape}")
        else:
            imu_min, imu_max = None, None

        if ins in ['VIDIMU', 'VID']:
            video_data = torch.stack([dataset[i][0] for i in train_idx])
            video_min = video_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
            video_max = video_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0]
            print(f"Video min shape: {video_min.shape}, Video max shape: {video_max.shape}")
        else:
            video_min, video_max = None, None

    collate = partial(collate_fn, ins=ins, out=out, imu_min=imu_min, imu_max=imu_max, video_min=video_min, video_max=video_max)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    return train_loader, test_loader