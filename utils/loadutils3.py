import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

activities = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13"]

video_joint_names = [
    'pelvis', 'left_hip', 'right_hip', 'torso', 'left_knee', 'right_knee', 'neck', 'left_ankle', 
    'right_ankle', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 
    'right_heel', 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky_knuckle', 'right_pinky_knuckle', 
    'left_middle_tip', 'right_middle_tip', 'left_index_knuckle', 'right_index_knuckle', 'left_thumb_tip', 'right_thumb_tip'
]

def cache_data(filepath, data):
    """Save data to disk as a PyTorch tensor."""
    torch.save(data, filepath)

def load_cached_data(filepath):
    """Load cached data if it exists."""
    if os.path.exists(filepath):
        return torch.load(filepath, weights_only=True)
    return None

def preprocess_and_cache_data(video_file, imu_file, video_cache_path, imu_cache_path, find_start_of_data_func):
    """Preprocess and cache video and IMU data."""
    # Check for cached data first
    cached_video_data = load_cached_data(video_cache_path)
    cached_imu_data = load_cached_data(imu_cache_path)

    if cached_video_data is not None and cached_imu_data is not None:
        # print(f"Loaded cached data for {video_file} and {imu_file}")
        return cached_video_data, cached_imu_data

    # If not cached, preprocess and cache
    # print(f"Processing and caching data for {video_file} and {imu_file}")
    imu_data = pd.read_csv(imu_file, skiprows=find_start_of_data_func(imu_file), sep=r'\s+', dtype=np.float32).iloc[:, 1:]
    imu_data = torch.tensor(imu_data.values, dtype=torch.float32)

    video_data = pd.read_csv(video_file, dtype=np.float32).iloc[:, 1:]
    video_data.columns = video_data.columns.str.strip()  # Strip whitespace from column names
    root_positions = video_data[['pelvis_x', 'pelvis_y', 'pelvis_z']].values

    # Normalize each joint relative to pelvis
    for joint in video_joint_names:
        if joint != 'pelvis':
            for axis in ['x', 'y', 'z']:
                column_name = f'{joint}_{axis}'
                if column_name in video_data.columns:
                    video_data[column_name] -= root_positions[:, 'xyz'.index(axis)]
    
    video_data = torch.tensor(video_data.values, dtype=torch.float32)

    # Cache the preprocessed data
    cache_data(video_cache_path, video_data)
    cache_data(imu_cache_path, imu_data)
    
    return video_data, imu_data

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

        # Calculate windows per file for indexing
        sample_imu_data, _ = preprocess_and_cache_data(
            self.video_files[0],
            self.imu_files[0],
            f"{self.video_files[0]}.pt",
            f"{self.imu_files[0]}.pt",
            self._find_start_of_data  # Pass the method as an argument
        )
        self.num_windows_per_file = len(sample_imu_data) // self.imu_window_size

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

    def __len__(self):
        return len(self.imu_files) * self.num_windows_per_file
    
    def _find_start_of_data(self, filepath):
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if 'endheader' in line:
                    return i + 1
        return 0

    def __getitem__(self, idx):
        # Calculate file and window index
        file_idx = idx // self.num_windows_per_file
        window_idx = idx % self.num_windows_per_file
        imu_file = self.imu_files[file_idx]
        video_file = self.video_files[file_idx]

        # Load cached or preprocess data
        video_cache_path = f"{video_file}.pt"
        imu_cache_path = f"{imu_file}.pt"
        video_data, imu_data = preprocess_and_cache_data(
            video_file, imu_file, video_cache_path, imu_cache_path, self._find_start_of_data
        )

        # Generate windows on-the-fly
        imu_start = window_idx * self.imu_window_size
        video_start = window_idx * self.video_window_size
        imu_window = imu_data[imu_start:imu_start + self.imu_window_size]
        video_window = video_data[video_start:video_start + self.video_window_size]
        
        # Get activity label
        activity_code = self._get_activity_from_filename(video_file)
        activity_label = self._get_activity_index(activity_code)
        
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

    # Stack batches based on ins and out settings
    if ins == 'VIDIMU' and out == 'act':
        return torch.stack(video_batch), torch.stack(imu_batch), torch.tensor(labels)
    elif ins == 'VID' and out == 'act':
        return torch.stack(video_batch), torch.tensor(labels)
    elif ins == 'IMU' and out == 'act':
        return torch.stack(imu_batch), torch.tensor(labels)

def standardize_vidimu(dpath, time_in_seconds=3.34, split=0.8, batch_size=32, activities=activities, ins='VIDIMU', out='act'):
    dataset = VIDIMU(data_dir=dpath, time_in_seconds=time_in_seconds, activities=activities, ins=ins, out=out)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=1 - split, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

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

    # Use functools.partial to pass additional arguments to collate_fn
    from functools import partial
    collate = partial(collate_fn, ins=ins, out=out, imu_min=imu_min, imu_max=imu_max, video_min=video_min, video_max=video_max)

    # Set num_workers for parallel loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=4)

    return train_loader, test_loader