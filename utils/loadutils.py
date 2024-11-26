import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

'''
VIDandIMU

ins = 'VID', 'IMU', 'VIDIMU'
out = 'IMU', 'act'
'''

activities = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13"]

video_joint_names = ['pelvis', 'left_hip', 'right_hip', 'torso', 
                     'left_knee', 'right_knee', 'neck', 'left_ankle', 
                     'right_ankle', 'left_big_toe', 'right_big_toe', 'left_small_toe', 
                     'right_small_toe', 'left_heel', 'right_heel', 'nose', 
                     'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                     'left_wrist', 'right_wrist', 'left_pinky_knuckle', 'right_pinky_knuckle', 
                     'left_middle_tip', 'right_middle_tip', 'left_index_knuckle', 'right_index_knuckle', 
                     'left_thumb_tip', ' right_thumb_tip']



class VIDIMU(Dataset):
    def __init__(self, data_dir, time_in_seconds, activities, ins='VIDIMU', out='act'):
        self.data_dir = data_dir
        self.time_in_seconds = time_in_seconds
        self.imu_sampling_rate = 50
        self.video_sampling_rate = 30
        self.activities = activities
        self.ins = ins
        self.out = out

        # Compute the window sizes for IMU and video based on the time input
        self.imu_window_size = int(self.time_in_seconds * self.imu_sampling_rate)
        self.video_window_size = int(self.time_in_seconds * self.video_sampling_rate)
        
        # Fetch the IMU and video data files
        self.imu_files = self._get_data_files(extension=".mot", prefix="ik_")
        self.video_files = self._get_data_files(extension=".csv", prefix="S")
        
        # Initialize lists to store the stacked windows
        self.all_imu_windows = []
        self.all_video_windows = []
        self.all_activity_labels = []

        # Load and preprocess the data
        self._load_data()

    def _get_data_files(self, extension, prefix, activity=None):
        """Fetch the file paths based on extension, prefix, and optional activity code."""
        data_files = []
        for subject in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject)
            if os.path.isdir(subject_path):
                for activity_file in os.listdir(subject_path):
                    # Only include files with specified prefix and extension, and filter by activity if given
                    if activity_file.startswith(prefix) and activity_file.endswith(extension):
                        if activity is None or any(act in activity_file for act in self.activities):
                            data_files.append(os.path.join(subject_path, activity_file))
        return data_files

    def _get_activity_from_filename(self, filename):
        """Extract activity code from filename."""
        base_filename = os.path.basename(filename)
        activity_code = base_filename.split('_')[1]  # Adjust split if needed based on naming convention
        return activity_code

    def _get_activity_index(self, activity_code):
        """Get the index of an activity code in the activities list."""
        return self.activities.index(activity_code)

    def _load_data(self):
        """Load and stack windows from IMU and video data files."""
        for imu_file, video_file in zip(self.imu_files, self.video_files):
            # Load IMU data
            imu_data = pd.read_csv(imu_file, skiprows=self._find_start_of_data(imu_file), sep=r'\s+').iloc[:, 1:]
            imu_data = torch.tensor(imu_data.values, dtype=torch.float32)
            # print(f"IMU data shape after loading: {imu_data.shape}")

            # Load video data
            video_data = pd.read_csv(video_file).iloc[:, 1:]
            ########
            root_joint = 'pelvis'
            root_joint_abs = root_joint+'_abs'
            root_positions = video_data[[f'{root_joint}_x', f'{root_joint}_y', f'{root_joint}_z']].values

            # Subtract root positions to normalize
            normalized_data = video_data.copy()

            for joint in video_joint_names:
                if joint != root_joint:
                    normalized_data[[f'{joint}_x']] -= root_positions[:, 0][:, None]
                    normalized_data[[f'{joint}_y']] -= root_positions[:, 1][:, None]
                    normalized_data[[f'{joint}_z']] -= root_positions[:, 2][:, None]

            # Add root_x, root_y, and root_z
            
            # # # Absolute pelvis position before normalization
            # for joint in video_joint_names:
            #     if joint not in root_joint_abs:
            #         normalized_data[[f'{joint}_x']] -= root_positions[:, 0][:, None]
            #         normalized_data[[f'{joint}_y']] -= root_positions[:, 1][:, None]
            #         normalized_data[[f'{joint}_z']] -= root_positions[:, 2][:, None]
            # pelvis_abs_positions = pd.DataFrame({
            #     'pelvis_abs_x': video_data[f'{root_joint}_x'].values,
            #     'pelvis_abs_y': video_data[f'{root_joint}_y'].values,
            #     'pelvis_abs_z': video_data[f'{root_joint}_z'].values
            # })
            # # Concatenate the absolute positions to the normalized data
            # normalized_data = pd.concat([normalized_data, pelvis_abs_positions], axis=1)
            # print("Original first three video channels (after normalization):")
            # print(video_data.iloc[:, :3].head())  # Show the first few rows
            video_data = normalized_data.copy()
            ########

            video_data = torch.tensor(video_data.values, dtype=torch.float32)

            # Get activity label from the file name
            activity_code = self._get_activity_from_filename(video_file)
            activity_label = self._get_activity_index(activity_code)

            # Generate windows for both IMU and video data
            imu_windows = [
                imu_data[i:i + self.imu_window_size]
                for i in range(0, len(imu_data) - self.imu_window_size + 1, self.imu_window_size)
            ]
            video_windows = [
                video_data[i:i + self.video_window_size]
                for i in range(0, len(video_data) - self.video_window_size + 1, self.video_window_size)
            ]
            # print(f"Generated {len(imu_windows)} IMU windows, each of shape: {imu_windows[0].shape}")
            # print(f"Generated {len(video_windows)} video windows, each of shape: {video_windows[0].shape}")

            # Align the number of windows
            num_windows = min(len(imu_windows), len(video_windows))
            imu_windows = imu_windows[:num_windows]
            video_windows = video_windows[:num_windows]

            # Stack and store the windows with labels
            self.all_imu_windows.extend(imu_windows)
            self.all_video_windows.extend(video_windows)
            self.all_activity_labels.extend([activity_label] * num_windows)

        # Convert lists to tensors
        self.all_imu_windows = torch.stack(self.all_imu_windows).permute(0, 2, 1)
        self.all_video_windows = torch.stack(self.all_video_windows).permute(0, 2, 1)
        self.all_activity_labels = torch.tensor(self.all_activity_labels, dtype=torch.long)
        # print(f"Final all_imu_windows shape: {self.all_imu_windows.shape}")
        # print(f"Final all_video_windows shape: {self.all_video_windows.shape}")


    def _find_start_of_data(self, filepath):
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if 'endheader' in line:
                    return i + 1
        return 0

    def __len__(self):
        return len(self.all_imu_windows)

    def __getitem__(self, idx):
        imu_window = self.all_imu_windows[idx].squeeze() 
        video_window = self.all_video_windows[idx].squeeze() 
        activity_label = self.all_activity_labels[idx]

        # Define input-output combinations based on ins and out
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


def standardize_vidimu(dpath, time_in_seconds=3.34, split=0.8, batch_size=32, activities=activities, ins='VIDIMU', out='act'):
    dataset = VIDIMU(data_dir=dpath, time_in_seconds=time_in_seconds, activities=activities, ins=ins, out=out)
    # print(f"Dataset length: {len(dataset)}")

    # Split the dataset into training and testing sets
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=1 - split, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # Calculate min and max values for training set only
    if ins in ['VIDIMU', 'IMU']:
        imu_data = torch.stack([dataset[i][1 if ins == 'VIDIMU' else 0] for i in train_idx])  # IMU at index 1 in VIDIMU, index 0 in IMU
        # imu_min = imu_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=True)[0]
        # imu_max = imu_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        imu_min = imu_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
        imu_max = imu_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
        print('imu min/max shapes:', imu_min.shape, imu_max.shape)
    else:
        imu_min, imu_max = None, None

    if ins in ['VIDIMU', 'VID']:
        video_data = torch.stack([dataset[i][0] for i in train_idx])  # Video at index 0 in both VIDIMU and VID
        # video_min = video_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=True)[0]
        # video_max = video_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        video_min = video_data.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=False)[0]
        video_max = video_data.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=False)[0]
        print('video min/max shapes:', video_min.shape, video_max.shape)
    else:
        video_min, video_max = None, None

    def min_max_scale(data, data_min, data_max):
        return (data - data_min) / (data_max - data_min + 1e-8)

    def collate_fn(batch):
        video_batch, imu_batch, labels = [], [], []
        for sample in batch:
            if ins == 'VIDIMU' and out == 'act':
                video, imu, label = sample
                imu_batch.append(min_max_scale(imu, imu_min, imu_max))
                video_batch.append(min_max_scale(video, video_min, video_max))
                labels.append(label)
            elif ins == 'VID' and out == 'act':
                video, label = sample
                video_batch.append(min_max_scale(video, video_min, video_max))
                labels.append(label)
            elif ins == 'IMU' and out == 'act':
                imu, label = sample
                imu_batch.append(min_max_scale(imu, imu_min, imu_max))
                labels.append(label)
        # print(f"video_batch shape after scaling: {[v.shape for v in video_batch]}")
        # print(f"imu_batch shape after scaling: {[i.shape for i in imu_batch]}")
 
        # Return batches according to ins and out settings
        if ins == 'VIDIMU' and out == 'act':
            return torch.stack(video_batch), torch.stack(imu_batch), torch.tensor(labels)
        elif ins == 'VID' and out == 'act':
            return torch.stack(video_batch), torch.tensor(labels)
        elif ins == 'IMU' and out == 'act':
            return torch.stack(imu_batch), torch.tensor(labels)

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader