import os
import sys
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

jointDict = {
    # Pelvis
    "pelvis_tilt": ['pelvis', 'torso', 'pelvis', 'neck'],
    "pelvis_list": ['pelvis', 'left_hip', 'pelvis', 'right_hip'],
    "pelvis_rotation": ['pelvis', 'left_hip', 'pelvis', 'right_hip'],
    "pelvis_tx": ['pelvis', 'left_hip', 'pelvis', 'right_hip'],
    "pelvis_ty": ['pelvis', 'left_hip', 'pelvis', 'right_hip'],
    "pelvis_tz": ['pelvis', 'left_hip', 'pelvis', 'right_hip'],
    
    # Right Leg
    "hip_flexion_r": ['pelvis', 'right_hip', 'right_hip', 'right_knee'],
    "hip_adduction_r": ['pelvis', 'right_hip', 'right_hip', 'right_knee'],
    "hip_rotation_r": ['pelvis', 'right_hip', 'right_hip', 'right_knee'],
    "knee_angle_r": ['right_hip', 'right_knee', 'right_knee', 'right_ankle'],
    "knee_angle_r_beta": ['right_hip', 'right_knee', 'right_knee', 'right_ankle'],
    "ankle_angle_r": ['right_knee', 'right_ankle', 'right_ankle', 'right_big_toe'],
    "subtalar_angle_r": ['right_ankle', 'right_heel', 'right_ankle', 'right_big_toe'],
    "mtp_angle_r": ['right_ankle', 'right_big_toe', 'right_big_toe', 'right_small_toe'],

    # Left Leg
    "hip_flexion_l": ['pelvis', 'left_hip', 'left_hip', 'left_knee'],
    "hip_adduction_l": ['pelvis', 'left_hip', 'left_hip', 'left_knee'],
    "hip_rotation_l": ['pelvis', 'left_hip', 'left_hip', 'left_knee'],
    "knee_angle_l": ['left_hip', 'left_knee', 'left_knee', 'left_ankle'],
    "knee_angle_l_beta": ['left_hip', 'left_knee', 'left_knee', 'left_ankle'],
    "ankle_angle_l": ['left_knee', 'left_ankle', 'left_ankle', 'left_big_toe'],
    "subtalar_angle_l": ['left_ankle', 'left_heel', 'left_ankle', 'left_big_toe'],
    "mtp_angle_l": ['left_ankle', 'left_big_toe', 'left_big_toe', 'left_small_toe'],

    # Spine and Lumbar
    "lumbar_extension": ['pelvis', 'torso', 'torso', 'neck'],
    "lumbar_bending": ['pelvis', 'torso', 'torso', 'neck'],
    "lumbar_rotation": ['pelvis', 'torso', 'torso', 'neck'],

    # Right Arm
    "arm_flex_r": ['right_shoulder', 'right_elbow', 'neck', 'torso'],
    "arm_add_r": ['right_shoulder', 'right_elbow', 'right_shoulder', 'right_wrist'],
    "arm_rot_r": ['right_shoulder', 'right_elbow', 'right_shoulder', 'right_wrist'],
    "elbow_flex_r": ['right_shoulder', 'right_elbow', 'right_elbow', 'right_wrist'],
    "pro_sup_r": ['right_elbow', 'right_wrist', 'right_elbow', 'right_wrist'],
    "wrist_flex_r": ['right_elbow', 'right_wrist', 'right_wrist', 'right_pinky_knuckle'],
    "wrist_dev_r": ['right_elbow', 'right_wrist', 'right_wrist', 'right_pinky_knuckle'],

    # Left Arm
    "arm_flex_l": ['left_shoulder', 'left_elbow', 'neck', 'torso'],
    "arm_add_l": ['left_shoulder', 'left_elbow', 'left_shoulder', 'left_wrist'],
    "arm_rot_l": ['left_shoulder', 'left_elbow', 'left_shoulder', 'left_wrist'],
    "elbow_flex_l": ['left_shoulder', 'left_elbow', 'left_elbow', 'left_wrist'],
    "pro_sup_l": ['left_elbow', 'left_wrist', 'left_elbow', 'left_wrist'],
    "wrist_flex_l": ['left_elbow', 'left_wrist', 'left_wrist', 'left_pinky_knuckle'],
    "wrist_dev_l": ['left_elbow', 'left_wrist', 'left_wrist', 'left_pinky_knuckle']
}

def cache_data(filepath, data):
    """Save data to disk as a PyTorch tensor."""
    torch.save(data, filepath)

def load_cached_data(filepath):
    """Load cached data if it exists."""
    if os.path.exists(filepath):
        return torch.load(filepath, weights_only=True)
    return None

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    epsilon = sys.float_info.epsilon
    return vector / (np.linalg.norm(vector) + epsilon)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2': """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    #c = np.cross(v2_u,v1_u)
    #if(c>0): 
    #    angle+= 180 
    return angle/np.pi * 180

def getJointAngleCsvAsNP(dfcsv, jointDict):
    bonesCSV = {}
    for ang in jointDict.keys():
        jointlist = []
        for j in jointDict[ang]:
            joint = dfcsv[[f"{j}_{axis}" for axis in ['x', 'y', 'z']]].to_numpy()
            jointlist.append(joint)
        bonesCSV[ang] = jointlist

    jointangles_video = pd.DataFrame(columns=jointDict.keys())

    for ang, bones in bonesCSV.items():
        bone1, bone2, bone3, bone4 = bonesCSV[ang]
        segmA = bone1 - bone2
        segmB = bone3 - bone4
        jointangle_video = np.zeros((segmA.shape[0]))
        for idx, (v1, v2) in enumerate(zip(segmA, segmB)):
            jointangle_video[idx] = angle_between(v2, v1)
        jointangles_video[ang] = jointangle_video

    return jointangles_video

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
        self.imu_files = self._get_data_files(extension=".mot", prefix="ik_")
        self.video_files = self._get_data_files(extension=".csv", prefix="S")

        # Initialize data lists
        self.all_imu_windows = []
        self.all_video_windows = []
        self.all_activity_labels = []

        # Load and preprocess data
        self._load_data()

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
        for imu_file, video_file in zip(self.imu_files, self.video_files):
            imu_data = pd.read_csv(imu_file, skiprows=self._find_start_of_data(imu_file), sep=r'\s+', dtype=np.float32).iloc[:, 1:].values
            imu_data = torch.tensor(imu_data, dtype=torch.float32)

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

            # Apply getJointAngleCsvAsNP to transform positions to angles
            video_data = getJointAngleCsvAsNP(video_data, jointDict)
            video_data = torch.tensor(video_data.values, dtype=torch.float32)

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
        imu_window = self.all_imu_windows[idx]
        video_window = self.all_video_windows[idx]
        activity_label = self.all_activity_labels[idx]
        
        if self.ins == 'VIDIMU' and self.out == 'act':
            return video_window, imu_window, activity_label
        elif self.ins == 'VID' and self.out == 'act':
            return video_window, activity_label
        elif self.ins == 'IMU' and self.out == 'act':
            return imu_window, activity_label
        elif self.ins == 'VID' and self.out == 'IMU':
            return video_window, imu_window
        else:
            raise ValueError("Invalid ins/out combination")

def collate_fn(batch, ins, out, imu_min=None, imu_max=None, video_min=None, video_max=None):
    video_batch, imu_batch, labels = [], [], []
    for sample in batch:
        if ins == 'VIDIMU' and out == 'act':
            video, imu, label = sample
            if imu_min is not None and imu_max is not None:
                imu = (imu - imu_min) / (imu_max - imu_min + 1e-8)
            if video_min is not None and video_max is not None:
                video = (video - video_min) / (video_max - video_min + 1e-8)
            imu_batch.append(imu)
            video_batch.append(video)
            labels.append(label)
        elif ins == 'VID' and out == 'act':
            video, label = sample
            if video_min is not None and video_max is not None:
                video = (video - video_min) / (video_max - video_min + 1e-8)
            video_batch.append(video)
            labels.append(label)
        elif ins == 'IMU' and out == 'act':
            imu, label = sample
            if imu_min is not None and imu_max is not None:
                imu = (imu - imu_min) / (imu_max - imu_min + 1e-8)
            imu_batch.append(imu)
            labels.append(label)
        elif ins == 'VID' and out == 'IMU':
            video, imu = sample
            if imu_min is not None and imu_max is not None:
                imu = (imu - imu_min) / (imu_max - imu_min + 1e-8)
            if video_min is not None and video_max is not None:
                video = (video - video_min) / (video_max - video_min + 1e-8)
            video_batch.append(video)
            imu_batch.append(imu)

    # Stack batches based on ins and out settings
    if ins == 'VIDIMU' and out == 'act':
        return torch.stack(video_batch), torch.stack(imu_batch), torch.tensor(labels)
    elif ins == 'VID' and out == 'act':
        return torch.stack(video_batch), torch.tensor(labels)
    elif ins == 'IMU' and out == 'act':
        return torch.stack(imu_batch), torch.tensor(labels)
    elif ins == 'VID' and out == 'IMU':
        return torch.stack(video_batch), torch.stack(imu_batch)

def standardize_vidimu(dpath, time_in_seconds=3.34, split=0.8, batch_size=32, activities=activities, ins='VIDIMU', out='act'):
    dataset = VIDIMU(data_dir=dpath, time_in_seconds=time_in_seconds, activities=activities, ins=ins, out=out)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=1 - split, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    with torch.no_grad():
        imu_min, imu_max, video_min, video_max = None, None, None, None

        if ins == 'VIDIMU':
            imu_data = torch.stack([dataset[i][1] for i in train_idx])  # IMU data extraction
            imu_min = imu_data.amin(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [39, 1]
            imu_max = imu_data.amax(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [39, 1]
            print(f"Computed IMU min shape: {imu_min.shape}, IMU max shape: {imu_max.shape}")

            video_data = torch.stack([dataset[i][0] for i in train_idx])  # Video data extraction
            video_min = video_data.amin(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [102, 1]
            video_max = video_data.amax(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [102, 1]
            print(f"Computed Video min shape: {video_min.shape}, Video max shape: {video_max.shape}")

        if ins == 'VID':
            video_data = torch.stack([dataset[i][0] for i in train_idx])
            video_min = video_data.amin(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [102, 1]
            video_max = video_data.amax(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [102, 1]
            print(f"Computed Video min shape: {video_min.shape}, Video max shape: {video_max.shape}")

            if out == 'IMU':
                imu_data = torch.stack([dataset[i][1] for i in train_idx])
                imu_min = imu_data.amin(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [39, 1]
                imu_max = imu_data.amax(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [39, 1]
                print(f"Computed IMU min shape: {imu_min.shape}, IMU max shape: {imu_max.shape}")

        if ins == 'IMU':
            imu_data = torch.stack([dataset[i][0] for i in train_idx])
            imu_min = imu_data.amin(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [39, 1]
            imu_max = imu_data.amax(dim=(0, 2), keepdim=True).squeeze(0)  # Shape: [39, 1]
            print(f"Computed IMU min shape: {imu_min.shape}, IMU max shape: {imu_max.shape}")

    # Use functools.partial to pass additional arguments to collate_fn
    from functools import partial
    collate = partial(collate_fn, ins=ins, out=out, imu_min=imu_min, imu_max=imu_max, video_min=video_min, video_max=video_max)

    # Set num_workers for parallel loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    return train_loader, test_loader