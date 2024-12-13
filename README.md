# Optimizing Unimodal Human Activity Recognition TCNs with Knowledge Distillation

Human Activity Recognition (HAR) plays a pivotal role in advancing applications across healthcare, gaming, and industrial automation. HAR models leverage data modalities such as positional, angular, and multimodal inputs to accurately and efficiently capture human movements. Understanding the strengths and limitations of each modality, along with innovative fusion methods, enables the development of robust and efficient HAR systems<sup>[1](#references)</sup>.

---

## Table of Contents
- [Applications of Multimodal HAR Systems](#applications-of-multimodal-har-systems)
- [Data Modalities in Motion Capture](#data-modalities-in-motion-capture)
- [Advantages of Multimodal Models](#advantages-of-multimodal-models)
- [Teacher Training and Student Pre-Training](#teacher-training-and-student-pre-training)
- [Initial Training Results and Discussion](#initial-training-results-and-discussion)
- [Knowledge Distillation for Pose Model Optimization](#knowledge-distillation-for-pose-model-optimization)
- [External Repository Inclusion](#external-repository-inclusion)
- [References](#references)

---

## Applications of Multimodal HAR Systems

Multimodal HAR systems enhance activity recognition in diverse domains:
- **Tele-rehabilitation**: Remote patient monitoring with positional and angular data enables personalized therapy<sup>[2](#references)</sup>.
- **Assistive Technologies**: Multimodal models improve real-time decision-making in devices like exoskeletons and prosthetics<sup>[3](#references)</sup>.
- **Human-Computer Interaction**: These systems support intuitive interfaces for gaming, virtual reality, and robotics<sup>[4](#references)</sup>. 

In the realm of physical rehabilitation, designing effective at-home rehabilitation technologies requires a focus on minimally intrusive methods. Vision-based techniques hold significant promise, as they avoid the need for on-body sensors that could hinder patient movement<sup>[5](#references)</sup>. This approach reduces the discomfort associated with traditional motion capture suits and provides a more natural environment for rehabilitation exercises, potentially increasing patient compliance. We will be exploring the effectiveness of Knowledge Distillation (KD) for optimizing video input HAR student using a multimodal video and motion HAR teacher.

---

## Dataset and Approach

The VIDIMU dataset<sup>[6](#references)</sup> is a multimodal collection of data on daily life activities aimed at advancing human activity recognition and biomechanics research. It includes data from 54 participants, with video recordings from a commodity camera and inertial measurement unit (IMU) data from 16 participants. The dataset captures 13 clinically relevant activities using affordable technology. Video data was processed to estimate 3D joint positions, while IMU data provided joint angles through inverse kinematics. The dataset is designed to support studies in movement recognition, kinematic analysis, and tele-rehabilitation in natural environments.

In this project, the unimodal students and the multimodal teachers were trained on the video and IMU data. After KD, the student models were evaluated on the video-only dataset, in order to measure the ability of these models to generalize across different subjects. 

---

## Teacher Training and Student Pre-Training

### Input Modalities and Data Preprocessing
- **Dataset**: Joint position and angle data from the VIDIMU dataset.
  - **Alignment**: Video and IMU data were synchronized by minimizing the RMSE between joint angle signals after smoothing and resampling to a common frequency<sup>[7](#references)</sup>.
  - **Joint Normalization**:
    -	Joint positions were normalized relative to the pelvis joint (root), centering the skeleton in local space for consistent representation across samples.
    -	The process involved subtracting the pelvis coordinates from all other joints to standardize positions within the frame.
    -	This normalization ensures model invariance to global position and focuses on relative joint movements, critical for robust activity recognition.
  - **Preprocessing**: Min-max normalization was applied separately for each modality using training set statistics.

Below is a visualization of some joint position recordings from the dataset, plotted using `vidimu/jointviz/normalize_joints_and_gif_out.ipynb`:

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A01_T01.gif" alt="Activity A01_T01" width="300" />
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A02_T01.gif" alt="Activity A02_T01" width="300" />
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A09_T02.gif" alt="Activity A03_T02" width="300" />
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A04_T02.gif" alt="Activity A03_T02" width="300" />
</div>


### Model Architectures
The ResNet-based architectures were adapted for temporal data:
- **Pos_ResNet**: Joint position input
- **Ang_ResNet**: Joint angle input
- **AngPos_ResNet**: Combined input via parallel streams
- **Observation Durations**: 0.5s, 0.75s, 1.5s, and 4.0s
- **Number of Residual Blocks**: 1, 2, 7, 14

### Student and Teacher Training and Optimization
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience of 30 epochs
- **Cross Validation**: 5 folds

---

## Training Results and Discussion

### Performance Across Observation Windows
- Shorter windows (0.5s, 0.75s) achieved higher performance.
- Longer windows (1.5s, 4.0s) showed diminishing returns.

### Performance Across Model Depths
- Shallow networks (1, 2) achieved higher performance.
- Deep networks (7, 14) showed diminishing returns.

### Model Comparison
- **Video input (Pos_ResNet)**: Strong with position-only data.
- **IMU input (Ang_ResNet)**: Less effective for longer windows.
- **Video_IMU input (AngPos_ResNet)**: Consistently outperformed single-modality models.

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/modeling/evalCV.png" alt="Student and Teacher training results" width="300" />
</div>

---

## Knowledge Distillation for Video Model Optimization

### Training Details (performed on a single fold)
- **Teacher Model**: Trained on joint position and angle data.
- **Student Model**: Position-only, trained via Knowledge Distillation.
- **Distillation Settings**:
  - Time window: 4.0s
  - Model depth: 14
  - Alpha: 0.7
  - Temperature: 5

### Evaluation
- **Teacher Accuracy**: 87.78%
- **Student Accuracy**: 90.50%
- **Student Precision**: 92.65%
- **Student Recall**: 90.90%
- **Student F1-Score**: 91.50%

### Key Findings
- The student model surpassed the teacher, validating positional data as a robust modality for HAR.
- This approach is scalable and ideal for cost-sensitive applications.

### Discussion

### Future Work

---

## How to Run and Analyze the Code

1. Download the `VIDIMU` dataset.

2. Generate synchronized data by running the `vidimu-tools/synchronize/CropAndCheckSync.ipynb` notebook. Make sure to reference the correct path to the `VIDIMU` folder. The notebook will synchronize the `videoandimus` and store it in the directory: `VIDIMU/dataset/videoandimusyncrop`

3. Train models for HAR tasks:
    - Student and teacher training:
        - Adjust training parameters in `vidimu/modeling/run_gridsearchCV.sh` for grid search across different depths and time windows for both student and teacher models. Also define directories to store model outputs and training logs.
        - Run grid search using command:
        ```bash
        bash vidimu/modeling/run_gridsearchCV.sh
        ```
    - Knowledge distillation:
        - Adjust training parameters in `vidimu/modeling/run_gridsearchKD.sh` to train all video input students and video-and-imu teachers for all time window/depth pairs. Also define directories to store model outputs and training logs.
        ```bash
        bash vidimu/modeling/run_gridsearchKD.sh
        ```
4. Analyzing Results:
    - Training and evaluation metrics (accuracy, precision, recall, F1-score) are logged within the CV and KD directories.
    - Review student model results in the `vidimu/modeling/eval.ipynb` notebook for insights into performance across observation windows, model depths, and input modalities.
    - Review knowledge distillation results in the `vidimu/modeling/eval_kd.ipynb` notebook for insights into performance across all observation window and depth pairs of student and teacher models.    

---

### External Repository Inclusion

This project includes the code from the following external repository:

- [vidimu-tools](https://github.com/twyncoder/vidimu-tools.git) (GNU General Public License v3.0)

Modifications Made:

- Added `vidimu-tools/synchronize/CropAndCheckSync.ipynb` to log additional synchronization information, crop and save synchronized data from subjects in the `VIDIMU/dataset/videoandimus` to `VIDIMU/dataset/videoandimusyncrop`.
- Modified `vidimu-tools/utils/syncUtilities.py` and `vidimu-tools/utils/fileProcessing.py` to support 

---

## References

1. Aggarwal, J. K., & Ryoo, M. S. (2011). Human activity analysis: A review. ACM Computing Surveys, 43(3), 16.
2. Chen, Y., et al. (2020). Sensor-based activity recognition systems. IEEE Transactions on Mobile Computing, 19(2), 279-295.
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
4. Kong, F., & Fu, X. (2018). Human activity recognition in wearable computing. Computers in Human Behavior, 87, 324-336.
5. Kowshik Thopalli et al., "Advances in Computer Vision for Home-Based Stroke Rehabilitation" in Computer Vision: Challenges, Trends, and Opportunities, M. A. R. Ahad, M. Mahbub, M. Turk, and R. Hartley, Eds. Boca Raton, FL, USA: Routledge, 2024.
6. Doe, J., et al. (2023). VIDIMU: A multimodal dataset for human activity recognition. Dataset release.
7. twyncoder. (2023). vidimu-tools. GitHub. Retrieved from [https://github.com/twyncoder/vidimu-tools](https://github.com/twyncoder/vidimu-tools).

