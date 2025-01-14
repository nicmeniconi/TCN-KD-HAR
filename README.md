# Optimizing Unimodal Human Activity Recognition TCNs with Knowledge Distillation

This project focuses on optimizing Human Activity Recognition (HAR) models—specifically those using motion capture video data—through Knowledge Distillation (KD). We explore various model depths and observation windows of ResNet-like Temporal Convolutional Network models using the [**VIDIMU**](https://zenodo.org/records/8210563) dataset to evaluate the effectiveness of KD for improving unimodal HAR model performance.

The outcome of this project contributes to the development of efficient HAR systems suitable for real-time applications while maintaining robustness and accuracy.

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

## Applications of Multimodal HAR Systems

HAR plays a pivotal role in advancing applications across healthcare, gaming, and industrial automation. HAR models leverage data modalities such as positional, angular, and multimodal inputs to accurately and efficiently capture human movements. Understanding the strengths and limitations of each modality, along with innovative fusion methods, enables the development of robust and efficient HAR systems<sup>[1](#references)</sup>.

Multimodal HAR systems enhance activity recognition in diverse domains:
- **Tele-rehabilitation**: Remote patient monitoring with positional and angular data enables personalized therapy<sup>[2](#references)</sup>.
- **Assistive Technologies**: Multimodal models improve real-time decision-making in devices like exoskeletons and prosthetics<sup>[3](#references)</sup>.
- **Human-Computer Interaction**: These systems support intuitive interfaces for gaming, virtual reality, and robotics<sup>[4](#references)</sup>. 

In the realm of physical rehabilitation, designing effective at-home rehabilitation technologies requires a focus on minimally intrusive methods. Vision-based techniques hold significant promise, as they avoid the need for on-body sensors that could hinder patient movement<sup>[5](#references)</sup>. This approach reduces the discomfort associated with traditional motion capture suits and provides a more natural environment for rehabilitation exercises, potentially increasing patient compliance.

## Dataset and Approach

The VIDIMU dataset<sup>[6](#references)</sup> is a multimodal collection of data on daily life activities aimed at advancing human activity recognition and biomechanics research. It includes data from 54 participants, with video recordings from a commodity camera and inertial measurement unit (IMU) data from 16 participants. The dataset captures 13 clinically relevant activities using affordable technology. Video data was processed to estimate 3D joint positions (sampled at 50Hz), while IMU data provided joint angles through inverse kinematics (sampled at 30Hz). The dataset is designed to support studies in movement recognition, kinematic analysis, and tele-rehabilitation in natural environments. In this study, we will explore the effectiveness of KD for optimizing motion capture-based HAR student models using a multimodal motion capture and IMU-based HAR teacher model.

## Teacher Training and Student Pre-Training

### Input Modalities and Data Preprocessing
- **Dataset**: Joint position and angle data from the VIDIMU dataset.
  - **Alignment**: Video and IMU data were synchronized by minimizing the RMSE between joint angle signals after smoothing and resampling to the IMU sampling frequency<sup>[7](#references)</sup>.
  - **Joint Normalization**:
    -	Joint positions were normalized relative to the pelvis joint (root), centering the skeleton in local space for consistent representation across samples.
    -	The process involved subtracting the pelvis coordinates from all other joints to standardize positions within the frame.
    -	This normalization ensures model invariance to global position and focuses on relative joint movements, critical for robust activity recognition.
  - **Preprocessing**: Min-max normalization was applied separately for each modality using training set statistics.

Below is a visualization of a sample of joint position recordings (Subject S40, activities A01, A02, A04, A09), plotted using `vidimu/jointviz/normalize_joints_and_gif_out.ipynb`:

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A01_T01.gif" alt="Activity A01_T01" width="300" />
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A02_T01.gif" alt="Activity A02_T01" width="300" />
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A09_T02.gif" alt="Activity A09_T02" width="300" />
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/jointviz/outs/skeleton_3d_normalized_A04_T02.gif" alt="Activity A04_T02" width="300" />
</div>


### Model Architectures
We employed a ResNet-based architecture for both student and teacher models, incorporating Temporal Convolutional layers in place of standard Convolutional layers to effectively capture sequential dependencies in time-series data.
- **Pos_ResNet**: Joint position input
- **Ang_ResNet**: Joint angle input
- **AngPos_ResNet**: Combined joint position and angle input via parallel streams
- **Observation Durations**: 0.5s, 0.75s, 1.5s, and 4.0s
- **Number of Residual Blocks**: 1, 2, 7, 14

### Student and Teacher Training and Optimization
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience of 30 epochs
- **Cross Validation (CV)**: 5 folds
- **Reproducibility**: a fixed seed was set for the training routines to ensure consistent and reproducible results

## Results - Teacher Training and Student Pre-Training

Below are the average accuracies of the best unimodal and multimodal models across 5 CV folds, for each depth and observation window combinations.

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/modeling/evalCV.png" alt="Student and Teacher training results" width="1200" />
</div>

A more in-depth analysis of these models can be found in the `vidimu/modeling/eval.ipynb` notebook.

### Performance Across Observation Windows
- Shorter windows (0.5s, 0.75s) achieved higher performance.
- Longer windows (1.5s, 4.0s) exhibited diminishing returns.

### Performance Across Model Depths
- Shallow networks (1, 2) achieved higher performance.
- Deep networks (7, 14) exhibited diminishing returns.

### Model Comparison
- **Video input (Pos_ResNet)**: Strong with position-only data.
- **IMU input (Ang_ResNet)**: Less effective for longer windows.
- **Video_IMU input (AngPos_ResNet)**: Consistently outperformed single-modality models.

---

## KD for Video Model Optimization

### Training Details 
- **Models**: KD experiments were performed on the first fold of the cross validation experiments of every observation window and residual block number combinations.
- **Teacher Model**: Trained on joint position and angle data.
- **Student Model**: Position-only, fine-tuned via KD.
- **Distillation Settings**:
  - **Observation Windows**: 0.5s, 0.75s, 1.5s, and 4.0s
  - **Number of Residual Blocks**: 1, 2, 7, 14
  - **Alpha**: 0.7
  - **Temperature**: 5
  - **Reproducibility**: a fixed seed was set for the training routines to ensure consistent and reproducible results

### Results - KD for Video Model Optimization

Below are the accuracies of the untrained student, the teacher, the trained student, and the accuracy improvements achieved via KD for each depth and observation window combinations.

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/nicmeniconi/TCN-KD-HAR/blob/main/modeling/evalKD.png" alt="Student and Teacher training results" width="1200" />
</div>

A more in-depth analysis of these models can be found in the `vidimu/modeling/eval_kd.ipynb` notebook.

### Discussion

- In this experiment, we observe that KD provides marginal accuracy improvements for deeper models, and showed diminishing returns for shallow models. The largest performance gain was achieved by the deepest model (14 residual blocks) with the longest observation window (4.0s). 
- In the context of this dataset, applications may prioritize shallower models with shorter observation windows for real-time implementation. However, for more complex HAR problems—such as those involving larger activity sets or activities requiring recognition over longer time periods—KD can be beneficial for enhancing the performance of unimodal input models.
- Future improvements for shallow models could involve a grid search to optimize KD parameters (Alpha and Temperature) for each time window and model depth. Additionally, validating the current models using the unimodal subset (joint position data only) will help assess their ability to generalize to unseen subjects.

---

## How to Run and Analyze the Code

1. Download the [**VIDIMU**](https://zenodo.org/records/8210563) dataset.
    - Store `analysis.zip` and `dataset.zip` into a directory named `VIDIMU`, and unzip files.

2. Generate synchronized data by running the `vidimu-tools/synchronize/CropAndCheckSync.ipynb` notebook. Make sure to reference the correct path to the `VIDIMU` folder. The notebook will synchronize the `videoandimus` and store it in the directory: `VIDIMU/dataset/videoandimusyncrop`

3. Train models for HAR tasks:
    - Student and teacher training:
        - Adjust training parameters in `vidimu/modeling/run_gridsearchCV.sh` for grid search across different depths and observation windows for both student and teacher models. Update the path to specify the directory for storing model outputs and training logs.
        - Run grid search using command:
        ```bash
        bash vidimu/modeling/run_gridsearchCV.sh
        ```
    - Knowledge distillation:
        - Adjust training parameters in `vidimu/modeling/run_gridsearchKD.sh` to train all video input students and video-and-imu teachers for all observation window/model depth combinations. Update the path to specify the directory for storing model outputs and training logs.
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
- Modified `vidimu-tools/utils/syncUtilities.py` and `vidimu-tools/utils/fileProcessing.py` to support `vidimu-tools/synchronize/CropAndCheckSync.ipynb`.

---

## References

1. Aggarwal, J. K., & Ryoo, M. S. (2011). Human activity analysis: A review. ACM Computing Surveys, 43(3), 16.
2. Chen, Y., et al. (2020). Sensor-based activity recognition systems. IEEE Transactions on Mobile Computing, 19(2), 279-295.
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
4. Kong, F., & Fu, X. (2018). Human activity recognition in wearable computing. Computers in Human Behavior, 87, 324-336.
5. Kowshik Thopalli et al., "Advances in Computer Vision for Home-Based Stroke Rehabilitation" in Computer Vision: Challenges, Trends, and Opportunities, M. A. R. Ahad, M. Mahbub, M. Turk, and R. Hartley, Eds. Boca Raton, FL, USA: Routledge, 2024.
6. Doe, J., et al. (2023). VIDIMU: A multimodal dataset for human activity recognition. Dataset release.
7. twyncoder. (2023). vidimu-tools. GitHub. Retrieved from [https://github.com/twyncoder/vidimu-tools](https://github.com/twyncoder/vidimu-tools).

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

