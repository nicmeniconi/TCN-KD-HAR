# VIDIMU-TOOLS

## Introduction

VIDIMU-TOOLS is a code repository related to the public dataset "VIDIMU. Multimodal video and IMU kinematic dataset on daily life activities using affordable devices."

## Related work 
- The scientific paper can be freely accessed in Scientific Data (Nature): https://doi.org/10.1038/s41597-023-02554-9
- The VIDIMU dataset can be freely accessed in Zenodo: https://doi.org/10.5281/zenodo.8210563

## Dependencies
The processing of raw data in the VIDIMU dataset was done using the scripts included in this repository, in combination with free tools [NVIDIA Maxine SDK](https://developer.nvidia.com/maxine) BodyTrack (v0.8)  and [OpenSim](https://opensim.stanford.edu) (v4.4).


## First steps
- Download the VIDIMU dataset to a folder in your local machine.
- In the desired Jupyter notebook included in VIDIMU-TOOLS, set the `fulldatasetpath` variable to the path of the previous folder.
## Repository files description
The code is organized in the following folder hierarchy:
- **imus** folder includes the following Jupyter notebooks: 
  - `PlotImusRawQuats.ipynb`: generates _.svg_ plots of the raw quaternion data acquired with custom IMU sensors and stored as _.raw_ files in the VIDIMU dataset.
  - `PlotImusIkJointAngles.ipynb`: generates _.svg_ plots of the joint angles estimated through Inverse Kinematics in [OpenSim](https://opensim.stanford.edu) and stored as _.mot_ files in the VIDIMU dataset.
- **video** folder includes the following Jupyter notebooks:
  - `ScriptsToBodyTrack.ipynb`: generates a list of video files stored as _.mp4_ in the VIDIMU dataset that can be used to process them with [NVIDIA Maxine SDK](https://developer.nvidia.com/maxine) BodyTrack.
  - `ConvertBodytrackToCSV.ipynb`: converts the plain text output of [NVIDIA Maxine SDK](https://developer.nvidia.com/maxine) BodyTrack stored as _.out_ files in the VIDIMU dataset, into comma separated values _.csv_ files.
  - `PlotVideoEstimatedJointAngles.ipynb`: generate plots the joint angles estimated from 3D joint positions inferred by [NVIDIA Maxine SDK](https://developer.nvidia.com/maxine) BodyTrack by reading _.csv_ files.
  - `RecodeMP4toSmallsizefiles.ipynb`: recodes original acquired and [NVIDIA Maxine SDK](https://developer.nvidia.com/maxine) BodyTrack generated _.mp4_ video files to significantly reduce their size and stores them in a different folder.
- **synchronize** folder includes the following Jupyter notebooks:
   - `EstimateFileSynchronization.ipynb`: computes ideal synchronization of IMU and video data records by estimating RMSE of shifted joint angles signals (_.mot_ and _.csv_ files), and writes this info to a file `infoToSync.csv`.
   - `ModifyFilesToSync.ipynb`: modify VIDIMU dataset files for estimated ideal synchronization according to `infoToSync.csv`.
- **utils** folder includes auxiliary Python functions employed in the Jupyter notebooks commented above.

## Reference

When using this code, please include a reference to this GitHub repository and the associated scientific paper:

Martínez-Zarzuela, M., González-Alonso, J., Antón-Rodríguez, M., Díaz-Pernas, F.J., Henning, M., Simón-Martínez, C. (2023) Multimodal video and IMU kinematic dataset on daily life activities using affordable devices. Sci Data 10, 648. https://doi.org/10.1038/s41597-023-02554-9.

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.txt)
