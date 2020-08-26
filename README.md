# Procrustean Regression Networks
Source code for 'Procrustean Regression Networks: Learning 3D Structure of Non-Rigid Objects from 2D Annotations', ECCV 2020

NOTE : The code is written in MATLAB and Caffe framework. You need to install matcaffe to run the code. We are also developing pyTorch version which will soon be available.

## PRN-FCN for Human 3.6M dataset
This training and test code reproduces the results of Table 1 (*GT-ortho* and *GT-persp*) in the paper.
### Instructions
* Download Human 3.6M dataset. *D2 positions* and *D3 positions* will be used. You should also download *VISUALIZATION AND LARGE SCALE PREDICTION SOFTWARE* in the dataset download page.
* Run MATLAB and add the path to H36Mcode so that `db = H36MDataBase.instance()` can be executed.
* For training PRN-FCN, go to the folder `src_human36m_orthoGT.m` or `src_human36m_perspGT.m` and run `trainH36M_orthoGT.m` or `trainH36M_perspGT.m`. Set `h36mpath` variable to your dataset path before running the code.
* For testing, run `testH36M_orthoGT.m` or `testH36M_perspGT.m`. Set `h36mpath` variable to your dataset path before running the code. There are pretrained models in `models` folders which reproduces the results in the paper.
