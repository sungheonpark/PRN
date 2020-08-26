lambda = 5e-2;
smallBatchNum = 32;
stride = 5;
h36mpath = '/home/hololo/PRN_CVPR/dataset/human36m/';
weightStr = 'models/net_H36M_orthoGT_iter_50000.caffemodel';
addpath(genpath('../funcs'))
train_H36M_orthoGT(lambda,smallBatchNum,stride,50000,100,h36mpath);
train_H36M_orthoGT(lambda,smallBatchNum,stride,70000,100,h36mpath,weightStr);