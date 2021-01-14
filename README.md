# FastPose
FastPose - Fast, Efficient pose tracking and estimation with TFLite models

This is barebones code for getting started with TFLite and python for Pose Tracking and Estimation. openPose barely achieved 3-4 fps on my laptop and it was hard to use 
other heavy weight models on edge devices. So, FastPose can track 17 keypoints (same model as PoseNet) at an average 35 fps on an XPS 13 9370 and 24 fps on Raspberry Pi 4
with a coral usb accelerator. 

"Changes should be made to the code for use on edgeTPU/ Jetson family."
