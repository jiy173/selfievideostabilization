# Real-Time Selfie Video Stabilization

This is the code for the paper "Real-Time Selfie Video Stabilization", CVPR 2021

Note: To use this code, you need to compile opencv-python from source with cuda and python support. 

Quick Start:

1. Download pretrained weights at 
2. Unzip the pretrained weights package. There are 5 files listed below:
 - 1.avi : an example video for demo
 - checkpt_fcn.pt : pretrained weight for the foreground/background segmentation
 - checkpt_stabnet.pt : pretrained weight for the selfie video stabilization network
 - default_face.npy : a default neutral pose 3D face in case no face is found in the frame
 - shape_predictor_68_face_landmarks.dat : used by the face landmark detector
3. Put "1.avi" under './example'
4. Put "checkpt_fcn.pt", "checkpt_stabnet.pt" and "default_face.npy" under './'
5. Put "shape_predictor_68_face_landmarks.dat" under "./landmark_detection"
6. Run "main.py", the stabilized result can be found in './result'

## Reference 
If you find our work useful, please cite our paper as:
````
@InProceedings{Selfie21,
  author       = "Jiyang Yu and Ravi Ramamoorthi and Keli Cheng and Michel Sarkis and Ning Bi",
  title        = "Real-Time Selfie Video Stabilization",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jun",
  year         = "2021"
}
````

