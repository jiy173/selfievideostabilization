# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:11:39 2020

@author: yjy
"""
import warnings
warnings.filterwarnings('ignore')

import torch
import cv2
import numpy as np
import os
import stabnet18
from torch.nn import DataParallel
import torchfcn
import utils

import sys
# sys.path is a list of absolute path strings
#sys.path.append('landmark_detection/')


from collections import deque
import scipy.io as sio
from landmark_detection.setup_3ddfa import dlib_and_3DDFA

input_video='example/1.avi' #input video path
output_video='result/1.avi'
alpha=0.3  #alpha value for MLS warping
nwin=5    #length of sliding window
FEATSIZE=512 #number of feature points per frame
reg=0.1#*torch.ones(1).float().cuda() #regularization on foreground/background

TESTWIDTH=832
TESTHEIGHT=448

################Prepare face detectors###########################
face_regressor,face_detector,model_3ddfa=dlib_and_3DDFA(dlib_landmark_model = 'landmark_detection/shape_predictor_68_face_landmarks.dat',checkpoint_fp = 'landmark_detection/phase1_wpdc_vdc.pth.tar',arch = 'mobilenet_1')
################Load default face (in case that there is no face in the video)##################
default_face=np.load('default_face.npy')
################Prepare foreground segmentation###########################
model=torchfcn.models.FCN8s(n_class=1)
model=model.cuda()
if os.path.isfile('checkpt_fcn.pt'):
    model.load_state_dict(torch.load('checkpt_fcn.pt'))
    print('Loaded foreground segmentation network weights...')
else:
    sys.exit('Foreground segmentation network checkpoint not found! Please put checkpt_fcn.pt in the main folder. Exiting...')
model.eval()

################Prepare stabilization network###########################
model_stab = stabnet18.stabnet(nwin=nwin)
model_stab=model_stab.cuda()
if os.path.isfile('checkpt_stabnet.pt'):
    model_stab.load_state_dict(torch.load('checkpt_stabnet.pt'))
    print('Loaded stabilization network weights...')
else:
    sys.exit('Stabilization network checkpoint not found! Please put checkpt_stabnet.pt in the main folder. Exiting...')
model_stab.eval()




flow_calculator = cv2.cuda_SparsePyrLKOpticalFlow.create((15,15),3,30)
feature_detector = cv2.cuda.createGoodFeaturesToTrackDetector(cv2.CV_8UC1,800,0.01,7,7)
count=0
###################################Determine video size#################################################
cap = cv2.VideoCapture(input_video)
if TESTWIDTH is None:
    WIDTH=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
else:
    WIDTH=TESTWIDTH
if TESTHEIGHT is None:
    HEIGHT=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    HEIGHT=TESTHEIGHT
MARGIN=100
HALFMARGIN=50

################################Prepare torch grid for warping########################################
xv, yv = np.meshgrid(np.linspace(-1,1,WIDTH+MARGIN), np.linspace(-1,1,HEIGHT+MARGIN))
xv=np.expand_dims(xv,axis=2)
yv=np.expand_dims(yv,axis=2)
grid=np.expand_dims(np.concatenate((xv,yv),axis=2),axis=0)
grid_large=np.repeat(grid,1,axis=0)
grid_large=torch.from_numpy(grid_large).float().cuda()
masforwarp=torch.zeros(1,1,HEIGHT+MARGIN,WIDTH+MARGIN).float().cuda()
masforwarp[:,:,HALFMARGIN:HALFMARGIN+HEIGHT,HALFMARGIN:HALFMARGIN+WIDTH]=1

################################Open output video########################################
writer = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (WIDTH+MARGIN,HEIGHT+MARGIN))
writer_mask = cv2.VideoWriter(output_video[:-4]+'_mask.avi',cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (WIDTH+MARGIN,HEIGHT+MARGIN))

points=np.zeros((nwin-1,8,FEATSIZE))
headind=np.random.choice(43867,FEATSIZE)
BOX=None
lasthead=default_face.copy()
lastfeat=None
lastframe_gpu = cv2.cuda_GpuMat()
thisframe_gpu = cv2.cuda_GpuMat()
lastframe_gray=cv2.cuda_GpuMat()
thisframe_gray=cv2.cuda_GpuMat()

video_temp = deque()

N=20
A=torch.linspace(0,HEIGHT+MARGIN-1,N)
B=torch.linspace(0,WIDTH+MARGIN-1,N)
vx,vy=torch.meshgrid(A,B)
reshaped_v = torch.cat((vx.reshape(1,1,N*N),vy.reshape(1,1,N*N)),0).float().cuda()#[2,1,n]
field=np.expand_dims(reshaped_v.squeeze().data.cpu().numpy().reshape(2,N,N),0)#np.zeros((1,2,N,N))
###################################begin#################################################
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        print("Processing frame #"+str(count)+"...")
        frame=cv2.resize(frame,(WIDTH,HEIGHT))
        frame_small=cv2.resize(frame,(832,448))
        #skip possible blank frames
        if np.any(frame)!=True:
            continue
        if count==0:
            #write frame and mask for the first frame
            towarp2=np.zeros((HEIGHT+MARGIN,WIDTH+MARGIN,3))
            towarp2[HALFMARGIN:HALFMARGIN+HEIGHT,HALFMARGIN:HALFMARGIN+WIDTH,:]=frame
            writer.write(towarp2.astype(np.uint8))
            fakemask=np.zeros((HEIGHT+MARGIN,WIDTH+MARGIN,3))
            fakemask[HALFMARGIN:HALFMARGIN+HEIGHT,HALFMARGIN:HALFMARGIN+WIDTH,:]=1
            writer_mask.write(255*fakemask.astype(np.uint8))
            #upload frame to GPU
            lastframe_gpu.upload(frame_small.astype(np.uint8))
            lastframe=frame_small.copy()
            lastframe_gray=cv2.cuda.cvtColor(lastframe_gpu, cv2.COLOR_BGR2GRAY)
            #locate head in the first frame
            _,BOX,status,lasthead=utils.facedetect(frame_small,BOX,lasthead,headind,face_detector,face_regressor,model_3ddfa)
        elif count<=nwin-1:
            #upload frame to GPU
            thisframe_gpu.upload(frame_small.astype(np.uint8))
            thisframe=frame_small
            thisframe_gray=cv2.cuda.cvtColor(thisframe_gpu, cv2.COLOR_BGR2GRAY)
            #detect both head vertices and feature points
            points[count-1,:,:],BOX,status_face,status_feat,lasthead,lastfeat=utils.getpointNhead(lastframe,thisframe,lastframe_gray,thisframe_gray,lasthead,BOX,lastfeat,model,face_detector,face_regressor,model_3ddfa,flow_calculator,feature_detector,headind,FEATSIZE)
            #shift frames
            lastframe_gpu=thisframe_gpu.clone()
            lastframe=thisframe.copy()
            lastframe_gray=thisframe_gray.clone()
            video_temp.append(frame)
        else:
            #prepare input to stabilization network
            feat_reshape=points[:,0:4,:].reshape(points.shape[0]*4,FEATSIZE)
            head_reshape=points[:,4:8,:].reshape(points.shape[0]*4,FEATSIZE)
            feat_reshape=torch.from_numpy(feat_reshape).float().cuda().unsqueeze(0)
            head_reshape=torch.from_numpy(head_reshape).float().cuda().unsqueeze(0)
            #infer the warp point     
            oup=model_stab(feat_reshape,head_reshape,reg)
            oup1=oup.view(1,nwin-2,2,FEATSIZE)
            #this is the frame we want to warp for this window
            towarp=video_temp.popleft()
            
            #adjust the point coordinates,add margin to frame
            temppoint=points[0:2,:,:].copy()
            temppoint[0,2,:]=temppoint[0,2,:]*WIDTH/832
            temppoint[0,3,:]=temppoint[0,3,:]*HEIGHT/448
            temppoint[0,2,:]+=HALFMARGIN
            temppoint[0,3,:]+=HALFMARGIN
            towarp2=np.zeros((HEIGHT+MARGIN,WIDTH+MARGIN,3))
            towarp2[HALFMARGIN:HALFMARGIN+HEIGHT,HALFMARGIN:HALFMARGIN+WIDTH,:]=towarp
            #need to adjust the magnitude of the network output as well
            oup2=oup1[0,0,:,:].clone()
            oup2[0,:]=oup2[0,:]*WIDTH/832
            oup2[1,:]=oup2[1,:]*HEIGHT/448
            
            #warp the frame
            tempframe,M,field=utils.warpframe(torch.from_numpy(towarp2).float().cuda(),temppoint,oup2,field,alpha,WIDTH,HEIGHT,MARGIN,HALFMARGIN,grid_large,masforwarp)
            #update the warped points
            points=utils.updatepoints(points,oup1[0,0,:,:],alpha)
            
            #upload the new frame to GPU
            thisframe_gpu.upload(frame_small.astype(np.uint8))
            thisframe=frame_small.copy()
            thisframe_gray=cv2.cuda.cvtColor(thisframe_gpu, cv2.COLOR_BGR2GRAY)
            #detect head vertices and feature points for the new frame
            newpoint,BOX,status_face,status_feat,lasthead,lastfeat=utils.getpointNhead(lastframe,thisframe,lastframe_gray,thisframe_gray,lasthead,BOX,lastfeat,model,face_detector,face_regressor,model_3ddfa,flow_calculator,feature_detector,headind,FEATSIZE)
            #append the new feature points to the queue
            points=np.concatenate((points[1:,:,:],newpoint),0)
            #shift frame and prepare for next frame
            lastframe_gpu=thisframe_gpu.clone()
            lastframe=thisframe.copy()
            lastframe_gray=thisframe_gray.clone()
            video_temp.append(frame)
            
            #write the result for the warped frame
            writer.write(tempframe.astype(np.uint8))
            writer_mask.write(255*np.tile(np.expand_dims(M,2),(1,1,3)).astype(np.uint8))            
        count=count+1
    else:
        break

#The following code wrap up the last few frames in the video
for i in range(nwin):
    print("Processing frame #"+str(count)+"...")
    feat_reshape=points[:,0:4,:].reshape(points.shape[0]*4,FEATSIZE)
    head_reshape=points[:,4:8,:].reshape(points.shape[0]*4,FEATSIZE)
    feat_reshape=torch.from_numpy(feat_reshape).float().cuda().unsqueeze(0)
    head_reshape=torch.from_numpy(head_reshape).float().cuda().unsqueeze(0)
    oup=model_stab(feat_reshape,head_reshape,reg)
    oup1=oup.view(1,nwin-2,2,FEATSIZE)
    towarp=video_temp.popleft()
    
    temppoint=points[0:2,:,:].copy()
    temppoint[0,2,:]=temppoint[0,2,:]*WIDTH/832
    temppoint[0,3,:]=temppoint[0,3,:]*HEIGHT/448
    temppoint[0,2,:]+=HALFMARGIN
    temppoint[0,3,:]+=HALFMARGIN
    towarp2=np.zeros((HEIGHT+MARGIN,WIDTH+MARGIN,3))
    towarp2[HALFMARGIN:HALFMARGIN+HEIGHT,HALFMARGIN:HALFMARGIN+WIDTH,:]=towarp
    tempframe,M,field=utils.warpframe(torch.from_numpy(towarp2).float().cuda(),temppoint,oup2,field,alpha,WIDTH,HEIGHT,MARGIN,HALFMARGIN,grid_large,masforwarp)
    points=utils.updatepoints(points,oup1[0,0,:,:],alpha)
    thisframe_gpu.upload(cv2.resize(video_temp[-1].copy().astype(np.uint8),(832,448)))
    thisframe=cv2.resize(video_temp[-1].copy(),(832,448))
    thisframe_gray=cv2.cuda.cvtColor(thisframe_gpu, cv2.COLOR_BGR2GRAY)       
    newpoint,BOX,status_face,status_feat,lasthead,lastfeat=utils.getpointNhead(lastframe,thisframe,lastframe_gray,thisframe_gray,lasthead,BOX,lastfeat,model,face_detector,face_regressor,model_3ddfa,flow_calculator,feature_detector,headind,FEATSIZE)
    points=np.concatenate((points[1:,:,:],newpoint),0)
    lastframe_gpu=thisframe_gpu.clone()
    lastframe=thisframe.copy()
    lastframe_gray=thisframe_gray.clone()
    video_temp.append(video_temp[-1].copy())
    writer.write(tempframe.astype(np.uint8))
    writer_mask.write(255*np.tile(np.expand_dims(M,2),(1,1,3)).astype(np.uint8))
    count=count+1

writer.release()
writer_mask.release()
print("Stabilization Finished.")