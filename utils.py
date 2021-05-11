#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:24:51 2021

@author: yjy
"""
import torch
import numpy as np
from landmark_detection.utils.inference import (
    parse_roi_box_from_landmark,
    crop_img,
    predict_68pts,
    predict_dense,
)
import torchvision.transforms as transforms
from landmark_detection.utils.ddfa import ToTensorGjz, NormalizeGjz
import cv2
import torch.nn.functional as F

transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])


def updatepoints(points,oup,alpha):
    feat_src_old=torch.from_numpy(points[0,2:4,:]).float().cuda()#2,1024
    face_src_old=torch.from_numpy(points[0,6:8,:]).float().cuda()#2,1024
    feat_tar_old=torch.from_numpy(points[0,0:2,:]).float().cuda()#2,1024
    feat_tar_new=torch.from_numpy(points[1,0:2,:]).float().cuda()#2,1024
    face_tar_new=torch.from_numpy(points[1,4:6,:]).float().cuda()#2,1024
    
    control=feat_src_old
    warped_tar_new=warp_point_torch(control,control+oup,feat_tar_new,alpha)
    temp=warped_tar_new.data.cpu().numpy()
    points[1,0:2,:]=temp
    
    warped_face_tar_new=warp_point_torch(control,control+oup,face_tar_new,alpha)
    temp=warped_face_tar_new.data.cpu().numpy()
    points[1,4:6,:]=temp
    
    if np.any(np.isnan(points[1,:,:])):
        print('Find invalid numbers in warped points, something is wrong...')
    
    return points

def warp_point_torch(control_src,control_tar,point,alpha):
    p=control_src.permute(1,0)            #[m,2]
    q=control_tar.permute(1,0)            #[m,2]
    
    reshaped_p =p.unsqueeze(2).unsqueeze(3)#[m,2,1,1]
    reshaped_v=point.unsqueeze(1)+0.001#[2,1,n]
    
    w = 1.0 / torch.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha#[m,1,n]
    sum_w = torch.sum(w,0)#[1,n]
    pstar = torch.sum(w * reshaped_p.permute(1, 0, 2, 3), axis=1) / sum_w#[2,1,n]
    phat = reshaped_p - pstar#[m,2,1,n]
    reshaped_phat = phat.unsqueeze(1)#[m,1,2,1,n]
    reshaped_w = w.unsqueeze(1).unsqueeze(1)#[m,1,1,1,n]
    neg_phat_verti = torch.zeros_like(phat).float().cuda()#[m,2,1,n]
    neg_phat_verti[:,0,:,:]=phat[:,1,:,:]
    neg_phat_verti[:,1,:,:]=-phat[:,0,:,:]
    reshaped_neg_phat_verti = neg_phat_verti.unsqueeze(1)#[m,1,2,1,n]
    mul_left = torch.cat((reshaped_phat, reshaped_neg_phat_verti), axis=1)#[m,2,2,1,n]
    vpstar = reshaped_v - pstar#[2,1,n]
    reshaped_vpstar = vpstar.unsqueeze(1)#[2,1,1,n]
    neg_vpstar_verti=torch.zeros_like(vpstar).float().cuda()#[2,1,n]
    neg_vpstar_verti[0,:,:]=vpstar[1,:,:]
    neg_vpstar_verti[1,:,:]=-vpstar[0,:,:]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.unsqueeze(1)#[2,1,1,n]
    mul_right = torch.cat((reshaped_vpstar, reshaped_neg_vpstar_verti), 1)#[2,2,1,n]
    reshaped_mul_right = mul_right.unsqueeze(0)#[1,2,2,1,n]
    A = torch.matmul((reshaped_w * mul_left).permute(0, 3, 4, 1, 2), #[m,1,n,2,2]
               reshaped_mul_right.permute(0, 3, 4, 1, 2))#[1,1,n,2,2]
    reshaped_q = q.unsqueeze(2).unsqueeze(3)#[m,2,1,1]
    qstar = torch.sum(w * reshaped_q.permute(1, 0, 2, 3), 1) / torch.sum(w, 0)#[2,1,n]
    qhat = reshaped_q - qstar#[m,2,1,n]
    reshaped_qhat = qhat.unsqueeze(1).permute(0, 3, 4, 1, 2)#[m,1,n,1,2]
    temp = torch.sum(torch.matmul(reshaped_qhat, A), 0).permute(2, 3, 0, 1)#[1,2,1,n]
    reshaped_temp = temp.squeeze(0)#[2,1,n]
    norm_reshaped_temp = torch.norm(reshaped_temp,p=2, dim=0, keepdim=True)#[1,1,n]
    norm_vpstar = torch.norm(vpstar, p=2,dim=0, keepdim=True)#[1,1,n]
    transformers = reshaped_temp / (norm_reshaped_temp+0.00001) * norm_vpstar  + qstar
    transformers1=transformers.squeeze()
    return transformers1

def mls_torch8(control_src,control_tar,image,field,alpha,WIDTH,HEIGHT,MARGIN,HALFMARGIN,grid_large,mas):
    
    N=field.shape[2]
    p=control_src.permute(1,0).flip(1)            #[m,2]
    q=control_tar.permute(1,0).flip(1)           #[m,2]
    
    reshaped_p =p.unsqueeze(2).unsqueeze(3)#[m,2,1,1]
    # Make grids on the original image
    A=torch.linspace(0,HEIGHT+MARGIN-1,N)
    B=torch.linspace(0,WIDTH+MARGIN-1,N)
    vx,vy=torch.meshgrid(A,B)
    reshaped_v = torch.cat((vx.reshape(1,1,N*N),vy.reshape(1,1,N*N)),0).float().cuda()#[2,1,n]
    
    w = 1.0 / torch.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha#[m,1,n]
    sum_w = torch.sum(w,0)#[1,n]
    pstar = torch.sum(w * reshaped_p.permute(1, 0, 2, 3), axis=1) / sum_w#[2,1,n]
    phat = reshaped_p - pstar#[m,2,1,n]
    reshaped_phat = phat.unsqueeze(1)#[m,1,2,1,n]
    reshaped_w = w.unsqueeze(1).unsqueeze(1)#[m,1,1,1,n]
    neg_phat_verti = torch.zeros_like(phat).float().cuda()#[m,2,1,n]
    neg_phat_verti[:,0,:,:]=phat[:,1,:,:]
    neg_phat_verti[:,1,:,:]=-phat[:,0,:,:]
    reshaped_neg_phat_verti = neg_phat_verti.unsqueeze(1)#[m,1,2,1,n]
    mul_left = torch.cat((reshaped_phat, reshaped_neg_phat_verti), axis=1)#[m,2,2,1,n]
    vpstar = reshaped_v - pstar#[2,1,n]
    reshaped_vpstar = vpstar.unsqueeze(1)#[2,1,1,n]
    neg_vpstar_verti=torch.zeros_like(vpstar).float().cuda()#[2,1,n]
    neg_vpstar_verti[0,:,:]=vpstar[1,:,:]
    neg_vpstar_verti[1,:,:]=-vpstar[0,:,:]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.unsqueeze(1)#[2,1,1,n]
    mul_right = torch.cat((reshaped_vpstar, reshaped_neg_vpstar_verti), 1)#[2,2,1,n]
    reshaped_mul_right = mul_right.unsqueeze(0)#[1,2,2,1,n]
    A = torch.matmul((reshaped_w * mul_left).permute(0, 3, 4, 1, 2), #[m,1,n,2,2]
               reshaped_mul_right.permute(0, 3, 4, 1, 2))#[1,1,n,2,2]
    reshaped_q = q.unsqueeze(2).unsqueeze(3)#[m,2,1,1]
    qstar = torch.sum(w * reshaped_q.permute(1, 0, 2, 3), 1) / torch.sum(w, 0)#[2,1,n]
    qhat = reshaped_q - qstar#[m,2,1,n]
    reshaped_qhat = qhat.unsqueeze(1).permute(0, 3, 4, 1, 2)#[m,1,n,1,2]
    temp = torch.sum(torch.matmul(reshaped_qhat, A), 0).permute(2, 3, 0, 1)#[1,2,1,n]
    reshaped_temp = temp.squeeze(0)#[2,1,n]
    norm_reshaped_temp = torch.norm(reshaped_temp,p=2, dim=0, keepdim=True)#[1,1,n]
    norm_vpstar = torch.norm(vpstar, p=2,dim=0, keepdim=True)#[1,1,n]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar  + qstar
    transformers=transformers.squeeze()#[2,n]
        
    target=transformers
    
    
    
    target=reshaped_v.squeeze().data.cpu().numpy().reshape(2,N,N)
    source=transformers.data.cpu().numpy().reshape(2,N,N)
    
    if field.shape[0]<10:
        field=np.append(field,np.expand_dims(source,0),axis=0)
    else:
        field=np.roll(field,-1,axis=0)
        field[-1,:,:,:]=source
    source-=np.mean(field,axis=0)
    source+=target
    
    
    towarp=image.permute(2,0,1).unsqueeze(0)
    towarp=torch.cat((towarp,mas),1)

    
    shift1=torch.from_numpy(source-target).unsqueeze(0).cuda()#1,2,N,N
    shift=torch.zeros_like(shift1).float().cuda()
    shift[:,1,:,:]=shift1[:,0,:,:]/image.shape[0]*2
    shift[:,0,:,:]=shift1[:,1,:,:]/image.shape[1]*2
    shiftmap=F.grid_sample(shift,grid_large)

    result=F.grid_sample(towarp,grid_large+shiftmap.permute(0,2,3,1))
    
    
    return result[0,0:3,:,:].permute(1,2,0).data.cpu().numpy(),result[0,3,:,:].data.cpu().numpy(),field

def warpframe(frame,points,oup,field,alpha,WIDTH,HEIGHT,MARGIN,HALFMARGIN,grid_large,masforwarp):   
    
    control=torch.from_numpy(points[0,2:4,:]).float().cuda()
    tempframe,M,field=mls_torch8(control+oup.data,control,frame,field,alpha,WIDTH,HEIGHT,MARGIN,HALFMARGIN,grid_large,masforwarp)
    return tempframe,M,field

def facedetect(frame,inpbox,lasthead,headind,face_detector,face_regressor,model_3ddfa):
    if inpbox is None or np.any(np.isnan(inpbox)) or inpbox[2]-inpbox[0]<150:
        last_frame_pts = []
        rects = face_detector(frame, 1)
        if rects is None or len(rects)<1:
            vertices=np.transpose(lasthead[0:2,:])
            ret=np.transpose(vertices[headind])
            box=inpbox
            return ret,box,1,np.transpose(vertices)
        else:
            rect=rects[0]
        pts = face_regressor(frame,rect ).parts()#
        if len(pts)==0:
            vertices=np.transpose(lasthead[0:2,:])
            ret=np.transpose(vertices[headind])
            box=inpbox
            return ret,box,1,np.transpose(vertices)
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        last_frame_pts.append(pts)
        box = parse_roi_box_from_landmark(last_frame_pts[0])
    else:
        box=inpbox
    
    img = crop_img(frame, box)
    img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    with torch.no_grad():
        input = input.cuda()
        param = model_3ddfa(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
    vertices = predict_dense(param, box)
    
    if np.mean(vertices[0:2,:]-lasthead[0:2,:])>30:
        vertices=np.transpose(lasthead[0:2,:])
        ret=np.transpose(vertices[headind])
        box=inpbox
        return ret,box,1,np.transpose(vertices)
    
    vertices=np.transpose(vertices[0:2,:])
    ret=np.transpose(vertices[headind])
    
    box=parse_roi_box_from_landmark(ret)
    
    return ret,box,0,np.transpose(vertices)

def getfacemask(frame,head,model):
    with torch.no_grad():
        frame=frame.astype(np.float32)
        img1=torch.from_numpy(np.transpose(frame,(2,0,1))).unsqueeze(0).data.cuda()/255
        oup=model(img1.data)
        mask1=oup[0,0,:,:]<0.05
        left=np.max([int(np.min(head[0,:])-20),0])
        right=np.min([int(np.max(head[0,:])+20),832])
        top=np.max([int(np.min(head[1,:])-20),0])
        bottom=np.min([int(np.max(head[1,:])+20),448])
        mask1[top:bottom,left:right]=0
    return mask1

prevPoints=cv2.cuda_GpuMat()
nextPoints=cv2.cuda_GpuMat()
def getfeature(frame1_gpu,frame2_gpu,frame1,frame2,mask,lastfeat,flow_calculator,feature_detector,FEATSIZE):
    status=0
    global prevPoints,nextPoints
    if lastfeat is None or lastfeat.shape[0]<30:
        prevPoints=feature_detector.detect(frame1_gpu)
    else:
        prevPoints.upload(np.expand_dims(lastfeat,0))
    flow=flow_calculator.calc(frame1_gpu, frame2_gpu, prevPoints,nextPoints)
    nextPoints=flow[0]
    valid=flow[1].download().squeeze()
    err=flow[2].download().squeeze()
    p0=prevPoints.download()[0,:,:]
    p1=nextPoints.download()[0,:,:]
    p0=p0[(valid>0)*(err<10)]
    p1=p1[(valid>0)*(err<10)]
    lastfeat=p1.copy()
    status=p0.shape[0]
    
    
    if len(p0)<30:
        status=-1
        p0=np.concatenate((np.random.randint(0,832,(1000,1)),np.random.randint(0,448,(1000,1))),1)
        p1=p0.copy()

    good_new=p1.squeeze()
    good_old=p0.squeeze()
    ind=(good_new[:,0]>=30)*(good_new[:,0]<832-30)*(good_new[:,1]>=30)*(good_new[:,1]<448-30)
    good_new=good_new[ind,:]
    good_old=good_old[ind,:]
    temp=mask[(good_new[:,1]*448/448).astype(np.int64),(good_new[:,0]*832/832).astype(np.int64)].cpu().numpy()
    good_new=good_new[temp==True,:]
    good_old=good_old[temp==True,:]
    
    ind=np.random.choice(good_new.shape[0],FEATSIZE)
    good_new=good_new[ind,:]
    good_old=good_old[ind,:]
    
    return np.transpose(good_old),np.transpose(good_new),status,lastfeat
    
    
def getpointNhead(frame1,frame2,frame1_gpu,frame2_gpu,frame1head,BOX,lastfeat,model,face_detector,face_regressor,model_3ddfa,flow_calculator,feature_detector,headind,FEATSIZE):
    result=np.zeros((1,8,FEATSIZE))
    result[0,4:6,:]=np.transpose(np.transpose(frame1head[0:2,:])[headind])
    result[0,6:8,:],BOX,status_face,frame1head=facedetect(frame2,BOX,frame1head,headind,face_detector,face_regressor,model_3ddfa)
    mask=getfacemask(frame1,frame1head,model)
    
    result[0,0:2,:],result[0,2:4,:],status_feat,lastfeat=getfeature(frame1_gpu,frame2_gpu,frame1,frame2,mask,lastfeat,flow_calculator,feature_detector,FEATSIZE)

    return result,BOX,status_face,status_feat,frame1head,lastfeat