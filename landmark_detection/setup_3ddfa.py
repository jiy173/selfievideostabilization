#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:09:08 2021

@author: yjy
"""
import torch
import dlib
from landmark_detection import mobilenet_v1


def dlib_and_3DDFA(dlib_landmark_model = 'shape_predictor_68_face_landmarks.dat',checkpoint_fp = 'phase1_wpdc_vdc.pth.tar',arch = 'mobilenet_1'):
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()
    
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model_3ddfa = getattr(mobilenet_v1, arch)(
        num_classes=62
    )  # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict = model_3ddfa.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model_3ddfa.load_state_dict(model_dict)
    model_3ddfa = model_3ddfa.cuda()
    model_3ddfa.eval()
    
    return face_regressor,face_detector,model_3ddfa