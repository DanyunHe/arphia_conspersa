


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import rawpy
import imageio

from PIL import Image
import pandas as pd
import os
import sys
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook

from skimage import data, filters, color, morphology,exposure
from skimage.segmentation import flood, flood_fill


#A. Flood fill background to get background mask
def _get_flood_fill_bg_mask(img, type01):
    if type01==0:
      #im0
      img_hsv = color.rgb2hsv(img)
      bgmask = flood(img_hsv[..., 1], (2000, 2000), tolerance=0.06)
      bgmask = bgmask.astype(int)
    else:
      #im1
      img_hsv = color.rgb2hsv(img)
      bgmask = flood(img_hsv[..., 2], (2000, 2000), tolerance=0.2)
      bgmask = bgmask.astype(int)

    return bgmask


#B. SAM to get base forewing and hindwing masks
def _iterative_SAM_procedure(predictor, img, input_point,input_label):
    predictor.set_image(img)
    #get the wing mask
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    #another iteration
    image_copy=img.copy()
    image_copy[masks[0,:,:]==False] = (255, 255, 255)
    predictor.set_image(image_copy)
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    return masks
    
#C. Correct wing segmentation based on background mask
#New masks to find the difference between background masks and wing masks
def _correction_from_flood_fill_bg_mask(fw_bool, hw_bool, bgmask, masks_fw=None, masks_hw=None):
    new_fw_mask=None
    new_hw_mask=None
    
    im_diff_mask=np.copy(bgmask)
    if fw_bool:
        masks_fw[0,:,:][bgmask==1]=False
        im_diff_mask[masks_fw[0,:,:]==True]=2
        new_fw_mask=np.copy(masks_fw[0,:,:])


    if hw_bool:
        masks_hw[0,:,:][bgmask==1]=False
        im_diff_mask[masks_hw[0,:,:]==True]=3
        new_hw_mask=np.copy(masks_hw[0,:,:])

    height,width=im_diff_mask.shape

    r=25

    for i in range(0,width):
        for j in range(0,height):
            if im_diff_mask[j,i]==0:
                ri=5
                Continue=True
                while ri<=r and Continue:
                    jl=max(j-ri,0)
                    jh=min(j+ri,height-1)
                    il=max(i-ri,0)
                    ih=min(i+ri,width-1)
                    ri=ri+5
                    ihh=min(i+50,width-1)
                    ill=max(i-50,0)
                    jhh=min(j+50,height-1)
                    jll=max(j-50,0)
                    if (im_diff_mask[jl,i]==1 and im_diff_mask[jh,i]==2) or (im_diff_mask[jl,i]==2 and im_diff_mask[jh,i]==1):
                        if (im_diff_mask[jll,i]!=0 and im_diff_mask[jhh,i]!=0):
                            Continue=False
                            new_fw_mask[j,i]=1
                    if (im_diff_mask[j,il]==1 and im_diff_mask[j,ih]==2) or (im_diff_mask[j,il]==2 and im_diff_mask[j,ih]==1):
                        if (im_diff_mask[j,ill]!=0 and im_diff_mask[j,ihh]!=0):
                            Continue=False
                            new_fw_mask[j,i]=1

                    if (im_diff_mask[jl,i]==1 and im_diff_mask[jh,i]==3) or (im_diff_mask[jl,i]==3 and im_diff_mask[jh,i]==1):
                        if (im_diff_mask[jll,i]!=0 and im_diff_mask[jhh,i]!=0):
                            Continue=False
                            new_hw_mask[j,i]=1
                    if (im_diff_mask[j,il]==1 and im_diff_mask[j,ih]==3) or (im_diff_mask[j,il]==3 and im_diff_mask[j,ih]==1):
                        if (im_diff_mask[j,ill]!=0 and im_diff_mask[j,ihh]!=0):
                            Continue=False
                            new_hw_mask[j,i]=1
                            
    return new_fw_mask, new_hw_mask
    
    
def _cleanup_masks(fw_bool, hw_bool,new_fw_mask=None, new_hw_mask=None):
    if fw_bool:
        #remove small disconnected components in new wing masks
        new_fw_mask = morphology.remove_small_objects(new_fw_mask, min_size=6000)
        #close small holes in new wing masks
        new_fw_mask = morphology.binary_closing(new_fw_mask, morphology.disk(5))
        
    if hw_bool:
        new_hw_mask = morphology.remove_small_objects(new_hw_mask, min_size=6000)
        new_hw_mask = morphology.binary_closing(new_hw_mask, morphology.disk(5))
    return new_fw_mask, new_hw_mask

#Save full size wing segmentation images
def _save_full_masks(fw_bool, hw_bool,img, out_dir, wing_name, new_fw_mask=None, new_hw_mask=None):
    if fw_bool:
        image_copy=img.copy()
        image_copy[new_fw_mask==False] = (255, 255, 255)
        cv2.imwrite("{}/{}_fw_full.tif".format(out_dir, wing_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))

    if hw_bool:
        image_copy=img.copy()
        image_copy[new_hw_mask==False] = (255, 255, 255)
        cv2.imwrite("{}/{}_hw_full.tif".format(out_dir, wing_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
        
    
def extract_wing_im(predictor, img, type01, w_name, bg_x, bg_y, hw_x0,hw_y0,fw_x0,fw_y0,hw_x1,hw_y1,fw_x1,fw_y1,body_x0,body_y0,body_x1,body_y1,body_x2,body_y2, sticky, out_dir):
    #A. Flood fill background to get background mask
    bgmask = _get_flood_fill_bg_mask(img, type01)
    
    #B. SAM to get base forewing and hindwing masks
    if hw_x0!=-1 and fw_x0!=-1:
        input_point = np.array([[bg_x,bg_y], [hw_x0,hw_y0],[hw_x1,hw_y1], [fw_x0, fw_y0],[fw_x1, fw_y1], [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
        fw_input_label = np.array([0, 0, 0, 1, 1, 0,0,0])
        hw_input_label = np.array([0, 1, 1, 0, 0, 0,0,0])
      
        #get the fw mask: assume true
        masks_fw=_iterative_SAM_procedure(predictor, img, input_point, fw_input_label)
      
        #get the hw mask: assume true
        masks_hw=_iterative_SAM_procedure(predictor, img, input_point, hw_input_label)
        
        #C. Correct wing segmentation based on background mask
        #New masks to find the difference between background masks and wing masks
        new_fw_mask, new_hw_mask = _correction_from_flood_fill_bg_mask(fw_bool=True, hw_bool=True, bgmask=bgmask, masks_fw=masks_fw, masks_hw=masks_hw)
        
        #remove small disconnected components in new wing masks
        #close small holes in new wing masks
        new_fw_mask, new_hw_mask = _cleanup_masks(fw_bool=True, hw_bool=True,new_fw_mask=new_fw_mask, new_hw_mask=new_hw_mask)
        
        #Save full size wing segmentation images
        if sticky==1:
            _save_full_masks(fw_bool=True, hw_bool=True, img=img, out_dir=out_dir+"/sticky_full", wing_name=w_name, new_fw_mask=new_fw_mask, new_hw_mask=new_hw_mask)
        
        else:
            _save_full_masks(fw_bool=True, hw_bool=True, img=img, out_dir=out_dir+"/perfect_full", wing_name=w_name, new_fw_mask=new_fw_mask, new_hw_mask=new_hw_mask)
      
      
    elif hw_x0!=-1 and fw_x0==-1:
        input_point = np.array([[bg_x,bg_y], [hw_x0,hw_y0], [hw_x1,hw_y1],  [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
        hw_input_label = np.array([0, 1, 1, 0,0,0])
      
        #get the hw mask
        masks_hw=_iterative_SAM_procedure(predictor, img, input_point, hw_input_label)
       
        
        #C. Correct wing segmentation based on background mask
        #New masks to find the difference between background masks and wing masks
        _, new_hw_mask = _correction_from_flood_fill_bg_mask(fw_bool=False, hw_bool=True, bgmask=bgmask, masks_hw=masks_hw)
        
        #remove small disconnected components in new wing masks
        #close small holes in new wing masks
        _, new_hw_mask = _cleanup_masks(fw_bool=False, hw_bool=True, new_hw_mask=new_hw_mask)
      
        #Save full size wing segmentation images
        _save_full_masks(fw_bool=False, hw_bool=True, img=img, out_dir=out_dir+"/missing_full", wing_name=w_name, new_hw_mask=new_hw_mask)
        
    
    elif hw_x0==-1 and fw_x0!=-1:
        input_point = np.array([[bg_x,bg_y], [fw_x0,fw_y0],[fw_x1,fw_y1], [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
        fw_input_label = np.array([0, 1, 1, 0,0,0])
      
        #get the fw mask: assume true
        masks_fw = _iterative_SAM_procedure(predictor, img, input_point, fw_input_label)
      
        #C. Correct wing segmentation based on background mask
        #New masks to find the difference between background masks and wing masks
        new_fw_mask,_ = _correction_from_flood_fill_bg_mask(fw_bool=True, hw_bool=False, bgmask=bgmask, masks_fw=masks_fw)
      
        #remove small disconnected components in new wing masks
        #close small holes in new wing masks
        new_fw_mask,_ = _cleanup_masks(fw_bool=True, hw_bool=False,new_fw_mask=new_fw_mask)
      
      
        #Save full size wing segmentation images
        _save_full_masks(fw_bool=True, hw_bool=False, img=img, out_dir=out_dir+"/missing_full", wing_name=w_name, new_fw_mask=new_fw_mask)




###RUN
for wi in range(32,N_wing+1):

  w_name_base=population_excel["bodyparts"][wi][:-6]
  w_name_0=w_name_base+"_0"
  w_name_1=w_name_base+"_1"

  print("{}/{}: {}\n".format(wi, N_wing,w_name_0+".dng"))

  with rawpy.imread(file_path_img+w_name_0+".dng") as raw:
      im_0 = raw.postprocess()
  with rawpy.imread(file_path_img+w_name_1+".dng") as raw:
      im_1 = raw.postprocess()

  bg_x=float(population_excel["background"][wi])
  bg_y=float(population_excel["background.1"][wi])

  hw_x0=float(population_excel["hind1"][wi])
  hw_y0=float(population_excel["hind1.1"][wi])
  fw_x0=float(population_excel["fore1"][wi])
  fw_y0=float(population_excel["fore1.1"][wi])

  hw_x1=float(population_excel["hind2"][wi])
  hw_y1=float(population_excel["hind2.1"][wi])
  fw_x1=float(population_excel["fore2"][wi])
  fw_y1=float(population_excel["fore2.1"][wi])


  body_x0=float(population_excel["bodypart1"][wi])
  body_y0=float(population_excel["bodypart1.1"][wi])
  body_x1=float(population_excel["bodypart2"][wi])
  body_y1=float(population_excel["bodypart2.1"][wi])
  body_x2=float(population_excel["bodypart3"][wi])
  body_y2=float(population_excel["bodypart3.1"][wi])

  sticky=0 #population_excel["sticky?"][wi]

  #ball_x=population_excel["ball, x"][wi]
  #ball_y=population_excel["ball, y"][wi]

  extract_wing_im(predictor, im_0, 0, w_name_0, bg_x, bg_y, hw_x0,hw_y0,fw_x0,fw_y0,hw_x1,hw_y1,fw_x1,fw_y1,body_x0,body_y0,body_x1,body_y1,body_x2,body_y2, sticky)
  extract_wing_im(predictor, im_1, 1, w_name_1, bg_x, bg_y, hw_x0,hw_y0,fw_x0,fw_y0,hw_x1,hw_y1,fw_x1,fw_y1,body_x0,body_y0,body_x1,body_y1,body_x2,body_y2, sticky)









