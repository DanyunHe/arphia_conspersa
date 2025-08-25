# To extract forewings and hindwings from the original images, we have the following steps:
# 1. Segment Anything (SAM) to obtain forewing and hindwing from original images.
# 2. Use flood fill algorithm on the background, to obtain accurate wing edges.
# 3. Use information in 2 to correct wing edges in 1.
# 4. Remove loose parts and close small holes in the wing segmentations.
# 5. Save full size wing segmentation images.
# 6. Crop wing segmentation images and save.

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




population_id=44
file_path_base="/content/drive/My Drive/wing/"
file_path_main=file_path_base+"wing_new_populations/"
file_path_img =file_path_main +"population_{}_original/".format(population_id)
file_path_save =file_path_main+"images_masks_population_{}/".format(population_id)
#population_excel = pd.read_excel(file_path_base+"extract_wings_populations.xlsx", sheet_name="population_{}".format(population_id))
population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
N_wing=len(population_excel["bodyparts"][1:])


if not os.path.exists(file_path_save):
   os.makedirs(file_path_save)
if not os.path.exists(file_path_save+"problematic_full/"):
   os.makedirs(file_path_save+"problematic_full/")
#if not os.path.exists(file_path_save+"sticky_full/"):
#   os.makedirs(file_path_save+"sticky_full/")
#if not os.path.exists(file_path_save+"sticky_cropped/"):
#   os.makedirs(file_path_save+"sticky_cropped/")
if not os.path.exists(file_path_save+"perfect_full/"):
   os.makedirs(file_path_save+"perfect_full/")
if not os.path.exists(file_path_save+"perfect_cropped/"):
   os.makedirs(file_path_save+"perfect_cropped/")
#if not os.path.exists(file_path_save+"missing_full/"):
#   os.makedirs(file_path_save+"missing_full/")
#if not os.path.exists(file_path_save+"missing_cropped/"):
#   os.makedirs(file_path_save+"missing_cropped/")




def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = file_path_base+"fine_tuned_sam_im1b.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)



#TESTING
###Plotting test
wi=20
wi=wi+1
#w_name_base=population_excel["insect id"][wi]
w_name_base=population_excel["bodyparts"][wi][:-6]
w_name_0=w_name_base+"_0"
w_name_1=w_name_base+"_1"
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



plt.imshow(im_1)
plt.plot(bg_x,bg_y, '*', color='red', markersize=5)
plt.plot(bg_x,bg_y, '*', color='red', markersize=5)
plt.plot(hw_x0,hw_y0, '*', color='green', markersize=5)
plt.plot(hw_x1,hw_y1, '*', color='green', markersize=5)
plt.plot(fw_x0,fw_y0, '*', color='blue', markersize=5)
plt.plot(fw_x1,fw_y1, '*', color='blue', markersize=5)
plt.plot(body_x0,body_y0, '*', color='yellow', markersize=5)
plt.plot(body_x1,body_y1, '*', color='yellow', markersize=5)
plt.plot(body_x2,body_y2, '*', color='yellow', markersize=5)
plt.show()



#im0
img_hsv = color.rgb2hsv(im_1)
bgmask = flood(img_hsv[..., 2], (2000, 2000), tolerance=0.2)
bgmask = bgmask.astype(int)




plt.imshow(im_1)
show_mask(bgmask, plt.gca())



image_copy=im_1.copy()
image_copy[bgmask[:,:]==True] = (255, 255, 255)

plt.imshow(im_1[2000:3000,3400:4400])







checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)




predictor.set_image(im_1)
input_point = np.array([[bg_x,bg_y], [hw_x0,hw_y0],[hw_x1,hw_y1], [fw_x0, fw_y0],[fw_x1, fw_y1], [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
fw_input_label = np.array([0, 0, 0, 1, 1, 0,0,0])
hw_input_label = np.array([0, 1, 1, 0, 0, 0,0,0])

masks_hw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=hw_input_label,
        multimask_output=False,
    )



plt.imshow(im_1)
show_mask(masks_hw, plt.gca())


plt.imshow(im_1[2200:3200,3400:4400])
show_mask(masks_hw[0][2200:3200,3400:4400], plt.gca())








#######PROCESS WING
def extract_wing_im(predictor, img, type01, w_name, bg_x, bg_y, hw_x0,hw_y0,fw_x0,fw_y0,hw_x1,hw_y1,fw_x1,fw_y1,body_x0,body_y0,body_x1,body_y1,body_x2,body_y2, sticky):
  #A. Flood fill background to get background mask
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


  #B. SAM to get base forewing and hindwing masks
  predictor.set_image(img)

  if hw_x0!=-1 and fw_x0!=-1:
    input_point = np.array([[bg_x,bg_y], [hw_x0,hw_y0],[hw_x1,hw_y1], [fw_x0, fw_y0],[fw_x1, fw_y1], [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
    fw_input_label = np.array([0, 0, 0, 1, 1, 0,0,0])
    hw_input_label = np.array([0, 1, 1, 0, 0, 0,0,0])

    #input_point = np.array([[bg_x,bg_y], [hw_x0,hw_y0], [fw_x0, fw_y0], [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
    #fw_input_label = np.array([0, 0, 1, 0,0,0])
    #hw_input_label = np.array([0, 1, 0, 0,0,0])

    #get the fw mask: assume true
    masks_fw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=fw_input_label,
        multimask_output=False,
    )
    #another iteration
    image_copy=img.copy()
    image_copy[masks_fw[0,:,:]==False] = (255, 255, 255)
    predictor.set_image(image_copy)
    masks_fw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=fw_input_label,
        multimask_output=False,
    )

    #get the hw mask: assume true
    masks_hw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=hw_input_label,
        multimask_output=False,
    )
    #another iteration
    image_copy=img.copy()
    image_copy[masks_hw[0,:,:]==False] = (255, 255, 255)
    predictor.set_image(image_copy)
    masks_hw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=hw_input_label,
        multimask_output=False,
    )

    #C. Correct wing segmentation based on background mask
    #New masks to find the difference between background masks and wing masks
    masks_fw[0,:,:][bgmask==1]=False
    masks_hw[0,:,:][bgmask==1]=False

    im_diff_mask=np.copy(bgmask)
    im_diff_mask[masks_fw[0,:,:]==True]=2
    im_diff_mask[masks_hw[0,:,:]==True]=3

    new_fw_mask=np.copy(masks_fw[0,:,:])
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


    #remove small disconnected components in new wing masks
    new_fw_mask = morphology.remove_small_objects(new_fw_mask, min_size=6000)
    new_hw_mask = morphology.remove_small_objects(new_hw_mask, min_size=6000)
    #close small holes in new wing masks
    new_fw_mask = morphology.binary_closing(new_fw_mask, morphology.disk(5))
    new_hw_mask = morphology.binary_closing(new_hw_mask, morphology.disk(5))

    #Save full size wing segmentation images
    if sticky==1:
      image_copy=img.copy()
      image_copy[new_fw_mask==False] = (255, 255, 255)
      cv2.imwrite("{}/sticky_full/{}_fw_full.tif".format(file_path_save,w_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
      #np.savetxt('{}/sticky/{}_fw_mask.txt'.format(file_path_save,w_name), masks_fw[0,:,:])

      image_copy=img.copy()
      image_copy[new_hw_mask==False] = (255, 255, 255)
      cv2.imwrite("{}/sticky_full/{}_hw_full.tif".format(file_path_save,w_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
      #np.savetxt('{}/sticky/{}_hw_mask.txt'.format(file_path_save,w_name), masks_hw[0,:,:])
    else:
      image_copy=img.copy()
      image_copy[new_fw_mask==False] = (255, 255, 255)
      cv2.imwrite("{}/perfect_full/{}_fw_full.tif".format(file_path_save,w_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
      #np.savetxt('{}/perfect/{}_fw_mask.txt'.format(file_path_save,w_name), masks_fw[0,:,:])

      image_copy=img.copy()
      image_copy[new_hw_mask==False] = (255, 255, 255)
      cv2.imwrite("{}/perfect_full/{}_hw_full.tif".format(file_path_save,w_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
      #np.savetxt('{}/perfect/{}_hw_mask.txt'.format(file_path_save,w_name), masks_hw[0,:,:])

  elif hw_x0!=-1 and fw_x0==-1:
    input_point = np.array([[bg_x,bg_y], [hw_x0,hw_y0], [hw_x1,hw_y1],  [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
    hw_input_label = np.array([0, 1, 1, 0,0,0])

    #get the hw mask
    masks_hw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=hw_input_label,
        multimask_output=False,
        )
    #another iteration
    image_copy=img.copy()
    image_copy[masks_hw[0,:,:]==False] = (255, 255, 255)
    predictor.set_image(image_copy)
    masks_hw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=hw_input_label,
        multimask_output=False,
    )

    #C. Correct wing segmentation based on background mask
    #New masks to find the difference between background masks and wing masks
    masks_hw[0,:,:][bgmask==1]=False

    im_diff_mask=np.copy(bgmask)
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

                    if (im_diff_mask[jl,i]==1 and im_diff_mask[jh,i]==3) or (im_diff_mask[jl,i]==3 and im_diff_mask[jh,i]==1):
                        if (im_diff_mask[jll,i]!=0 and im_diff_mask[jhh,i]!=0):
                            Continue=False
                            new_hw_mask[j,i]=1
                    if (im_diff_mask[j,il]==1 and im_diff_mask[j,ih]==3) or (im_diff_mask[j,il]==3 and im_diff_mask[j,ih]==1):
                        if (im_diff_mask[j,ill]!=0 and im_diff_mask[j,ihh]!=0):
                            Continue=False
                            new_hw_mask[j,i]=1

    #remove small disconnected components in new wing masks
    new_hw_mask = morphology.remove_small_objects(new_hw_mask, min_size=6000)
    #close small holes in new wing masks
    new_hw_mask = morphology.binary_closing(new_hw_mask, morphology.disk(5))

    #Save full size wing segmentation images
    image_copy=img.copy()
    image_copy[new_hw_mask==False] = (255, 255, 255)
    cv2.imwrite("{}/missing_full/{}_hw_full.tif".format(file_path_save,w_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
    #np.savetxt('{}/missing/{}_hw_mask.txt'.format(file_path_save,w_name), masks_hw[0,:,:])

  elif hw_x0==-1 and fw_x0!=-1:
    input_point = np.array([[bg_x,bg_y], [fw_x0,fw_y0],[fw_x1,fw_y1], [body_x0, body_y0],[body_x1, body_y1],[body_x2, body_y2]])
    fw_input_label = np.array([0, 1, 1, 0,0,0])

    #get the hw mask
    masks_fw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=fw_input_label,
        multimask_output=False,
        )
    #another iteration
    image_copy=img.copy()
    image_copy[masks_fw[0,:,:]==False] = (255, 255, 255)
    predictor.set_image(image_copy)
    masks_fw, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=fw_input_label,
        multimask_output=False,
    )

    #C. Correct wing segmentation based on background mask
    #New masks to find the difference between background masks and wing masks
    masks_fw[0,:,:][bgmask==1]=False

    im_diff_mask=np.copy(bgmask)
    im_diff_mask[masks_fw[0,:,:]==True]=2

    new_fw_mask=np.copy(masks_fw[0,:,:])

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


    #remove small disconnected components in new wing masks
    new_fw_mask = morphology.remove_small_objects(new_fw_mask, min_size=6000)
    #close small holes in new wing masks
    new_fw_mask = morphology.binary_closing(new_fw_mask, morphology.disk(5))

    #Save full size wing segmentation images
    image_copy=img.copy()
    image_copy[new_fw_mask==False] = (255, 255, 255)
    cv2.imwrite("{}/missing_full/{}_fw_full.tif".format(file_path_save,w_name), cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
    #np.savetxt('{}/missing/{}_fw_mask.txt'.format(file_path_save,w_name), masks_fw[0,:,:])



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


####Crop segmented wing images and save cropped versions


def crop_wing_func(wing_0_im,wing_1_im, file_save_crop,sorted_file_list_full_0, sorted_file_list_full_1):
    w_0=cv2.cvtColor(wing_0_im, cv2.COLOR_BGR2GRAY)
    w_1=cv2.cvtColor(wing_1_im, cv2.COLOR_BGR2GRAY)

    nx0=len(w_0[1,:])
    ny0=len(w_0[:,1])
    nx1=len(w_1[1,:])
    ny1=len(w_1[:,1])

    min_x_w=0
    continue_search=True
    ii=0
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[:,ii]] else True
        if all_white==False:
            continue_search=False
            min_x_w=ii
        ii=ii+1

    max_x_w=nx0-1
    continue_search=True
    ii=nx0-1
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[:,ii]] else True
        if all_white==False:
            continue_search=False
            max_x_w=ii
        ii=ii-1


    min_y_w=0
    continue_search=True
    jj=0
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[jj,:]] else True
        if all_white==False:
            continue_search=False
            min_y_w=jj
        jj=jj+1

    max_y_w=ny0-1
    continue_search=True
    jj=ny0-1
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[jj,:]] else True
        if all_white==False:
            continue_search=False
            max_y_w=jj
        jj=jj-1


    #crop wing, with a margin of +/- 50 pixels if possible
    if max_x_w>=min(nx0,nx1):
        max_x_w=min(nx0,nx1)
    if max_y_w>=min(ny0,ny1):
        max_y_w=min(ny0,ny1)

    if min_x_w-50>=0:
        min_x_w=min_x_w-50
    if min_y_w-50>=0:
        min_y_w=min_y_w-50
    if max_x_w+50<min(nx0,nx1):
        max_x_w=max_x_w+50
    if max_y_w+50<min(ny0,ny1):
        max_y_w=max_y_w+50

    #img[y:y+h, x:x+w]
    w_0_im_crop=wing_0_im[min_y_w:max_y_w, min_x_w:max_x_w]
    w_1_im_crop=wing_1_im[min_y_w:max_y_w, min_x_w:max_x_w]

    #save cropped wing
    cv2.imwrite(file_save_crop+os.path.splitext(sorted_file_list_full_0)[0][:-5]+"_crop.tif", w_0_im_crop)
    cv2.imwrite(file_save_crop+os.path.splitext(sorted_file_list_full_1)[0][:-5]+"_crop.tif", w_1_im_crop)


    #RECORD
    wing_category=os.path.splitext(sorted_file_list_full_0)[0][-7:-9]
    #record in excel: original dimensions, crop locations, after crop dimensions
    append_data = pd.DataFrame([{'filename':os.path.splitext(sorted_file_list_full_0)[0][:-5],
                                 'w0':nx0,
                                 'h0':ny0,
                                 'w1':nx1,
                                 'h1':ny1,

                                 'c_ax_{}'.format(wing_category):min_x_w,
                                 'c_bx_{}'.format(wing_category):max_x_w,
                                 'c_ay_{}'.format(wing_category):min_y_w,
                                 'c_by_{}'.format(wing_category):max_y_w,
                                 'cw_{}'.format(wing_category):w_0_im_crop.shape[1],
                                 'ch_{}'.format(wing_category):w_0_im_crop.shape[0],

                                 'folder':file_folder

                                 }])

    if os.path.isfile(file_path_save+"cropping_log_population_{}.xlsx".format(population_id)):
        wb = load_workbook(file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
    else:
        wb = openpyxl.Workbook()
        wb.save(filename =file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
        wb = load_workbook(filename =file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
    if not "crop_{}".format(wing_category) in wb.sheetnames:
        wb.create_sheet("crop_{}".format(wing_category))
        ws = wb["crop_{}".format(wing_category)]
        ws.append(["filename", "w0", "h0", "w1", "h1", 'c_ax_{}'.format(wing_category), 'c_bx_{}'.format(wing_category), 'c_ay_{}'.format(wing_category),'c_by_{}'.format(wing_category),'cw_{}'.format(wing_category),'ch_{}'.format(wing_category),'folder'])

    ws = wb["crop_{}".format(wing_category)]

    for r in dataframe_to_rows(append_data, index=False, header=False):  #No index and don't append the column headers
        ws.append(r)
    wb.save(file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
    
    
    
### BATCH CROPPING
pid_list=[145,151,161,173]
for population_id in pid_list:
  file_path_base="/content/drive/My Drive/wing/"
  file_path_main=file_path_base+"wing_new_populations/"
  file_path_img =file_path_main +"population_{}_original/".format(population_id)
  file_path_save =file_path_main+"images_masks_population_{}/".format(population_id)
  population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
  N_wing=len(population_excel["bodyparts"][1:])

  if not os.path.exists(file_path_save+"perfect_cropped/"):
    os.makedirs(file_path_save+"perfect_cropped/")

  for file_i in range(1):
    file_category=["perfect"][file_i]
    # Read in full size segmented wing files
    file_folder_read="{}_full/".format(file_category)
    file_path_read=file_path_save+file_folder_read
    file_path_full = os.listdir(file_path_read)
    sorted_file_list_full=sorted(file_path_full)
    num_wings_crop=int(len(sorted_file_list_full)/4)
    if file_category=="missing":
      num_wings_crop=int(len(sorted_file_list_full)/2)

    #Save file locations
    file_folder="{}_cropped/".format(file_category)
    file_save_crop=file_path_save+file_folder
    for i in range(0,num_wings_crop):

      print(file_category+":{}/{}\n".format(i, num_wings_crop))

      if file_category=="missing":
        #READ DATA
        w_0_im=cv2.imread(file_path_read + sorted_file_list_full[2*i],-1)
        w_1_im=cv2.imread(file_path_read + sorted_file_list_full[2*i+1],-1)
        #CROP
        crop_wing_func(w_0_im,w_1_im, file_save_crop,sorted_file_list_full[2*i],sorted_file_list_full[2*i+1])


      if file_category=="perfect" or file_category=="select_perfect":

        #READ DATA
        fw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i],-1)
        hw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+1],-1)
        fw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+2],-1)
        hw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+3],-1)

        #CLEAR DISCONNECTED COMPONENTS
        population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
        wi=population_excel.loc[population_excel["bodyparts"]==sorted_file_list_full[4*i][:-13] + "1.dng"].index[0]
        fw_x0=int(float(population_excel["fore1"][wi]))
        fw_y0=int(float(population_excel["fore1.1"][wi]))
        hw_x1=int(float(population_excel["hind2"][wi]))
        hw_y1=int(float(population_excel["hind2.1"][wi]))


        bgmask = flood(hw_1_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (hw_y1,hw_x1), tolerance=0.1)
        hw_1_im[wingmask==False]=(255,255,255)

        bgmask = flood(hw_0_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (hw_y1,hw_x1), tolerance=0.1)
        hw_0_im[wingmask==False]=(255,255,255)

        bgmask = flood(fw_1_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (fw_y0,fw_x0), tolerance=0.1)
        fw_1_im[wingmask==False]=(255,255,255)

        bgmask = flood(fw_0_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (fw_y0,fw_x0), tolerance=0.1)
        fw_0_im[wingmask==False]=(255,255,255)

        #CROP
        crop_wing_func(fw_0_im,fw_1_im, file_save_crop,sorted_file_list_full[4*i],sorted_file_list_full[4*i+2])
        crop_wing_func(hw_0_im,hw_1_im, file_save_crop,sorted_file_list_full[4*i+1],sorted_file_list_full[4*i+3])


      if file_category=="sticky":
        #READ DATA
        fw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i],-1)
        hw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+1],-1)
        fw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+2],-1)
        hw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+3],-1)
        #CROP
        crop_wing_func(fw_0_im,fw_1_im, file_save_crop,sorted_file_list_full[4*i],sorted_file_list_full[4*i+2])
        crop_wing_func(hw_0_im,hw_1_im, file_save_crop,sorted_file_list_full[4*i+1],sorted_file_list_full[4*i+3])




### CLEAN UP FLOOD FILL
population_id=173
file_path_base="/content/drive/My Drive/wing/"
file_path_main=file_path_base+"wing_new_populations/"
file_path_img =file_path_main +"population_{}_original/".format(population_id)
file_path_save =file_path_main+"images_masks_population_{}/".format(population_id)
population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
N_wing=len(population_excel["bodyparts"][1:])
file_category=["perfect"][0]
# Read in full size segmented wing files
file_folder_read="{}_full/".format(file_category)
file_path_read=file_path_save+file_folder_read



fw_0_im=cv2.imread(file_path_read + "population_173+FMNH_4603358+stack_0_fw_full.tif",-1)
hw_0_im=cv2.imread(file_path_read + "population_173+FMNH_4603358+stack_0_hw_full.tif",-1)
fw_1_im=cv2.imread(file_path_read + "population_173+FMNH_4603358+stack_1_fw_full.tif",-1)
hw_1_im=cv2.imread(file_path_read + "population_173+FMNH_4603358+stack_1_fw_full.tif",-1)


plt.imshow(hw_0_im)



hw_0_im[...,1]


bgmask = flood(hw_0_im[..., 1], (1,1), tolerance=0.1)
bgmask = bgmask.astype(int)

wingmask=flood(bgmask, (2000,5000), tolerance=0.1)


plt.imshow(bgmask)



plt.imshow(wingmask)



###PROCESS
pid_list=[68,119,134,138,145,151,161,173]
for population_id in pid_list:
  file_path_base="/content/drive/My Drive/wing/"
  file_path_main=file_path_base+"wing_new_populations/"
  file_path_img =file_path_main +"population_{}_original/".format(population_id)
  file_path_save =file_path_main+"images_masks_population_{}/".format(population_id)
  population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
  N_wing=len(population_excel["bodyparts"][1:])

  for file_i in range(1):
    file_category=["perfect"][file_i]
    # Read in full size segmented wing files
    file_folder_read="{}_full/".format(file_category)
    file_path_read=file_path_save+file_folder_read
    file_path_full = os.listdir(file_path_read)
    sorted_file_list_full=sorted(file_path_full)
    num_wings_crop=int(len(sorted_file_list_full)/4)
    if file_category=="missing":
      num_wings_crop=int(len(sorted_file_list_full)/2)

    #Save file locations
    file_folder="{}_cropped/".format(file_category)
    file_save_crop=file_path_save+file_folder
    for i in range(0,num_wings_crop):

      print(file_category+":{}/{}\n".format(i, num_wings_crop))

      if file_category=="missing":
        #READ DATA
        w_0_im=cv2.imread(file_path_read + sorted_file_list_full[2*i],-1)
        w_1_im=cv2.imread(file_path_read + sorted_file_list_full[2*i+1],-1)
        #CROP
        crop_wing_func(w_0_im,w_1_im, file_save_crop,sorted_file_list_full[2*i],sorted_file_list_full[2*i+1])


      if file_category=="perfect" or file_category=="select_perfect":

        #READ DATA
        fw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i],-1)
        hw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+1],-1)
        fw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+2],-1)
        hw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+3],-1)
        #CROP
        crop_wing_func(fw_0_im,fw_1_im, file_save_crop,sorted_file_list_full[4*i],sorted_file_list_full[4*i+2])
        crop_wing_func(hw_0_im,hw_1_im, file_save_crop,sorted_file_list_full[4*i+1],sorted_file_list_full[4*i+3])


      if file_category=="sticky":
        #READ DATA
        fw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i],-1)
        hw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+1],-1)
        fw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+2],-1)
        hw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+3],-1)
        #CROP
        crop_wing_func(fw_0_im,fw_1_im, file_save_crop,sorted_file_list_full[4*i],sorted_file_list_full[4*i+2])
        crop_wing_func(hw_0_im,hw_1_im, file_save_crop,sorted_file_list_full[4*i+1],sorted_file_list_full[4*i+3])



