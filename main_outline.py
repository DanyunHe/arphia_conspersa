import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import cv2

from cellpose import models
from cellpose.io import imread
from cellpose import io
from cellpose import plot, utils


# read image info

folder_name="./images_align/select_perfect_cropped/"

with open(folder_name+"name_list.txt", 'r') as f:
    text = f.read()
    names = text.split()

for i in range(len(names)):
    
    # read images
    # file_name=folder_name+df.iloc[i,0]+'_1_pos'
    # print("processing: ", i, file_name)
    # # img=imread(file_name)
    # # imgs.append(img)
    # # file_names.append(file_name)
    # dat = np.load(file_name+'_seg.npy', allow_pickle=True).item()
 
    # # save images for binarization 
    # result=dat['img']/255.
    # result[np.nonzero(dat['masks'])]=1
    # result=result.astype('float64')
    # edge=utils.masks_to_edges(dat['masks'])
    # result[np.nonzero(edge)]=0.5


    # # plt.imshow(dat1['img'])
    # plt.imsave(file_name+'_outline.png',result,cmap=plt.cm.gray)
    # np.save(file_name+'_outline',result)
    

    file_name=names[i]
    
    # Load original images
    orig_img = cv2.imread(folder_name+file_name+'+stack_1_hw_crop.tif')
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) #255 is mask
    
    # Load cellpose results
    dat = np.load(folder_name+file_name+'_hw_1_seg.npy',allow_pickle=True).item()
    edge=utils.masks_to_edges(dat['masks'])
    result=orig_img/255.
    # result[np.nonzero(dat['masks'])]=1
    # result=result.astype('float64')
    # result[np.nonzero(edge)]=0.5

    nx=len(result[0,:])
    ny=len(result[:,0])


    for jj in range(0,ny):
        for ii in range(0,nx):
            if orig_img[jj,ii]==255:
                result[jj,ii]=0.5 # background
            elif dat['masks'][jj,ii]!=0:
                result[jj,ii]=0 # cells
            else:
                result[jj,ii]=1
                
            if edge[jj,ii]!=0:
                result[jj,ii]=1 # edges


            #if result[jj,ii]==1:
            #    result[jj,ii]=0
            #if result[jj,ii]>0 and result[jj,ii]<1:
            #    result[jj,ii]=1



    # plt.imshow(dat1['img'])
    # plt.ylim(0,2600)
    plt.imshow(result)
    plt.imsave(folder_name+'outline/'+file_name+'_hw_outline.png',result)
    np.save(folder_name+'outline/'+file_name+'_hw_outline',result)