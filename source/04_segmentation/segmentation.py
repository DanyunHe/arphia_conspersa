# take a svd image and segment the domains and veins using cellpose 
# output cellpose segmentation results and outline images

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import cv2

from cellpose import models
from cellpose.io import imread
from cellpose import io
from cellpose import plot, utils

if __name__ == "__main__":
    
    # load cellpose model 
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/Users/dhe/.cellpose/models"
    # model_type='cyto' or 'nuclei' or 'cyto2'
    model = models.CellposeModel(model_type='CP_20230503_151910')
    # model = models.CellposeModel(pretrained_model="/Users/dhe/.cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_img_vein_2023_05_12_17_52_06.124239")

    # read image
    folder_name =  "../../result/03_svd/"
    file_name = "population_34+FMNH_4669526"
    # filename = sys.argv[1]  # e.g., /path/to/image_population_34+FMNH_4669630+stack_1.png
    img=imread(folder_name+file_name+'_hw_1.png')
    channels = [[0,0]]
    masks, flows, styles = model.eval(img, diameter=100, channels=channels)
    # save cellpose outputs
    io.masks_flows_to_seg(img, masks, flows, folder_name+file_name, channels)

    # generate outline image
    # Load original images
    orig_img = cv2.imread(folder_name+file_name+'+stack_1_hw_crop.tif')
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) #255 is mask
    
    # Load cellpose results
    # dat = np.load(filename+'_hw_1_seg.npy',allow_pickle=True).item()
    edge=utils.masks_to_edges(masks)
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
            elif masks[jj,ii]!=0:
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
    plt.imsave("../../result/04_segmentation/"+file_name+'_hw_outline.png',result)
    np.save("../../result/04_segmentation/"+file_name+'_hw_outline',result)