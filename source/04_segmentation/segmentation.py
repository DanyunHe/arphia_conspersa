# take a svd image and segment the domains and veins using cellpose
# output cellpose segmentation results and outline images

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import cv2
import argparse

from cellpose import models
from cellpose.io import imread
from cellpose import io
from cellpose import plot, utils

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Segment wing domains and veins using Cellpose')
    parser.add_argument('--folder_name', type=str, default='../../result/03_svd/',
                        help='Input folder containing SVD images (default: ../../result/03_svd/)')
    parser.add_argument('--file_name', type=str, default='population_34+FMNH_4669526',
                        help='Base filename without extension (default: population_34+FMNH_4669526)')
    parser.add_argument('--model_type', type=str, default='CP_20230503_151910',
                        help='Cellpose model type (default: CP_20230503_151910)')
    parser.add_argument('--diameter', type=int, default=100,
                        help='Cell diameter for Cellpose (default: 100)')
    parser.add_argument('--cellpose_models_path', type=str, default=None,
                        help='Path to Cellpose models directory (default: ~/.cellpose/models)')
    args = parser.parse_args()

    # Set cellpose model path if provided, otherwise use default (~/.cellpose/models)
    if args.cellpose_models_path:
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = args.cellpose_models_path
    elif "CELLPOSE_LOCAL_MODELS_PATH" not in os.environ:
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.expanduser("~/.cellpose/models")

    # Load cellpose model
    # model_type='cyto' or 'nuclei' or 'cyto2' or custom model name
    model = models.CellposeModel(model_type=args.model_type)
    # Alternative: load from specific file path
    # model = models.CellposeModel(pretrained_model="/path/to/custom_model.pth")

    # Read image
    folder_name = args.folder_name
    file_name = args.file_name
    output_folder = "../../result/04_segmentation/"
    os.makedirs(output_folder, exist_ok=True)

    img=imread(folder_name+file_name+'_hw_1.png')
    channels = [[0,0]]
    masks, flows, styles = model.eval(img, diameter=args.diameter, channels=channels)
    # save cellpose outputs to result/04_segmentation/
    io.masks_flows_to_seg(img, masks, flows, output_folder+file_name, channels)

    # generate outline image
    # Load original images from Step 03 output
    # Try Step 03 output first, then fall back to Step 02
    orig_img_path = folder_name+file_name+'+stack_1_hw_crop.tif'
    if not os.path.exists(orig_img_path):
        # Fall back to Step 02 output
        import glob
        possible_paths = glob.glob(f"../../result/02_extraction/population_*/perfect_cropped/{file_name}+stack_1_hw_crop.tif")
        if possible_paths:
            orig_img_path = possible_paths[0]

    orig_img = cv2.imread(orig_img_path)
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
    plt.imsave(output_folder+file_name+'_hw_outline.png',result)
    np.save(output_folder+file_name+'_hw_outline',result)