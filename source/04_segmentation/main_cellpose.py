import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from cellpose import models
from cellpose.io import imread
from cellpose import io

os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/Users/danyunhe/.cellpose/models"
# model_type='cyto' or 'nuclei' or 'cyto2'
model = models.Cellpose(model_type='CP_20230503_151910')

# read image info
folder_name="./images_fore_hind_wing_cropped/"
df=pd.read_excel(folder_name+"cropping_log.xlsx")
print(df.iloc[0,0])
imgs=[]
file_names=[]
for i in range(df.shape[0]):
    
    # read images
    file_name=folder_name+df.iloc[i,0]
    print("processing: ", i, file_name)
    img=imread(file_name+'_1_pos.png')
    imgs.append(img)
    file_names.append(file_name)
 
nimg=len(imgs)   
# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[0,0]]


# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended)
# diameter can be a list or a single number for all images

masks, flows, styles, diams = model.eval(imgs, diameter=100, channels=channels)
    
# save outputs
io.masks_flows_to_seg(imgs, masks, flows, diams, file_names, channels)