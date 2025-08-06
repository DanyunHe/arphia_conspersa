'''
Goal: Find the points for wing extraction 
Input: wing filename 
Output: points for wing extraction 
'''

import deeplabcut as dlc
import pandas as pd
from PIL import Image
import sys
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # take the image and resize to 1024x682
    working_dir = sys.argv[1] #/Users/dhe/arphia_conspersa

    # note: here the input is the transmitted light image, namely, the image with _1.png
    pid=sys.argv[2] # population_34+FMNH_4669630+stack_1.png
    filename=working_dir+'/data/'+pid
    image=Image.open(filename)
    print('Image size before resizing:', image.size)
    y1,x1=image.size
    y2,x2=1024,682
    new_size = (y2, x2)
    resized_image = image.resize(new_size)

    # Save the resized image
    resized_image.save(working_dir+'/result/01_find_pt/resize_img/'+pid)
    
    # deeplabcut whole 
    config_path=working_dir+'/data/deeplabcut_whole/segment_whole-dy-2024-06-06/config.yaml'    
    dlc.analyze_time_lapse_frames(config_path,working_dir+'/result/01_find_pt/resize_img/',save_as_csv=True)

    # convert the label to the original image size 
    xfac=x1/x2
    yfac=y1/y2
    df = pd.read_csv(working_dir+'/result/01_find_pt/resize_img/resize_imgDLC_resnet50_segment_wholeJun6shuffle1_2000.csv')
    df2=df.copy(deep=True)
    for i in range(8):
        df2.iloc[2,i*3+1]=int(float(df.iloc[2,i*3+1])*xfac)
        df2.iloc[2,i*3+2]=int(float(df.iloc[2,i*3+2])*yfac)

    df2.to_csv(working_dir+'/result/01_find_pt/01_output.csv')

    # validation: plot the image with the points 
    plt.imshow(image)
    plt.scatter(df2.iloc[2,1::3], df2.iloc[2,2::3], s=10)
    plt.savefig(working_dir+'/result/01_find_pt/01_output.png')
    plt.close()




# need my old computer to check 
# somewhere to save the results? 
# input: folder with images, txt file of population id 