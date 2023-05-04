import svd 
import pandas as pd
import cv2


# read all cropped images in a folder and do svd 
folder_name="./images_fore_hind_wing_cropped/"
df=pd.read_excel(folder_name+"cropping_log.xlsx")
print(df.iloc[0,0])
for i in range(df.shape[0]):
    file_name=folder_name+df.iloc[i,0]
    print("processing: ", i, file_name)

    # Read images
    # Reflected image
    im_rl = cv2.imread(file_name+'_0_hw_crop.png')
    #remove background
    # im_rl = remove(im_rl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_rl=cv2.cvtColor(im_rl, cv2.COLOR_RGBA2RGB) 
    im_rl=im_rl.astype(np.float64);im_rl/=255.0

    # Transmitted image 
    im_tl = cv2.imread(file_name+'_1_hw_crop.png')
   
    result=svd.save_comp(file_name,im_rl,im_tl)
    
