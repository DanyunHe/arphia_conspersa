import numpy as np
import tifffile as tif 


# read all cropped images in a folder and do svd 
fn="../imgs/population_173/"
name_list=open(fn+"name_list.txt",'r')
names=name_list.readlines()

for file_name in names:
    file_name=file_name.strip()
    print("processing: ", file_name)
    im_rl = tif.imread(fn+file_name+"+stack_0_hw_crop.tif")
    im_rl=np.array(im_rl).astype('float32');im_rl/=255.0
    im_tl = tif.imread(fn+file_name+"+stack_1_hw_crop.tif")
    im_tl=np.array(im_tl).astype('float32');im_tl/=255.0
    (N,M,z)=im_rl.shape

    # Copy the data from images into the array to perform the least squares fitting
    A=np.zeros((M*N,3+1))

    for l in range(3):
        c=np.zeros((N,M))
        c[:,:]=im_rl[:,:,l]
        c.shape=((M*N))
        A[:,l]=c
    A[:,3]=1.0

    # Create the source term
    y=np.zeros((M*N,3))
    for l in range(3):
        c=np.zeros((N,M))
        c[:,:]=im_tl[:,:,l]
        c.shape=((M*N))
        y[:,l]=c
        
        # Perform the least squares fitting
    F=np.linalg.lstsq(A,y,rcond=None)[0]
    print(F)

    # Reconstruct the regular image (objects)
    ao=np.zeros((N,M,3))
    for l in range(3):
        c=np.dot(A,F[:,l]);
        c.shape=(N,M)
        ao[:,:,l]=c

    np.clip(ao,0,1,out=ao)
    ao*=255
    ao=ao.astype('uint8')
    tif.imwrite(fn+file_name+"+stack_0_hw_crop_fit.tif",ao)
