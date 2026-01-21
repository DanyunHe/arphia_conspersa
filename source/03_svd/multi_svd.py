import numpy as np
from PIL import Image
from skimage import io
import cv2
import tifffile as tif 
import matplotlib.pyplot as plt

def save_comp(fn, im_rl, im_tl, temp_dir=None):
    """
    Save SVD components.

    Args:
        fn: Base filename for outputs
        im_rl: Reflected light image
        im_tl: Transmitted light image
        temp_dir: Optional directory for intermediate files. If None, saves all to same location as fn.
    """

    (N,M,z)=im_tl.shape

    # Rotate the image 90 degree
    im_rl=np.transpose(im_rl, (1, 0, 2))
    im_tl=np.transpose(im_tl,(1,0,2))
    
    # Choose a long rectangle range 
    x1=N//2-1000;x2=N//2+1000
    y1=M//2-500;y2=M//2+500

    # y1=1500;x1=1000;y2=2000;x2=3000
    #pick window 1
    crop_im_rl=im_rl[x1:x2,y1:y2,:]
    crop_im_tl=im_tl[x1:x2,y1:y2,:]
    
    # Save crop1 to temp directory if provided, otherwise to main directory
    import os
    crop1_fn = os.path.join(temp_dir, os.path.basename(fn) + "_crop1.png") if temp_dir else "%s_crop1.png"%fn
    io.imsave(crop1_fn, (crop_im_rl*255).astype(np.uint8))
    # Get the dimension of image
    (N,M,z)=crop_im_rl.shape

    #2mn*12, halfe(6) x f, other half(6) x g f(...=0;...=1); g=1-f
    mn=M*N
    B=np.zeros((mn,12))
    B_avg=np.zeros(12)

    f=np.zeros((N,M))
    dx=1./(N-1)
    for i in range(N):
        f[i,:]=i*dx

    for l in range(3):
        c=np.zeros((N,M))
        c[:,:]=crop_im_rl[:,:,l]
        B_avg[l]=np.mean(c)
        c-=B_avg[l]
        temp=c*f
        temp.shape=((mn));
        B[:,l]=temp
        temp=c*(1-f)
        temp.shape=((mn));
        B[:,l+6]=temp

    for l in range(3):
        c=np.zeros((N,M))
        c[:,:]=crop_im_tl[:,:,l]
        B_avg[l+3]=np.mean(c)
        c-=B_avg[l+3]
        temp=c*f
        temp.shape=((mn));
        B[:,3+l]=temp
        temp=c*(1-f)
        temp.shape=((mn));
        B[:,3+l+6]=temp

    # Compute SVD and save data
    (u,s,v)=np.linalg.svd(B.T,full_matrices=0)


    (N2,M2,z)=im_rl.shape
    # Loop over six vectors
    sz=M2*N2
    b=np.zeros((sz))

    # use the full image
    A=np.zeros((M2*N2,12))
    A_avg=np.zeros(12)
    g=np.zeros((N2,M2))
    dx=1./(N2-1)
    for i in range(N2):
        g[i,:]=i*dx

    for l in range(3):
        c=np.zeros((N2,M2))
        c[:,:]=im_rl[:,:,l]
        A_avg[l]=np.mean(c)
        c-=A_avg[l]
        temp=c*g
        temp.shape=((sz));
        A[:,l]=temp
        temp=c*(1-g)
        temp.shape=((sz));
        A[:,l+6]=temp

    for l in range(3):
        c=np.zeros((N2,M2))
        c[:,:]=im_tl[:,:,l]
        A_avg[l+3]=np.mean(c)
        c-=A_avg[l+3]
        temp=c*g
        temp.shape=((sz));
        A[:,3+l]=temp
        temp=c*(1-g)
        temp.shape=((sz));
        A[:,3+l+6]=temp

    # output first vector
    # construct the result using u1, u2
    result=A@u[:,0]

    pos=np.zeros(sz)
    neg=np.zeros(sz)
    for j in range(sz):
        if result[j]<0:
            neg[j]=-result[j]
            # result[j]=0 # change
        elif result[j]<1:
            pos[j]=result[j]
        # else:
        #     pos[j]=1 #change
        #     result[j]=1 #change
            
    result=result.reshape((N2,M2))
    pos=pos.reshape((N2,M2))
    neg=neg.reshape((N2,M2))

    # Make background black
    np.clip(result,0,1)
    im_bkg=np.mean(im_rl,axis=2)
    im_bkg2=np.mean(im_tl,axis=2)

    result[im_bkg==1.]=0
    result[im_bkg2==1.]=0
    pos[im_bkg==1.]=0

    result = 1.-result
    result[im_bkg==1.]=0
    result[im_bkg2==1.]=0

    # Transpose back to the original orientation
    result=np.transpose(result, (1, 0))
    pos=np.transpose(pos, (1, 0))
    neg=np.transpose(neg, (1, 0))

    # Save final output to main directory
    io.imsave("%s_hw_1.png"%fn,(result*255).astype(np.uint8))
    # Save intermediate pos/neg to temp directory if provided
    pos_fn = os.path.join(temp_dir, os.path.basename(fn) + "_hw_1_pos.png") if temp_dir else "%s_hw_1_pos.png"%fn
    neg_fn = os.path.join(temp_dir, os.path.basename(fn) + "_hw_1_neg.png") if temp_dir else "%s_hw_1_neg.png"%fn
    io.imsave(pos_fn,(pos*255).astype(np.uint8))
    io.imsave(neg_fn,(neg*255).astype(np.uint8))


# Make dark background for im_target according to background in im_ref
def dark_bkg(fn,im_ref,im_target):
     # Make background black

    im_bkg=np.mean(im_ref,axis=2)
    # print(im_bkg)
    im_target[im_bkg==1.]=0
    io.imsave("%s_hw_1.png"%fn,im_target)
    

if __name__=="__main__":
    
    folder_name="./imgs/population_34/"
    
    file_name="population_34+FMNH_4669526"
    
     # Read images
    # Reflected image
    # im_rl = cv2.imread('./imgs/population_34/mapped.png')
    im_rl = cv2.imread(folder_name+file_name+'+stack_0_hw_crop_mapped.png')
    print(type(im_rl))
    #remove background
    # im_rl = remove(im_rl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_rl=cv2.cvtColor(im_rl, cv2.COLOR_RGBA2RGB) 
    im_rl=im_rl.astype(np.float64);
    im_rl/=255.0

    # Transmitted image 
    im_tl = cv2.imread(folder_name+file_name+'+stack_1_hw_crop.tif')
    # im_tl = cv2.imread('./align_img/target.png')

    #remove background
    # im_tl = remove(im_tl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_tl=cv2.cvtColor(im_tl, cv2.COLOR_RGBA2RGB) 
    im_tl=im_tl.astype(np.float64);
    im_tl/=255.0
    print(im_tl.shape)

    fn=folder_name+"svd_result/"+file_name
    save_comp(fn,im_rl,im_tl)
