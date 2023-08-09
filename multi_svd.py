import numpy as np
from PIL import Image
from skimage import io
import cv2
import tifffile as tif 
import matplotlib.pyplot as plt



def save_comp(fn,im_rl,im_tl):

    (N,M,z)=im_tl.shape
    print(N,M)
    
    # Choose a long rectangle range 
    x1=N//2-1000;x2=N//2+1000
    y1=M//2-500;y2=M//2+500

    # y1=1500;x1=1000;y2=2000;x2=3000
    #pick window 1
    crop_im_rl=im_rl[x1:x2,y1:y2,:]
    crop_im_tl=im_tl[x1:x2,y1:y2,:]

    io.imsave("%s_crop1.png"%fn,crop_im_rl)
    # Get the dimension of image
    (N,M,z)=crop_im_rl.shape
    print(N,M)

    #2mn*12, halfe(6) x f, other half(6) x g f(...=0;...=1); g=1-f
    mn=M*N
    B=np.zeros((mn,12))
    B_avg=np.zeros(12)

    f=np.zeros((N,M))
    dx=1./(N-1)
    for i in range(N):
        f[i,:]=i*dx

    print(f[10,10],f[-10,-10])


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

    print(B[10,:],B[-10,:])
    print(B)
    plt.imshow(B)
    plt.imsave('B.png',B)
    # cv2.imwrite('B.png',B)

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
    print(u[:,0])
    # result[:M2*x1]=A[:M2*x1,:]@u1[:,0]
    # result[M2*(N2-x2):]=A[M2*(N2-x2):,:]@u2[:,0]


    # gap=x2-x1
    # for i in range(gap):
    #     idx=x1+i
    #     result[M2*idx:M2*(idx+1)]=A[M2*idx:M2*(idx+1),:]@(u2[:,0]*(1.*i)/gap+u1[:,0]*(1.-i/gap))

    pos=np.zeros(sz)
    neg=np.zeros(sz)
    for j in range(sz):
        if result[j]>0:
            pos[j]=result[j]
        else:
            neg[j]=-result[j]
            
    result=result.reshape((N2,M2))
    pos=pos.reshape((N2,M2))
    neg=neg.reshape((N2,M2))
    # tif.imsave("%s_1.tif"%fn,1.-result)
    # tif.imsave("%s_1_pos.tif"%fn,1.-pos)
    # tif.imsave("%s_1_neg.tif"%fn,1.-neg)
    
    result=np.transpose(result, (1, 0))
    pos=np.transpose(pos, (1, 0))
    neg=np.transpose(neg, (1, 0))

    io.imsave("%s_hw_1.png"%fn,result)
    io.imsave("%s_hw_1_pos.png"%fn,pos)
    io.imsave("%s_hw_1_neg.png"%fn,neg)


if __name__=="__main__":
    
    folder_name="./images_align/select_perfect_cropped/"
    
    file_name="population_60+FMNH_4602200"
    
     # Read images
    # Reflected image
    im_rl = cv2.imread(folder_name+file_name+'+stack_0_hw_crop_mapped.png')
    print(type(im_rl))
    #remove background
    # im_rl = remove(im_rl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_rl=cv2.cvtColor(im_rl, cv2.COLOR_RGBA2RGB) 
    im_rl=im_rl.astype(np.float64);
    im_rl/=255.0
    im_rl=np.transpose(im_rl, (1, 0, 2))

    # Transmitted image 
    im_tl = cv2.imread(folder_name+file_name+'+stack_1_hw_crop.tif')

    #remove background
    # im_tl = remove(im_tl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_tl=cv2.cvtColor(im_tl, cv2.COLOR_RGBA2RGB) 
    im_tl=im_tl.astype(np.float64);
    im_tl/=255.0
    im_tl=np.transpose(im_tl, (1, 0, 2))
    print(im_tl.shape)

    fn=folder_name+"svd_result/"+file_name
    save_comp(fn,im_rl,im_tl)
