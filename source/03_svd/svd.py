import numpy as np
from PIL import Image
from skimage import io
from rembg import remove
import cv2

# input: result/02.output. transmitted and reflected light images 
# output: svd result. 03.output 

# align the two images. alignment result save as 03_mapped.png, 03_target.png  
fn="uc"

# Read images
# Reflected image
im_rl = cv2.imread('../../result/03_mapped.png')
#remove background
# im_rl = remove(im_rl)
# convert four channels (RGBA) to three channels (RGB)
# im_rl=cv2.cvtColor(im_rl, cv2.COLOR_RGBA2RGB) 
im_rl=im_rl.astype(np.float64);im_rl/=255.0

# Transmitted image 
im_tl = cv2.imread('../../result/03_target.png')
#remove background
# im_tl = remove(im_tl)
# convert four channels (RGBA) to three channels (RGB)
# im_tl=cv2.cvtColor(im_tl, cv2.COLOR_RGBA2RGB) 
im_tl=im_tl.astype(np.float64);im_tl/=255.0

#pick a area
crop_im_rl=im_rl[3500:4000,4500:5000,:]
crop_im_tl=im_tl[3500:4000,4500:5000,:]

io.imsave("%s_crop.png"%fn,crop_im_rl)

# Get the dimension of image
(N,M,z)=crop_im_rl.shape

B=np.zeros((M*N,6))
B_avg=np.zeros(6)

for l in range(3):
    c=np.zeros((N,M))
    c[:,:]=crop_im_rl[:,:,l]
    c.shape=((M*N));
    B_avg[l]=np.mean(c)
    c-=B_avg[l]
    B[:,l]=c

for l in range(3):
    c=np.zeros((N,M))
    c[:,:]=crop_im_tl[:,:,l]
    c.shape=((M*N));
    B_avg[l+3]=np.mean(c)
    c-=B_avg[l+3]
    B[:,3+l]=c

# Compute SVD and save data
(u,s,v)=np.linalg.svd(B.T,full_matrices=0)

(N2,M2,z)=im_rl.shape
# Loop over six vectors
sz=M2*N2
b=np.zeros((sz))

# use the full image
A=np.zeros((M2*N2,6))
A_avg=np.zeros(6)

for l in range(3):
    c=np.zeros((N2,M2))
    c[:,:]=im_rl[:,:,l]
    c.shape=((M2*N2));
    A_avg[l]=np.mean(c)
    c-=A_avg[l]
    A[:,l]=c

for l in range(3):
    c=np.zeros((N2,M2))
    c[:,:]=im_tl[:,:,l]
    c.shape=((M2*N2));
    A_avg[l+3]=np.mean(c)
    c-=A_avg[l+3]
    A[:,3+l]=c

# output first vector
result=A@u[:,0]
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
io.imsave("%s_1.png"%fn,result)
io.imsave("%s_1_pos.png"%fn,pos)
io.imsave("%s_1_neg.png"%fn,neg)

# output three vectors [u2,u1,u3]
result=A@u[:,[1,0,2]]
pos=np.zeros((sz,3))
neg=np.zeros((sz,3))
for j in range(sz):
    for k in range(3):
        if result[j,k]>0:
            pos[j,k]=result[j,k]
        else:
            neg[j,k]=-result[j,k]
        
result=result.reshape((N2,M2,3))
pos=pos.reshape((N2,M2,3))
neg=neg.reshape((N2,M2,3))
io.imsave("%s_3.png"%fn,result)
io.imsave("%s_3_pos.png"%fn,pos)
io.imsave("%s_3_neg.png"%fn,neg)

'''
for i in range(1):
    print("i %d\n"%i)

    # Plot the positive component
    maxu=np.amax(u[:,i])
    imaxu=1./maxu
    for j in range(sz):
        b[j]=v[j,i]*imaxu
        if b[j]<0.: b[j]=0.
        if b[j]>1.: b[j]=1.
    result=np.dot(b.reshape(-1,1),v[i,:].reshape(1,-1))*s[i]

    result=1.-result
    result=result.reshape((N,M,6))
    io.imsave("u_pos_rl%d.png" % (i),result[:,:,:3])
    io.imsave("u_pos_tl%d.png" % (i),result[:,:,3:])


    # Plot the negative component
    minu=np.amin(u[:,i])
    iminu=1./minu
    for j in range(sz):
        b[j]=u[j,i]*iminu
        if b[j]<0.: b[j]=0.
        if b[j]>1.: b[j]=1.
    result=np.dot(b.reshape(-1,1),v[i,:].reshape(1,-1))

    result=1.-result
    result=result.reshape((N,M,6))
    io.imsave("u_neg_rl%d.png" % (i),result[:,:,:3])
    io.imsave("u_neg_tl%d.png" % (i),result[:,:,3:])
'''
