import numpy as np
import tifffile as tif 
import sys
import os

if __name__=="__main__":

    # read cropped images from Step 02 output and do color conversion
    # Input: cropped images from result/02_extraction/
    # Output: fitted image to result/03_svd/population_XX/
    input_dir = "../../../result/02_extraction/"
    output_base_dir = "../../../result/03_svd/"
    file_name = sys.argv[1]  # e.g., population_34+FMNH_4669630

    # Extract population ID from filename to create subdirectory
    population_id = file_name.split('+')[0]  # e.g., "population_34"
    output_dir = os.path.join(output_base_dir, population_id) + "/"

    # Try to find the images - they could be in population_XX subdirectories
    import os
    import glob
    possible_paths = [
        input_dir + file_name + "+stack_0_hw_crop.tif",
        input_dir + f"population_*/perfect_cropped/" + file_name + "+stack_0_hw_crop.tif"
    ]

    found_path = None
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            found_path = os.path.dirname(matches[0]) + "/"
            break

    if found_path is None:
        found_path = input_dir
        print(f"Warning: Using default path {found_path}")

    # print("processing: ",found_path + file_name + "+stack_0_hw_crop.tif")
    # found_path = input_dir + f"population_%d/perfect_cropped/"%pid
    # print("processing: ",found_path + file_name + "+stack_0_hw_crop.tif")
    im_rl = tif.imread(found_path + file_name + "+stack_0_hw_crop.tif")
    im_rl=np.array(im_rl).astype('float32');im_rl/=255.0
    # print("processing: ",found_path + file_name + "+stack_1_hw_crop.tif")
    im_tl = tif.imread(found_path + file_name + "+stack_1_hw_crop.tif")
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
    # Save fitted image to result/03_svd/ directory
    os.makedirs(output_dir, exist_ok=True)
    tif.imwrite(output_dir + file_name + "+stack_0_hw_crop_fit.tif", ao)
