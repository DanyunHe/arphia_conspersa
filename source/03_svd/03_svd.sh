#!/bin/bash 
# This script run the alignmenet code to align the transmitted and reflated images 

# Exit immediately if any command fails
set -e

cd ./03_svd/align_img
# Read input from the command line argument 
# Read up to two example arguments (you can add more as needed)
ARG1=$1
echo "Running alignment test with arguments: $ARG1"

# Convert color to fit 
python3 ./rl2tl_single.py "$@"
# python3 ./rl2tl_single.py population_34+FMNH_4669526

# Compile and run the alignment test program wit arguments
make clean
make
./align_single "$@"
# ./align_single population_34+FMNH_4669526

# svd 
cd ../
# Run the svd image processing script with the same arguments
python3 svd_img.py "$@"

# End of script
