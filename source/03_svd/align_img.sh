#!/bin/bash 
# This script run the alignmenet code to align the transmitted and reflated images 

# Exit immediately if any command fails
set -e

# Read input from the command line argument 

# run align_test.cc
cd ./align_img
make clean
make

# Read up to two example arguments (you can add more as needed)
ARG1=$1
ARG2=$2
echo "Running alignment test with arguments: $ARG1 $ARG2"

# Compile and run the alignment test program wit arguments
./align_test "$@"

# End of script
