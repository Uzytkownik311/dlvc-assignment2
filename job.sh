#!/bin/bash
#PBS -S /bin/bash  
#PBS -m bea
#PBS -M lafenokm@gmail.com

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

python3 cnn_cats_dogs.py --fpath ../datasets/cifar-10-batches-py
