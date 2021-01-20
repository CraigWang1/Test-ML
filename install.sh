#!/bin/bash

sudo apt update
cd /tmp

# Install tensorflow for C
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz 
tar -C /usr/local -xzf /tmp/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz  
git clone https://github.com/serizba/cppflow.git          
sudo ldconfig     

# Install OpenCV
sudo apt-get install python3-opencv
