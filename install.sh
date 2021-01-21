#!/bin/bash

# Install OpenCV
sudo apt update
sudo apt-get install python3-opencv

# Install Tensorflow for C
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz 
sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz  
git clone https://github.com/serizba/cppflow.git          
sudo ldconfig     

# Install Tensorflow for Python
sudo apt install python3-pip
pip3 install -r requirements.txt