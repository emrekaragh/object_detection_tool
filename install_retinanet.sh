#!/usr/bin/env bash

# Install the required libraries
pip install numpy scipy h5py
pip install scikit-learn Pillow imutils
pip install beautifulsoup4
pip install tensorflow-gpu
pip install keras
pip install opencv-contrib-python

# Install Retinanet
cd retinanet/keras_retinanet
python setup.py install
