#!/usr/bin/env bash

# Install the required libraries
pip install numpy scipy h5py
pip install Pillow
pip install Cython
pip install matplotlib
pip install scikit-image
pip install imgaug
pip install ipython
pip install beautifulsoup4
pip install tensorflow-gpu==1.15.3
pip install keras
pip install opencv-contrib-python

# Install Retinanet
cd mrcnn
python setup.py install
