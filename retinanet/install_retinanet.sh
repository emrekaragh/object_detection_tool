#!/usr/bin/env bash

# Install the required libraries
conda install -c anaconda numpy==1.17.4
conda install -c anaconda scipy==1.5.0
conda install -c anaconda h5py==2.10.0
# conda install -c anaconda scikit-learn=
conda install -c anaconda pillow==5.2.0
conda install -c conda-forge imutils==0.5.3
conda install -c anaconda beautifulsoup4==4.9.0
conda install -c anaconda cython
conda install -c anaconda matplotlib==3.2.1
conda install -c anaconda tensorflow-gpu==1.15.3
conda install -c anaconda keras
conda install -c conda-forge opencv

# Install Retinanet
cd retinanet/keras_retinanet
python setup.py install
