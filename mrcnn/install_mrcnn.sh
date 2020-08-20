#!/usr/bin/env bash

# Install the required libraries
conda install -c anaconda numpy
conda install -c anaconda scipy
conda install -c anaconda h5py
conda install -c anaconda pillow
conda install -c anaconda cython
conda install -c anaconda matplotlib
conda install -c anaconda scikit-image==0.16.2
conda install -c conda-forge imgaug
conda install -c anaconda ipython
conda install -c anaconda beautifulsoup4
conda install -c anaconda tensorflow-gpu==1.15.0
conda install -c anaconda keras
conda install -c anaconda opencv

# Install Mrcnn
cd mrcnn
python setup.py install
