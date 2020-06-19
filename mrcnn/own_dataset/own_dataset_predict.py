# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 00:27:39 2020

@author: EmreKARA
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath(__file__ + "/../../")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import own_dataset_train
import argparse

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BASE_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

def main(args):
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect your Own Dataset.')
    parser.add_argument('--device',
                        help='cpu or gpu device ex: /cpu:0 ex: /gpu:0')
    parser.add_argument('--weights',
                        help="Path to folder contains weights .h5 file and num_classes.txt. Default: last trained weights")
    parser.add_argument('--dataset',
                        help="Path to folder contains test images and annotations")
    args = parser.parse_args()
    
        
    
    config = own_dataset_train.OwnDatasetConfig()
    
    
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    
    if args.weights:
        weights_path = args.weights
    else:
        try:
            weights_path = model.find_last()
        except FileNotFoundError as fne:
            print('FileNotFoundError: [Errno 2] Could not find weight files in directory')
            print('! Please please run a successful train OR give --weights path \nOR delete last old unsuccessful log folders from mrcnn/logs(reccomended)')
            sys.exit()
        
    if args.device:
        DEVICE = args.device
    else:
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    
    if args.dataset:
        DATA_DIR = args.dataset
    else:
        DATA_DIR = os.path.abspath(ROOT_DIR + "/../dataset")
    
    num_classes_txt_path = os.path.join((weights_path+'/../'), 'num_classes.txt')
    
    try:
        f = open(num_classes_txt_path, "r")
        num_classes_from_dataset = int(f.read()) +1 # +1 is for Back Ground Class
        f.close()
        config.NUM_CLASSES = num_classes_from_dataset
        config.refresh()
        config.display()
    except Exception as ex:
        print('Please run build_dataset to create num_classes.txt')
        print('ex:', ex)
        sys.exit()
        
        
    
    
    
    dataset = own_dataset_train.OwnDataset()
    dataset.load_dataset(DATA_DIR, "test")
    
    # Must call before using the dataset
    dataset.prepare()
    
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    
    
    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    
    
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = model.find_last()
    
    
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                               dataset.image_reference(image_id)))
        
        # Run object detection
        results = model.detect([image], verbose=1)
        
        
        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        
        fig = plt.gcf()
        save_path =  os.path.join(ROOT_DIR, ('outputs/' + str(image_id) + '.png'))
        fig.savefig(save_path)
        
        
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
    
if __name__ == '__main__':
    main(sys.argv)