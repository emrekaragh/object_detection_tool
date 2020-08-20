#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

# %%
# Download the test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
# First we will download the images that we will use throughout this tutorial. The code snippet
# shown bellow will download the test images from the `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
# and save them inside the ``data/images`` folder.
import os
import sys
import argparse
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../','../'))
DATASET_DIR = os.path.abspath(os.path.join(__file__, '../','../','../','dataset'))

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def get_last_saved_model():
    directory = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs'))
    all_subdirs = [ f.path for f in os.scandir(directory) if f.is_dir() ]
    newest_subdir = max(all_subdirs, key=os.path.getmtime)
    saved_model_dir = os.path.abspath(os.path.join(directory, newest_subdir, 'saved_model'))
    return saved_model_dir 
    
def main(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir",help='Path to saved model file')
    ap.add_argument("-l", "--label_map",
        help="A .pbtxt file that contains all unique classes and their int map by given format")
    ap.add_argument("-i", "--images", default=os.path.join(DATASET_DIR,'test','images'),
         help="Path to images")
    ap.add_argument("-o", "--outputs", default=os.path.join(ROOT_DIR,'outputs'),
         help="Path to images")
    args = vars(ap.parse_args())
    
    
    PATH_TO_IMAGES_DIR = args['images']
    PATH_TO_MODEL_DIR = args['model_dir']
    PATH_TO_LABELS = args['label_map']
    PATH_TO_OUTPUTS = args['outputs']

    if PATH_TO_MODEL_DIR is None:
        PATH_TO_MODEL_DIR = get_last_saved_model()
        print('saved model:',PATH_TO_MODEL_DIR)
        PATH_TO_LABELS =  os.path.abspath(os.path.join(PATH_TO_MODEL_DIR, '../','ssd_label_map.pbtxt'))
    
    IMAGE_PATHS = sorted([f for f in glob.glob(PATH_TO_IMAGES_DIR + "**/*.jpg", recursive=True)])

    print('Loading model...', end='')
    start_time = time.time()
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
    
    counter = 0
    for image_path in IMAGE_PATHS:

        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=0,
              agnostic_mode=False)

        plt.figure()
        plt.imsave(fname = (PATH_TO_OUTPUTS+'/'+str(counter)+'.jpg'), arr = image_np_with_detections)
        counter += 1
        print('Done')


    # sphinx_gallery_thumbnail_number = 2

if (__name__ == '__main__'):
    # sys.argv.append('-d')
    main(sys.argv)
