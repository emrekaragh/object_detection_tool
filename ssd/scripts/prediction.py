# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:30:23 2020

@author: EmreKARA
"""
import numpy as np
import os
import sys
import tensorflow as tf



from matplotlib import pyplot as plt
from PIL import Image


from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

###
import glob
import argparse

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../','../'))
DATASET_DIR = os.path.abspath(os.path.join(__file__, '../','../','../','dataset'))

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_num_class(num_classes_txt_path):
    try:
        f = open(num_classes_txt_path, "r")
        num_classes_from_dataset = int(f.read())
        f.close()
        return num_classes_from_dataset
    except Exception as ex:
        print('Please run build_dataset to create num_classes.txt')
        print('ex:', ex)
        sys.exit()

def get_last_frozen_inference_graph():
    directory = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs'))
    all_subdirs = [ f.path for f in os.scandir(directory) if f.is_dir() ]
    newest_subdir = max(all_subdirs, key=os.path.getmtime)
    inference_graph = os.path.abspath(os.path.join(directory, newest_subdir, 'output_inference_graph_v1.pb','frozen_inference_graph.pb'))
    return inference_graph
    

def main(args):
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-f", "--frozen_inference_graph",
        help='Path to frozen_inference_graph .pb file')
    ap.add_argument("-l", "--label_map",
        help="A .pbtxt file that contains all unique classes and their int map by given format")
    ap.add_argument("-n","--num_classes",
     	help="Path to number of classes txt file")
    ap.add_argument("-i", "--images", default=os.path.join(DATASET_DIR,'test','images'),
     	help="Path to images")
    args = vars(ap.parse_args())
    
    
    PATH_TO_TEST_IMAGES_DIR = args['images']
    PATH_TO_CKPT = args['frozen_inference_graph']
    num_classes = args['num_classes']
    PATH_TO_LABELS = args['label_map']


    if PATH_TO_CKPT is None:
        PATH_TO_CKPT = get_last_frozen_inference_graph()
        print('frozen_inference_graph:',PATH_TO_CKPT)
        num_classes = os.path.abspath(os.path.join(PATH_TO_CKPT, '../','../','ssd_num_classes.txt'))
        NUM_CLASSES = get_num_class(num_classes)
        PATH_TO_LABELS =  os.path.abspath(os.path.join(PATH_TO_CKPT, '../','../','ssd_label_map.pbtxt'))
    
    print('frozen_inference_graph:',PATH_TO_CKPT)
    print('NUM_CLASSES:',NUM_CLASSES)
    print('label_map:',PATH_TO_LABELS)
    print('PATH_TO_TEST_IMAGES_DIR:',PATH_TO_TEST_IMAGES_DIR)
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
        
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    TEST_IMAGE_PATHS = sorted([f for f in glob.glob(PATH_TO_TEST_IMAGES_DIR + "**/*.jpg", recursive=True)])
    
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    
    counter = 0
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      # plt.imshow(image_np)
      outputs = os.path.join(ROOT_DIR,'outputs')
      plt.imsave(fname = (outputs+'/'+str(counter)+'.jpg'), arr = image_np)
      counter += 1

if (__name__ == '__main__'):
    # sys.argv.append('-d')
    main(sys.argv)