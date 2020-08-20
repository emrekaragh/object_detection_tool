import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(__file__,'../','../'))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree
import datetime
import numpy as np
import skimage.draw
from bs4 import BeautifulSoup

from own_config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn import model as modellib, utils
from mrcnn.model import MaskRCNN


###### Sadece deneme sürümü için
import pandas as pd

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.abspath(os.path.join(ROOT_DIR, "mask_rcnn_coco.h5"))

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR =  os.path.abspath(os.path.join(ROOT_DIR, "logs"))

class OwnDatasetConfig(Config):
    NAME = "OwnDataset"
class OwnDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset, annnotations_csv=None):
        # adding all unique classes which in our dataset
        self.add_all_classes(dataset_dir)
        
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        print('dataset_dir:', dataset_dir)
        
        #set images an annotations paths
        images_dir = os.path.abspath(os.path.join(dataset_dir, 'images'))
        annotations_dir = os.path.abspath(os.path.join(dataset_dir, 'annotations'))
        
        #Iterate through all files in the folder
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            
            # setting image file
            img_path = os.path.abspath(os.path.join(images_dir, filename))
            # setting annotations file
            
            ann_path = annnotations_csv
            
            self.add_image('own_dataset', image_id=image_id, path=img_path, annotation=ann_path)
    def add_all_classes(self, dataset_dir):
        csv_path = os.path.abspath(os.path.join(dataset_dir, 'train','mrcnn_classes.csv'))
        from csv import reader
        with open(csv_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                self.add_class('own_dataset', (int(row[1]) + 1), row[0]) #(int(row[1]) + 1) 0. index reserverd for BG class (I Think!)
    def extract_boxes(self, filename):
        
        df = pd.read_csv(filename)
        
        width = df['width'].values[0]
        height = df['height'].values[0]
        boxes = []
        coors = df[['xmin','ymin','xmax','ymax','class']].values
        
        for row in coors:
            elements = []
            for element in row:
                elements.append(element)
            boxes.append(elements)
            
        return boxes, width, height
    def load_mask(self, image_id):
        
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        
        # load XML
        boxes, w, h = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(box[4]))
        return masks, asarray(class_ids, dtype='int32')
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = OwnDataset()
    dataset_train.load_dataset(args.dataset, "train", annnotations_csv = args.annnotations_csv)
    dataset_train.prepare()

    # Validation dataset
    dataset_test = OwnDataset()
    dataset_test.load_dataset(args.dataset, "test", annnotations_csv = args.annnotations_csv)
    dataset_test.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_test,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
    

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect your Own Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/your/dataset/",
                        help='Directory of the Your dataset. Should includes images,annotations folders and mrcnn_num_classes.txt(its should includes count of distinct class in your dataset)')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('-a','--annnotations_csv', required=False,
                        metavar="path to your annotations csv",
                        help='path to your annotations csv')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
               
    num_classes_from_dataset = 0
    try:
        f = open(os.path.abspath(os.path.join(args.dataset,'train','mrcnn_num_classes.txt')), "r")
        num_classes_from_dataset = int(f.read()) +1  # +1 is for Back Ground
        f.close()
        OwnDatasetConfig.NUM_CLASSES = num_classes_from_dataset
    except:
        print('Please run build_dataset to create num_classes.txt')
        sys.exit()
               
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    

    # Configurations
    if args.command == "train":
        config = OwnDatasetConfig()
    else:
        class InferenceConfig(OwnDatasetConfig()):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
        
    
    #For prediction
    num_classes_path = os.path.abspath(os.path.join(model.log_dir, "num_classes.txt"))
    num_classes = open(num_classes_path, "w")
    num_classes.write(str(num_classes_from_dataset - 1)) # -1 is Back Ground Class
    num_classes.close()