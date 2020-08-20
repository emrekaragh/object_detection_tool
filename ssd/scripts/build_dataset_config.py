# import the necessary packages
import os

# initialize the base path for the logos dataset
BASE_PATH = os.path.abspath(os.path.join(__file__,'../','../','../','dataset','train'))

# build the path to the annotations and input images
ANNOT_PATH = os.path.sep.join([BASE_PATH, 'annotations'])
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])

# degine the training/testing split
TRAIN_TEST_SPLIT = 0.0

#  build the path to the output training and test .csv files
TRAIN_CSV = os.path.sep.join([BASE_PATH, 'ssd_train.csv'])
TEST_CSV = os.path.sep.join([BASE_PATH, 'ssd_test.csv'])

# build the path to the output classes CSV files
CLASSES_CSV = os.path.sep.join([BASE_PATH, 'ssd_classes.csv'])
NUM_CLASSES_TXT = os.path.sep.join([BASE_PATH, 'ssd_num_classes.txt'])

# build the path to the output classes pbtxt file
LABEL_MAP_PBTXT = os.path.sep.join([BASE_PATH, 'ssd_label_map.pbtxt'])

#build the path to the TFRECORDS files
TRAIN_TFRECORD = os.path.sep.join([BASE_PATH, 'ssd_train.record'])
TEST_TFRECORD = os.path.sep.join([BASE_PATH, 'ssd_test.record'])

# build the path to the output predictions dir
OUTPUT_DIR = os.path.sep.join([BASE_PATH, 'predictions'])
