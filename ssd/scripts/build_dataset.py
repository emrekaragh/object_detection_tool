# import the necessary packages
from bs4 import BeautifulSoup
from imutils import paths
import argparse
import random
import os, sys
import pandas as pd

import tensorflow as tf
from collections import namedtuple
import io
from PIL import Image

from object_detection.utils import dataset_util




import build_dataset_config as config

def to_pbtxt(classes, path):
    """ 
    Writes .pbtxt file that contains all unique classes and their int map by given format below:
    item {
        id: 1
        name: 'cat'
    }
    
    item {
        id: 2
        name: 'dog'
    }
    
    Parameters:
    ----------
    classes: {set}
        The set object that includes all unique class names
    path : {str}
        The path to write file
    """
    
    f = open(path, "w")
    counter = 1
    for name in classes:
        template_text = 'item {\n    id: '+ str(counter) +'\n    name: \''+ name +'\'\n}\n'
        f.write(template_text)
        counter += 1
    f.close()

def xml_to_csv(annot_path, images_path, train_csv, test_csv, classes_csv, num_classes, train_test_split, label_map):
    # grab all image paths then construct the training and testing split
    imagePaths = list(paths.list_files(images_path))
    random.shuffle(imagePaths)
    i = int(len(imagePaths) * train_test_split)
    testImagePaths = imagePaths[:i]
    trainImagePaths= imagePaths[i:]
    
    # create the list of datasets to build
    dataset = [ ("train", trainImagePaths, train_csv),
                ("test", testImagePaths, test_csv)]
    
    # initialize the set of classes we have
    CLASSES = set()
    
    
    head_tail = os.path.split(train_csv) 
    os.makedirs(head_tail[0], exist_ok=True)
    # loop over the datasets
    for (dType, imagePaths, outputCSV) in dataset:
        # load the contents
        print ("[INFO] creating '{}' set...".format(dType))
        print ("[INFO] {} total images in '{}' set".format(len(imagePaths), dType))
        
        xml_list = []
        # loop over the image paths
        for imagePath in imagePaths:
            # build the corresponding annotation path
            fname = imagePath.split(os.path.sep)[-1]
            img_name = fname
            fname = "{}.xml".format(fname[:fname.rfind(".")])
            annotPath = os.path.sep.join([annot_path, fname])
    
            # load the contents of the annotation file and buid the soup
            contents = open(annotPath).read()
            soup = BeautifulSoup(contents, "html.parser")
            
            # extract the image dimensions
            width = int(soup.find("width").string)
            height = int(soup.find("height").string)
    
            # loop over all object elements
            for o in soup.find_all("object"):
                #extract the label and bounding box coordinates
                label = o.find("name").string
                xMin = int(float(o.find("xmin").string))
                yMin = int(float(o.find("ymin").string))
                xMax = int(float(o.find("xmax").string))
                yMax = int(float(o.find("ymax").string))
    
                # truncate any bounding box coordinates that fall outside
                # the boundaries of the image
                xMin = max(0, xMin)
                yMin = max(0, yMin)
                xMax = min(width, xMax)
                yMax = min(height, yMax)
    
                # ignore the bounding boxes where the minimum values are larger
                # than the maximum values and vice-versa due to annotation errors
                if xMin >= xMax or yMin >= yMax:
                    continue
                elif xMax <= xMin or yMax <= yMin:
                    continue
    
                # write the image path, bb coordinates, label to the output CSV
                value = [str(img_name), str(width),str(height), str(label) ,str(xMin), str(yMin), str(xMax),
                        str(yMax)]
                CLASSES.add(label)
                xml_list.append(value)
            
        column_name = ['filename', 'width', 'height',
            'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv(outputCSV, index=None)
    
    # write the classes to file
    print("[INFO] writing classes...")
    classes_df = pd.DataFrame(list(CLASSES), columns =['label_name'])
    classes_df['label_index'] = range(1,(len(CLASSES)+1))
    classes_df.to_csv(classes_csv,index=False)
    to_pbtxt(CLASSES, label_map)
    
    num_classes = open(num_classes, "w")
    num_classes.write( str(len(CLASSES)) )
    num_classes.close()

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
def create_tf_example(group, path, classes_map):
#    print('\n\n\n\n\n\n\n\n\n\n Başladı \n\n\n\n\n\n\n')
    with tf.gfile.GFile(os.path.abspath(os.path.join(path, '{}'.format(group.filename)), 'rb')) as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(classes_map[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
def get_classes_map(path):
    f = open(path, "r")
    classes_map = {}
    f.readline()
    while True:
        line = f.readline()
        if line == '':
            break
        else: 
            line_list = line.split(',')
            classes_map[line_list[0]] = int(line_list[1][:-1])
    f.close()
    return classes_map

def csv_to_tfrecords(images_path , train_record, test_record, classes_csv, train_csv, test_csv):
    
    classes_map = get_classes_map(classes_csv)
    # create the list of datasets to create tfrecords both train and test
    dataset = []
    ##for train.record section
    csv_input = train_csv
    img_path = images_path
    output_path = train_record
    
    dataset.append((csv_input, img_path, output_path))
    
    ##for test.record section
    csv_input = test_csv
    img_path = images_path #This is for evaluate images in train folder
    output_path = test_record
    
    dataset.append((csv_input, img_path, output_path))
    
    for (csv_input, img_path, output_path) in dataset:
        writer = tf.python_io.TFRecordWriter(output_path)
        path = os.path.abspath(os.path.join(os.getcwd(), img_path))
        examples = pd.read_csv(csv_input)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path, classes_map)
            writer.write(tf_example.SerializeToString())
    
        writer.close()
        final_output_path = os.path.abspath(os.path.join(os.getcwd(), output_path))
        print('Successfully created the TFRecords: {}'.format(final_output_path))
    pass
    
    
    
def build(args):
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-d", "--dataset",action="store_true"
    #     help='path to dataset. if this option is specified then you must pass others too')
    ap.add_argument("-a", "--annotations", default=config.ANNOT_PATH,
        help='path to annotations')
    ap.add_argument("-i", "--images", default=config.IMAGES_PATH,
     	help="path to images")
    ap.add_argument("-t", "--train", default=config.TRAIN_CSV,
     	help="path to output training CSV file")
    ap.add_argument("-e", "--test", default=config.TEST_CSV,
     	help="path to output test CSV file")
    ap.add_argument("-c", "--classes", default=config.CLASSES_CSV,
     	help="path to output classes CSV file")
    ap.add_argument("--num_classes", default=config.NUM_CLASSES_TXT,
     	help="path to number of classes txt file")
    ap.add_argument("-s", "--split", type=float, default=config.TRAIN_TEST_SPLIT,
     	help="train and test split")
    ap.add_argument("-l", "--labelMap", default = config.LABEL_MAP_PBTXT,
        help="A .pbtxt file that contains all unique classes and their int map by given format")
    ap.add_argument("--trainRecord", default = config.TRAIN_TFRECORD,
        help="A .record file contains tfrecords for train set")
    ap.add_argument("--testRecord", default = config.TEST_TFRECORD,
        help="A .record file contains tfrecords for test set")

    args = vars(ap.parse_args())
    
    # Create easy variable names for all the arguments
    # dataset = args["dataset"]
    annot_path = args["annotations"]
    images_path = args["images"]
    train_csv = args["train"]
    test_csv = args["test"]
    classes_csv = args["classes"]
    num_classes = args["num_classes"]
    train_test_split = args["split"]
    label_map = args["labelMap"]
    train_record = args["trainRecord"]
    test_record = args["testRecord"]
    
    xml_to_csv(annot_path, images_path, train_csv, test_csv, classes_csv, num_classes, train_test_split, label_map)
    csv_to_tfrecords(images_path , train_record, test_record, classes_csv, train_csv, test_csv)
    
    print('\n\n Dataset Succesfully builded. You can go train step. \n')
    

if (__name__ == '__main__'):
    build(sys.argv)