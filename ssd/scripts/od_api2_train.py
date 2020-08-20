# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:24:10 2020

@author: EmreKARA
"""
import argparse
import sys
import os

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

import requests
import tarfile
from subprocess import run as spRun
import time
from datetime import datetime
from shutil import copy2
import json


ROOT_DIR = os.path.abspath(os.path.join(__file__, '../','../'))
DATASET_DIR = os.path.abspath(os.path.join(__file__, '../','../','../','dataset'))
PATH_TO_MODEL_MAIN_PY = os.path.abspath(os.path.join(__file__, '../','model_main_tf2.py'))
PATH_TO_EXPORT_INFERENCE_GRAPHPY = os.path.abspath(os.path.join(__file__, '../','exporter_main_v2.py'))
print('ROOT_DIR:',ROOT_DIR)

tf.compat.v1.disable_eager_execution()


    
def download_od2_model(model_name, models_dir):
    od2_models_dict_dir =  os.path.abspath(os.path.join(__file__, '../','od2_models_dict.json'))
    od2_models_dict = open(od2_models_dict_dir, 'r')
    urls = json.load(od2_models_dict)
    od2_models_dict.close()
    url = urls[model_name]
    
    start_time = time.time()
    print('Download started...')
    
    file_name = model_name + '.tar.gz'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.raw.read())
    
    tar = tarfile.open(file_name, "r:gz")
    tar.extractall(path = models_dir)
    tar.close()

    print('Download and uncompressing is Succesfully done in',end=' ')
    print("%s seconds" % (time.time() - start_time))
                                        
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

def get_pre_trained_model(model_name, pre_trained_models_dir):
    print('get_pre_trained_model():', model_name)
    sub_dirs  = next(os.walk(pre_trained_models_dir))[1]
    check_list = [file.startswith(model_name) for file in sub_dirs]
    if True in check_list:
        file_index = check_list.index(True)
        pre_trained_model = os.path.abspath(os.path.join(pre_trained_models_dir, sub_dirs[file_index], 'checkpoint','ckpt-0'))
        return pre_trained_model
    else:
        print('Pre trained files doesnt exist. Downloading...')
        model_dir = os.path.abspath(os.path.join(pre_trained_models_dir))
        download_od2_model(model_name, model_dir)
        return get_pre_trained_model(model_name, pre_trained_models_dir)



def edit_config(model_name, pre_trained_models_dir, pre_trained_model, batch_size=None, fine_tune_checkpoint=True):
    config_file_name = model_name + '.config'
    config_file = os.path.abspath(os.path.join(ROOT_DIR, 'training', 'configs', (model_name+'.config')))
    """
    print('config_file:', config_file)
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_file, "r") as f: 
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_test.record')) # https://stackoverflow.com/questions/55323907/dynamically-editing-pipeline-config-for-tensorflow-object-detection
    pipeline_config.eval_input_reader[0].label_map_path = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_label_map.pbtxt'))         # https://stackoverflow.com/questions/55323907/dynamically-editing-pipeline-config-for-tensorflow-object-detection
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_train.record'))
    pipeline_config.train_input_reader.label_map_path = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_label_map.pbtxt'))
    #pipeline_config.model.ssd.num_classes = get_num_class(os.path.abspath(os.path.join(DATASET_DIR, 'train','od2_num_classes.txt')))
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_config.use_bfloat16 = False
    pipeline_config.train_config.num_steps = 500 #for trying
    
    if batch_size:
        pipeline_config.train_config.batch_size = batch_size
    if fine_tune_checkpoint:
        pipeline_config.train_config.fine_tune_checkpoint = pre_trained_model
    
    config_text = text_format.MessageToString(pipeline_config)
    config_file_new = os.path.abspath(os.path.join(ROOT_DIR, 'training', 'configs', config_file_name))
    
    with tf.io.gfile.GFile(config_file_new, "wb") as f:
        f.write(config_text)
    """
    config_file_new = os.path.abspath(os.path.join(ROOT_DIR, 'training', 'configs', config_file_name))
    ## set num classes
    num_classes = get_num_class(os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_num_classes.txt')))
    file = open(config_file_new,'r')
    old_string = file.read()
    sub_index = old_string.find('num_classes: ')
    sub_index += 13
    new_line_index =  old_string.find('\n', sub_index)
    new_string =  old_string[:sub_index] + str(num_classes) + old_string[new_line_index:]
    file.close()
    
    file = open(config_file_new,'w')
    file.write(new_string)
    file.close()
    
    return config_file
        
    
def main(args):
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-m","--model", default = 0,type=int,
        choices = [0,1,2,3,4,5],
        help=("Model to be trained"
        "\n0: ssd_mobilenet_v2_320x320_coco17_tpu-8 (DEFAULT)"
        "\n1: efficientdet_d7_coco17_tpu-32"
        "\n2: centernet_resnet50_v1_fpn_512x512_coco17_tpu-8"
        "\n3: mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8"
        "\n4: ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8"
        "\n5: faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8")
        )
    ap.add_argument("-b", "--batch_size",type=int,
        help='optional batch_size parameter for training. (Higher values require more memory and vice-versa)')
    ap.add_argument("-d", "--dont_use_checkpoint",action='store_false',
        help='If this parameter is passed. Training process does not use pre-trained weight, it starts from scratch')
    
    args = vars(ap.parse_args())
    
    model_index = args["model"]
    batch_size = args["batch_size"]
    dont_use_checkpoint = args["dont_use_checkpoint"]
    
    models = ['ssd_mobilenet_v2_320x320_coco17_tpu-8','efficientdet_d7_coco17_tpu-32', 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8','mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8','ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8', 'faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8']
    model_name = models[model_index]
    
    pre_trained_models_dir = os.path.abspath(os.path.join(ROOT_DIR, 'pre-trained-models'))
    pre_trained_model = get_pre_trained_model(model_name, pre_trained_models_dir)
    config_file_new = edit_config(model_name, pre_trained_models_dir, pre_trained_model, batch_size, dont_use_checkpoint)
    
    
    ### call model_main.py with args for training
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = os.path.abspath(os.path.join(ROOT_DIR, 'training',dt_string))
    os.makedirs(os.path.abspath(os.path.join(model_dir, 'export', 'Servo')))
    args_for_model_main = ['--alsologtostderr', '--model_dir='+model_dir, ('--pipeline_config_path='+config_file_new)]
    spRun((['python', PATH_TO_MODEL_MAIN_PY] + args_for_model_main))

    print('\n\nmodel_main_tf_2.py is Done...\n\n')
    
    ### call export_inference_graph.py with args to prepare model for prediction 
    model_ckpt = os.path.abspath(os.path.join(model_dir))
    inference_graph_dir = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs',dt_string))
    os.makedirs(inference_graph_dir)
    inference_graph_path = os.path.abspath(os.path.join(inference_graph_dir))
    args_for_inference_graph = ['--input_type=image_tensor', ('--pipeline_config_path='+config_file_new), ('--trained_checkpoint_dir='+model_ckpt), ('--output_directory='+inference_graph_path)]
    print('args_for_inference_graph:', args_for_inference_graph)
    spRun((['python', PATH_TO_EXPORT_INFERENCE_GRAPHPY]+ args_for_inference_graph))
    
    label_map_src = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_label_map.pbtxt'))
    label_map_dst = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs',dt_string,'ssd_label_map.pbtxt'))
    copy2(label_map_src, label_map_dst)
    
    num_classes_src = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_num_classes.txt'))
    num_classes_dst = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs',dt_string,'ssd_num_classes.txt'))
    copy2(num_classes_src, num_classes_dst)
    
    


if (__name__ == '__main__'):
    # sys.argv.append('-d')
    main(sys.argv)
    
    
