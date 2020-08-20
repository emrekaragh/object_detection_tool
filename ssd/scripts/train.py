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



ROOT_DIR = os.path.abspath(os.path.join(__file__, '../','../'))
DATASET_DIR = os.path.abspath(os.path.join(__file__, '../','../','../','dataset'))
PATH_TO_MODEL_MAIN_PY = os.path.abspath(os.path.join(__file__, '../','model_main.py'))
PATH_TO_EXPORT_INFERENCE_GRAPHPY = os.path.abspath(os.path.join(__file__, '../','export_inference_graph.py'))
print('ROOT_DIR:',ROOT_DIR)

def get_and_uncompress(config_file_name, target_path_parent):
    urls = {'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config': 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
               'ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync.config': 'http://download.tensorflow.org/models/object_detection/ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz', 
               'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
        }
    start_time = time.time()
    url =urls[config_file_name]
    
    target_path = os.path.abspath(os.path.join(target_path_parent, url[url.index('ssd'):]))
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    fname = target_path
    
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path = target_path_parent)
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path = target_path_parent)
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

def get_fine_tune_checkpoint(config_file_name, check_points_path):
    print('get_fine_tune_checkpoint():', config_file_name)
    sub_dirs  = next(os.walk(check_points_path))[1]
    check_list = [file.startswith(config_file_name[:config_file_name.index('.')]) for file in sub_dirs]
    if True in check_list:
        file_index = check_list.index(True)
        check_points_file = os.path.abspath(os.path.join(check_points_path, sub_dirs[file_index], 'model.ckpt'))
        return check_points_file
    else:
        print('Pre trained files doesnt exist. Downloading...')
        get_and_uncompress(config_file_name, check_points_path)
        return get_fine_tune_checkpoint(config_file_name, check_points_path)



def edit_config(config_index, batch_size=None, fine_tune_checkpoint=True):
    configs_path_default = os.path.abspath(os.path.join(ROOT_DIR, 'training', 'configs','defaults'))
    configs_path_used = os.path.abspath(os.path.join(ROOT_DIR, 'training', 'configs'))
    configs = ['ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config',
               'ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync.config', 
               'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config',]
    
    
    
    config_file_default = configs[config_index]
    print('config_file_default:', config_file_default)
    if config_file_default in os.listdir(configs_path_default):
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        
        with tf.gfile.GFile(os.path.abspath(os.path.join(configs_path_default, config_file_default)), "r") as f:   
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)
        
        
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_test.record')) # https://stackoverflow.com/questions/55323907/dynamically-editing-pipeline-config-for-tensorflow-object-detection
        pipeline_config.eval_input_reader[0].label_map_path = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_label_map.pbtxt'))         # https://stackoverflow.com/questions/55323907/dynamically-editing-pipeline-config-for-tensorflow-object-detection
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_train.record'))
        pipeline_config.train_input_reader.label_map_path = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_label_map.pbtxt'))
        pipeline_config.model.ssd.num_classes = get_num_class(os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_num_classes.txt')))
        pipeline_config.train_config.num_steps = 500 #for trying
        
        if batch_size:
            pipeline_config.train_config.batch_size = batch_size
        if fine_tune_checkpoint:
            pipeline_config.train_config.fine_tune_checkpoint = get_fine_tune_checkpoint(config_file_default, os.path.abspath(os.path.join(ROOT_DIR, 'pre-trained-models')))
        
        
        
        config_text = text_format.MessageToString(pipeline_config)
        
        config_file_new = os.path.abspath(os.path.join(configs_path_used, config_file_default))
        with tf.gfile.Open(config_file_new, "wb") as f:
            f.write(config_text)
        return config_file_new, pipeline_config.train_config.num_steps
    else:
        print(config_file_default,('is not in directory, supported files are below:'
                                   '\nssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config'
                                   '\nssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync.config'
                                   '\nssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config'))
        sys.exit()
        
    
def main(args):
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-c","--configFile", default = 0,type=int,
        choices = [0,1,2,3],
        help=("A .config file contains train-evaluate-prediction configurations based selected backbone"
        "\n0: ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config (DEFAULT)"
        "\n1: ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync.config"
        "\n2: ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config")
        )
    ap.add_argument("-b", "--batch_size",type=int,
        help='optional batch_size parameter for training. (Higher values require more memory and vice-versa)')
    ap.add_argument("-d", "--dont_use_checkpoint",action='store_false',
        help='If this parameter is passed. Training process does not use pre-trained weight, it starts from scratch')
    
    args = vars(ap.parse_args())
    
    config_file = args["configFile"]
    batch_size = args["batch_size"]
    dont_use_checkpoint = args["dont_use_checkpoint"]
    config_file_new, num_steps_train = edit_config(config_file, batch_size, dont_use_checkpoint)
    
    
    ### call model_main.py with args for training
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = os.path.abspath(os.path.join(ROOT_DIR, 'training',dt_string))
    os.makedirs(os.path.abspath(os.path.join(model_dir, 'export', 'Servo')))
    model_dir_str = '\"'+str(model_dir)+'/\"'
    config_file_new_str = '\"'+str(config_file_new)+'\"'
    args_for_model_main = ['--alsologtostderr', ' --model_dir='+model_dir_str, (' --pipeline_config_path='+config_file_new_str)]
    spRun(['python', PATH_TO_MODEL_MAIN_PY, args_for_model_main])
    
    ### call export_inference_graph.py with args to prepare model for prediction 
    model_ckpt = os.path.abspath(os.path.join(model_dir, ('model.ckpt-'+str(num_steps_train))))
    inference_graph_dir = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs',dt_string))
    os.makedirs(inference_graph_dir)
    inference_graph_path = os.path.abspath(os.path.join(inference_graph_dir, 'output_inference_graph_v1.pb'))
    inference_graph_path_str = '\"'+str(inference_graph_path)+'\"' 
    model_ckpt_str = '\"'+str(model_ckpt)+'\"' 
    args_for_inference_graph = ['--input_type=image_tensor', (' --pipeline_config_path='+config_file_new_str), (' --trained_checkpoint_prefix='+model_ckpt_str), (' --output_directory='+inference_graph_path_str)]
    spRun(['python', PATH_TO_EXPORT_INFERENCE_GRAPHPY, args_for_inference_graph])
    
    label_map_src = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_label_map.pbtxt'))
    label_map_dst = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs',dt_string,'ssd_label_map.pbtxt'))
    copy2(label_map_src, label_map_dst)
    
    num_classes_src = os.path.abspath(os.path.join(DATASET_DIR, 'train','ssd_num_classes.txt'))
    num_classes_dst = os.path.abspath(os.path.join(ROOT_DIR, 'trained-inference-graphs',dt_string,'ssd_num_classes.txt'))
    copy2(num_classes_src, num_classes_dst)
    
    


if (__name__ == '__main__'):
    # sys.argv.append('-d')
    main(sys.argv)
    
    