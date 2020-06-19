# -*- coding: utf-8 -*-
"""
Created on Wed May 27 01:12:37 2020

@author: EmreKARA
"""
import argparse
import sys,os
from subprocess import run as spRun, check_output
import ast
import shutil

class DetectorHandler:
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.prediction_dataset = None
    def setTrainDataset(self,train_dataset):
        pass
    def setTestDataset(self,test_dataset):
        pass
    def setPredictionDataset(self,predict_dataset):
        pass
    def do_build_dataset(self, args=None):
        pass
    def do_train(self, args=None):
        pass
    def do_predict(self, args=None):
        pass
    def trainDefaultArgumentsSetter(self, args=None):
        pass
    def predictDefaultArgumentsSetter(self, args=None):
        pass
    
class MrcnnHandler(DetectorHandler):
    def __init__(self):
        DetectorHandler.__init__(self)
        self.PATH_TO_BUILD_DATASET = 'mrcnn\\own_dataset\\own_dataset_build_dataset.py'
        self.PATH_TO_TRAIN_PY = 'mrcnn\\own_dataset\\own_dataset_train.py'
        self.PATH_TO_DATASET = 'dataset'
        self.PATH_TO_PREDICT_PY = 'mrcnn\\own_dataset\\own_dataset_predict.py'
    def do_build_dataset(self, args):
        if args and not args:
            spRun(['python', self.PATH_TO_BUILD_DATASET])
        else:
            spRun(['python', self.PATH_TO_BUILD_DATASET, args])
    def do_predict(self, args=None):
        if args and not args:
            spRun(['python', self.PATH_TO_PREDICT_PY])
        else:
            spRun(['python', self.PATH_TO_PREDICT_PY, args])
    def do_train(self, args=None):
        print('You can find the config file from mrcnn/own_dataset/own_config.py that includes default train arguments assigning.')
        if args and args[0] == '':
            args = args[:-1]
        self.trainDefaultArgumentsSetter(args)
        train_check = spRun(['python', self.PATH_TO_TRAIN_PY,  args])
        try:
            train_check.check_returncode()
        except:
            print('From Tool: ATTENTION! \t mrcnn training was failure! Next processes may not work properly. ')
    def trainDefaultArgumentsSetter(self, args):
        if '--weights' not in args:
                args.insert(0, ('coco'))
                args.insert(0, ' --weights=')
        if '--dataset' not in args:
            args.insert(0, (self.PATH_TO_DATASET))
            args.insert(0, ' --dataset=')
        if 'train' not in args:
            args.insert(0, ' train')
class RetinaNetHandler(DetectorHandler):
    def __init__(self):
        DetectorHandler.__init__(self)
        self.PATH_TO_BUILD_DATASET = 'retinanet\\build_dataset.py'
        self.PATH_TO_RESNET50_WEİGHTS = 'retinanet\\resnet50_coco_best_v2.1.0.h5'
        self.PATH_TO_SNAPSHOTS = 'retinanet\\snapshots'
        self.PATH_TO_TENSORBOARDS = 'retinanet\\tensorboard'
        self.PATH_TO_TRAIN_CSV = 'dataset\\train\\train.csv'
        self.PATH_TO_TEST_CSV = 'dataset\\train\\test.csv'
        self.PATH_TO_CLASSES_CSV = 'dataset\\train\\classes.csv'
        self.PATH_TO_TEST_IMAGES = 'dataset\\test\\images'
        self.PATH_TO_BEST_MODEL = None
        self.PATH_TO_CONVERTED_MODEL = 'retinanet\\models\\output.h5'
        self.PATH_TO_PREDICT_PY = 'retinanet\\predict.py'
        self.PATH_TO_PREDICTION_OUTPUTS = 'retinanet\\outputs'
        
    def do_build_dataset(self, args):
        if args and not args:
            spRun(['python', self.PATH_TO_BUILD_DATASET])
            self.do_debug()
        else:
            spRun(['python', self.PATH_TO_BUILD_DATASET, args])
            self.do_debug()
    def do_train(self, args=None):
        if args and args[0] == '':
            args = args[:-1]
        self.trainDefaultArgumentsSetter(args)
        self.move_old_snapshots()
        train_check = spRun(['retinanet-train', args])
        try:
            train_check.check_returncode()
        except:
            print('From Tool: ATTENTION! \t retinanet training was failure! Next processes may not work properly. ')
        self.select_best_training_model()
        self.do_convert()
        self.do_evaluate()
    
    def do_predict(self, args=None):
        if args and args[0] == '':
            args = args[:-1]
        self.predictDefaultArgumentsSetter(args)
        spRun(['python', 'retinanet/predict.py', args])

    def do_convert(self, args=None):
        if args and args[0] == '':
            args = args[:-1]
        if not args:
            convert_check = spRun(['retinanet-convert-model', self.PATH_TO_BEST_MODEL , self.PATH_TO_CONVERTED_MODEL])
            try:
                convert_check.check_returncode()
            except:
                print('From Tool: ATTENTION! \t retinanet model converting was failure! Next processes may not work properly. ')
        else:
            spRun(['retinanet-convert-model', args])
    def do_evaluate(self, args=None):
        if args and args[0] == '':
            args = args[:-1]
        if not args:
            evaluate_check = spRun(['retinanet-evaluate', 'csv', self.PATH_TO_TRAIN_CSV ,self.PATH_TO_CLASSES_CSV, self.PATH_TO_CONVERTED_MODEL])
            try:
                evaluate_check.check_returncode()
            except:
                print('From Tool: ATTENTION! \t retinanet model evaluating was failure! Next processes may not work properly. ')
        else:
            spRun(['retinanet-evaluate', args])

    def do_debug(self, args=None):
        if args and args[0] == '':
            args = args[:-1]
        if not args:
            debug_check = spRun(['retinanet-debug', 'csv', self.PATH_TO_TRAIN_CSV ,self.PATH_TO_CLASSES_CSV])
            try:
                debug_check.check_returncode()
            except:
                print('From Tool: ATTENTION! \t retinanet model evaluating was failure! Next processes may not work properly. ')
        else:
            spRun(['retinanet-debug', args])
    def do_test(self, args):
        if args and args[0] == '':
            args = args[:-1]
        if not args:
            spRun(['retinanet-test'])
        else:
            spRun(['retinanet-test', args])

    def select_best_training_model(self):
        filelist= [file for file in os.listdir(self.PATH_TO_SNAPSHOTS) if file.endswith('.h5')]
        sorted_list = sorted(filelist)
        best_model = os.path.join(self.PATH_TO_SNAPSHOTS, sorted_list[-1])
        self.PATH_TO_BEST_MODEL = best_model
    
    def move_old_snapshots(self):
        filelist= [file for file in os.listdir(self.PATH_TO_SNAPSHOTS) if file.endswith('.h5')]
        dest = os.path.join(self.PATH_TO_SNAPSHOTS, 'olds')
        for file in filelist:
            src = os.path.join(self.PATH_TO_SNAPSHOTS, file)
            shutil.move(src, dest)

    def trainDefaultArgumentsSetter(self, args):
        if '--tensorboard-dir' not in args:
            args.insert(0, (' ' + self.PATH_TO_TENSORBOARDS +' ' ))
            args.insert(0, ' --tensorboard-dir')
        if '--snapshot-path' not in args:
            args.insert(0, (' ' + self.PATH_TO_SNAPSHOTS))
            args.insert(0, ' --snapshot-path')
        if '--weights' not in args and ' --backbone' not in args:
                args.insert(0, (' ' + self.PATH_TO_RESNET50_WEİGHTS))
                args.insert(0, ' --weights')
        if 'csv' not in args:
            args.append(' csv')
            args.append((' '+self.PATH_TO_TRAIN_CSV))
            args.append((' '+self.PATH_TO_CLASSES_CSV))
            args.append((' --val-annotations '+self.PATH_TO_TEST_CSV))
    def predictDefaultArgumentsSetter(self, args):
        if '--input' not in args:
            args.insert(0, (' ' + self.PATH_TO_TEST_IMAGES))
            args.insert(0, ' --input')
        if '--model' not in args:
            args.insert(0, (' ' + self.PATH_TO_CONVERTED_MODEL))
            args.insert(0, ' --model')
        if '--output' not in args:
            args.insert(0, (' ' + self.PATH_TO_PREDICTION_OUTPUTS))
            args.insert(0, ' --output')
        if '--labels' not in args:
            args.insert(0, (' ' + self.PATH_TO_CLASSES_CSV))
            args.insert(0, ' --labels')

class Invoker:    
    def retinanet(self, train_dataset, test_dataset, prediction_dataset, action, args):
        retina1 = RetinaNetHandler()
        retina1.setTrainDataset(train_dataset)
        retina1.setTestDataset(test_dataset)
        getattr(retina1, ('do_'+action))(args) #calling retina1.do_test(), retina1.do_test() or retina1.do_prediction() according to selected action
    
    def mrcnn(self, train_dataset, test_dataset, prediction_dataset, action, args):
        mrcnn1 = MrcnnHandler()
        mrcnn1.setTrainDataset(train_dataset)
        mrcnn1.setTestDataset(test_dataset)
        getattr(mrcnn1, ('do_'+action))(args) #calling retina1.do_test(), retina1.do_test() or retina1.do_prediction() according to selected action
    
    
    def invoke_detectors(self, args):
        """ Select detector.
        """
        parser = argparse.ArgumentParser(description='Script for object detection with vary models.', usage='choose a detector to give the arguments')
        subparsers = parser.add_subparsers(help='Arguments for specific detectors', dest='detector')
        subparsers.required = True
        
        retinanet_parser = subparsers.add_parser('retinanet')
        retinanet_parser.add_argument('action', help='Choose the action you want to do', choices=['build_dataset','train', 'convert', 'evaluate', 'predict','debug'])
        
        
        mrcnn_parser = subparsers.add_parser('mrcnn')
        mrcnn_parser.add_argument('action', help='Choose the action you want to do', choices=['build_dataset','train', 'predict'])
        
        ssd_parser = subparsers.add_parser('ssd')
        #ssd arguments
        
        parser.add_argument('--train_dataset', help='Path to a folder containing <images> and <annotations> folders (defaults to \'./dataset/train\')', default='./dataset/train')
        parser.add_argument('--test_dataset', help='Path to a folder containing <images> and <annotations> folders (defaults to \'./dataset/test\')', default='./dataset/test')
        parser.add_argument('--prediction-dataset', help='Path to a folder containing <images> folders (defaults to \'./dataset/prediction\')', default='./dataset/prediction')
        
        
        args, unknown = parser.parse_known_args()
        args_list = self.args2list(unknown)
        getattr(self, args.detector)(args.train_dataset, args.test_dataset, args.prediction_dataset, args.action, args_list)
    def args2list(self,args):
        args_str = str(args)
        args_str = args_str[1:-1]
        args_str = args_str.replace('\'','')
        args_list = args_str.split(',')
        return args_list
    
def main(args):
    invoker = Invoker()
    args = invoker.invoke_detectors(args)
    
if __name__ == '__main__':
    main(sys.argv)