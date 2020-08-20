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
from time import sleep

class DetectorHandler:
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.prediction_dataset = None
    def setTrainDataset(self,train_dataset):
        pass
    def setTestDataset(self,test_dataset):
        pass
    def setPredictionDataset(self,predict_dataset): pass
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

class SSDHandler(DetectorHandler):
    def __init__(self):
        DetectorHandler.__init__(self)
        self.PATH_TO_BUILD_DATASET = os.path.abspath(os.path.join(__file__,'../','ssd','scripts','build_dataset.py'))
        self.PATH_TO_TRAIN_PY = os.path.abspath(os.path.join(__file__,'../','ssd','scripts','od_api2_train.py'))
        self.PATH_TO_PREDICT_PY = os.path.abspath(os.path.join(__file__,'../','ssd','scripts','plot_object_detection_saved_model.py'))
    def do_build_dataset(self, args):
        if not args:
            spRun(['python', self.PATH_TO_BUILD_DATASET])
        else:
            spRun((['python', self.PATH_TO_BUILD_DATASET]+ args))
    def do_predict(self, args=None):
        if not args:
            spRun(['python', self.PATH_TO_PREDICT_PY])
        else:
            spRun((['python', self.PATH_TO_PREDICT_PY] +  args))
    def do_train(self, args=None):
        print('You can find the config file from ssd/training/configs/default that includes default train arguments assigning.')
        self.trainDefaultArgumentsSetter(args)
        train_check = spRun((['python', self.PATH_TO_TRAIN_PY] +  args))
        try:
            train_check.check_returncode()
        except:
            print('From Tool: ATTENTION! \t retinanet training was failure! Next processes may not work properly. ')
    def trainDefaultArgumentsSetter(self, args):
        if not args:
            args = []

class MrcnnHandler(DetectorHandler):
    def __init__(self):
        DetectorHandler.__init__(self)
        self.PATH_TO_BUILD_DATASET =  os.path.abspath(os.path.join(__file__,'../','mrcnn','own_dataset','own_dataset_build_dataset.py'))
        self.PATH_TO_TRAIN_PY = os.path.abspath(os.path.join(__file__,'../','mrcnn','own_dataset','own_dataset_train.py'))
        self.PATH_TO_DATASET = os.path.abspath(os.path.join(__file__,'../','dataset'))
        self.PATH_TO_PREDICT_PY = os.path.abspath(os.path.join(__file__,'../','mrcnn','own_dataset','own_dataset_predict.py'))
    def do_build_dataset(self, args):
        if not args:
            spRun(['python', self.PATH_TO_BUILD_DATASET])
        else:
            spRun((['python', self.PATH_TO_BUILD_DATASET] + args))
    def do_predict(self, args=None):
        if not args:
            spRun(['python', self.PATH_TO_PREDICT_PY])
        else:
            spRun((['python', self.PATH_TO_PREDICT_PY] + args))
    def do_train(self, args=None):
        print('You can find the config file from mrcnn/own_dataset/own_config.py that includes default train arguments assigning.')
        args = self.trainDefaultArgumentsSetter(args)
        train_check = spRun((['python', self.PATH_TO_TRAIN_PY] + args))
        try:
            train_check.check_returncode()
        except:
            print('From Tool: ATTENTION! \t retinanet training was failure! Next processes may not work properly. ')
    def trainDefaultArgumentsSetter(self, args):
        if not args:
            args = []
        if '--weights' not in args:
            args.insert(0, '--weights=coco')
        if '--dataset' not in args:
            args.insert(0, ('--dataset='+self.PATH_TO_DATASET))
        if 'train' not in args:
            args.insert(0, 'train')
        return args
class RetinaNetHandler(DetectorHandler):
    def __init__(self):
        DetectorHandler.__init__(self)
        self.PATH_TO_BUILD_DATASET = os.path.abspath(os.path.join(__file__,'../','retinanet','build_dataset.py'))
        self.PATH_TO_RESNET50_WEIGHTS = os.path.abspath(os.path.join(__file__,'../','retinanet','resnet50_coco_best_v2.1.0.h5'))
        self.PATH_TO_SNAPSHOTS = os.path.abspath(os.path.join(__file__,'../','retinanet','snapshots'))
        self.PATH_TO_TENSORBOARDS = os.path.abspath(os.path.join(__file__,'../','retinanet','tensorboard'))
        self.PATH_TO_TRAIN_CSV = os.path.abspath(os.path.join(__file__,'../','dataset','train','train.csv'))
        self.PATH_TO_TEST_CSV = os.path.abspath(os.path.join(__file__,'../','dataset','train','test.csv'))
        self.PATH_TO_CLASSES_CSV = os.path.abspath(os.path.join(__file__,'../','dataset','train','classes.csv'))
        self.PATH_TO_TEST_IMAGES = os.path.abspath(os.path.join(__file__,'../','dataset','test','images'))
        self.PATH_TO_BEST_MODEL = None
        self.PATH_TO_CONVERTED_MODEL = os.path.abspath(os.path.join(__file__,'../','retinanet','models','output.h5'))
        self.PATH_TO_PREDICT_PY = os.path.abspath(os.path.join(__file__,'../','retinanet','predict.py'))
        self.PATH_TO_PREDICTION_OUTPUTS = os.path.abspath(os.path.join(__file__,'../','retinanet','outputs'))

    def do_build_dataset(self, args):
        if not args:
            spRun(['python', self.PATH_TO_BUILD_DATASET])
            print('dataset succesfully builded')
            self.do_debug()
        else:
            spRun((['python', self.PATH_TO_BUILD_DATASET] + args))
            print('dataset succesfully builded')
            self.do_debug()
    def do_train(self, args=None):
        args = self.trainDefaultArgumentsSetter(args)
        self.move_old_snapshots()
        train_check = spRun((['retinanet-train']  + args))
        try:
            train_check.check_returncode()
        except:
            print('From Tool: ATTENTION! \t retinanet training was failure! Next processes may not work properly. ')
        self.select_best_training_model()
        self.do_convert()
        self.do_evaluate()

    def do_predict(self, args=None):
        args = self.predictDefaultArgumentsSetter(args)
        spRun((['python', 'retinanet/predict.py'] + args))

    def do_convert(self, args=None):
        if not args:
            convert_check = spRun(['retinanet-convert-model', self.PATH_TO_BEST_MODEL , self.PATH_TO_CONVERTED_MODEL])
            try:
                convert_check.check_returncode()
            except:
                print('From Tool: ATTENTION! \t retinanet model converting was failure! Next processes may not work properly. ')
        else:
            spRun((['retinanet-convert-model'] + args))
    def do_evaluate(self, args=None):
        if not args:
            evaluate_check = spRun(['retinanet-evaluate', 'csv', self.PATH_TO_TRAIN_CSV ,self.PATH_TO_CLASSES_CSV, self.PATH_TO_CONVERTED_MODEL])
            try:
                evaluate_check.check_returncode()
            except:
                print('From Tool: ATTENTION! \t retinanet model evaluating was failure! Next processes may not work properly. ')
        else:
            spRun((['retinanet-evaluate'] + args))

    def do_debug(self, args=None):
        if not args:
            debug_check = spRun(['retinanet-debug', 'csv', self.PATH_TO_TRAIN_CSV ,self.PATH_TO_CLASSES_CSV])
            try:
                debug_check.check_returncode()
            except:
                print('From Tool: ATTENTION! \t retinanet model evaluating was failure! Next processes may not work properly. ')
        else:
            spRun((['retinanet-debug'] + args))
    def do_test(self, args):
        if not args:
            spRun(['retinanet-test'])
        else:
            spRun((['retinanet-test']  + args))

    def select_best_training_model(self):
        filelist= [file for file in os.listdir(self.PATH_TO_SNAPSHOTS) if file.endswith('.h5')]
        sorted_list = sorted(filelist)
        best_model = os.path.join(self.PATH_TO_SNAPSHOTS, sorted_list[-1])
        self.PATH_TO_BEST_MODEL = best_model

    def move_old_snapshots(self):
        dest = os.path.join(self.PATH_TO_SNAPSHOTS, 'olds')
        filelist= [file for file in os.listdir(self.PATH_TO_SNAPSHOTS) if file.endswith('.h5')]
        filelistOlds= [file for file in os.listdir(dest) if file.endswith('.h5')]
        common_items = list(set(filelist).intersection(filelistOlds))
        if common_items:
            print('From Tool: ATTENTION! \t Stop this process within 3 seconds. If you don\'t want these files to be overwritten:')
            print(common_items)
            for i in range(3):
                sleep(1)
            for item in common_items:
                os.remove(os.path.join(dest, item))
        for file in filelist:
            src = os.path.join(self.PATH_TO_SNAPSHOTS, file)
            shutil.move(src, dest)

    def trainDefaultArgumentsSetter(self, args):
        if not args:
            args = []
        if '--tensorboard-dir' not in args:
            args.insert(0, ('--tensorboard-dir='+self.PATH_TO_TENSORBOARDS))
        if '--snapshot-path' not in args:
            args.insert(0, ('--snapshot-path='+self.PATH_TO_SNAPSHOTS))
        if '--weights' not in args and ' --backbone' not in args:
            args.insert(0, ('--weights='+self.PATH_TO_RESNET50_WEIGHTS))
        if 'csv' not in args:
            args.append('csv')
            args.append(self.PATH_TO_TRAIN_CSV)
            args.append(self.PATH_TO_CLASSES_CSV)
            args.append(('--val-annotations='+self.PATH_TO_TEST_CSV))
        print('args_of_reti_for_train:\n',args)
        return args
    def predictDefaultArgumentsSetter(self, args):
        if not args:
            args = []
        if '--input' not in args:
            args.insert(0, (' --input='+self.PATH_TO_TEST_IMAGES))
        if '--model' not in args:
            args.insert(0, (' --model='+self.PATH_TO_CONVERTED_MODEL))
        if '--output' not in args:
            args.insert(0, (' --output='+ self.PATH_TO_PREDICTION_OUTPUTS))
        if '--labels' not in args:
            args.insert(0, (' --labels='+ self.PATH_TO_CLASSES_CSV))
        return args

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
        getattr(mrcnn1, ('do_'+action))(args) #calling mrcnn1.do_train(), mrcnn1.do_build_dataset() or mrcnn1.do_prediction() according to selected action

    def SSD(self, train_dataset, test_dataset, prediction_dataset, action, args):
        SSD1 = SSDHandler()
        getattr(SSD1, ('do_'+action))(args) #calling SSD1.do_build_dataset(), SSD1.do_train() or SSD1.do_prediction() according to selected action
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

        SSD_parser = subparsers.add_parser('SSD')
        SSD_parser.add_argument('action', help='Choose the action you want to do', choices=['build_dataset','train', 'predict'])

        parser.add_argument('--train_dataset', help='Path to a folder containing <images> and <annotations> folders (defaults to \'./dataset/train\')', default='./dataset/train')
        parser.add_argument('--test_dataset', help='Path to a folder containing <images> and <annotations> folders (defaults to \'./dataset/test\')', default='./dataset/test')
        parser.add_argument('--prediction-dataset', help='Path to a folder containing <images> folders (defaults to \'./dataset/prediction\')', default='./dataset/prediction')


        args, unknown = parser.parse_known_args()
        args_list = self.args2list(unknown)
        if len(args_list)==1 and args_list[0]== '':
            args_list=None
        getattr(self, args.detector)(args.train_dataset, args.test_dataset, args.prediction_dataset, args.action, args_list)
    def args2list(self,args):
        args_str = str(args)
        args_str = args_str[1:-1]
        args_str = args_str.replace('\'','')
        args_list = args_str.split(',')
        for i in range(len(args_list)):
            if len(args_list[i])>0:
                if args_list[i][0] == ' ':
                    args_list[i] = args_list[i][1:]
        return args_list

def main(args):
    invoker = Invoker()
    args = invoker.invoke_detectors(args)

if __name__ == '__main__':
    main(sys.argv)
