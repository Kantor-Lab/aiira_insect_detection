#!/usr/bin/env python3

import rospkg
import rospy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from aiira_msgs.msg import InsectClassInfo

import json
import torch
import time
import torchvision
import evaluate
import cv2
import pandas as pd
import argparse
import logging

class InsectClassifier:
    insect_names = ''
    model_file = ''
    classes_file = ''
    def __init__(self, insect_names, model_file, classes_file):
        self.insect_names = insect_names
        self.model_file = model_file
        self.classes_file = classes_file
        logging.info(f'insects filename = {self.insect_names}')
        logging.info(f'model path = {self.model_file}')
        self.cmnDf = pd.read_csv(insect_names)
        self.model=torchvision.models.regnet_y_32gf()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logging.info('CUDA Device', torch.cuda.get_device_name(0))
            logging.info('Memory Usage:')
            logging.info('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            logging.info('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        self.weights=torch.load(self.model_file,map_location=torch.device('cpu'))['model']
        self.model.fc=torch.nn.Linear(3712, 2526)
        self.model.load_state_dict(self.weights, strict=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()

    def image_classify_by_file(self, image_filename, known_label=''):
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        sciPred, cmnPred, confirmed, order, other_classes = evaluate.evaluate(
            self.model,image,self.cmnDf,classes_file=self.classes_file)
        rospy.loginfo(f'\tScientific Name = {sciPred}\n' +
            f'\tCommon Name       = {cmnPred}\n' +
            f'\tConfirmed         = {confirmed}\n' +
            f'\tOrder             = {order}\n' +
            f'\tOther Classes     = {other_classes}')
        return sciPred, cmnPred, confirmed, order, other_classes

    def image_classify(self, image, seq=-1):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        start_time = time.time()
        sciPred, cmnPred, confirmed, order, other_classes = evaluate.evaluate(
            self.model,image,self.cmnDf,classes_file=self.classes_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo(f"Elapsed time: {elapsed_time:.3f} seconds")
        rospy.loginfo(f'\tHeader.seq = {seq}\n' +
            f'\tScientific Name = {sciPred}\n' +
            f'\tCommon Name       = {cmnPred}\n' +
            f'\tConfirmed         = {confirmed}\n' +
            f'\tOrder             = {order}\n' +
            f'\tOther Classes     = {other_classes}')
        res = {
            "Elapsed Seconds": elapsed_time,
            "Header.seq": seq,
            "Scientific Name": sciPred,
            "Common Name": cmnPred,
            "Confirmed": confirmed,
            "Order": order,
            "Other Classes": other_classes
            }
        return res



class InsectClassifierNode:
    def __init__(self, display_positives_only=True):
        model_file = '/home/frc-ag-101/wksp/insect-detection/model.pth'
        params_path = rospkg.RosPack().get_path('aiira_detection')
        insect_names = params_path + '/params/insectNames_new.csv'
        classes_file = params_path + '/params/classes.txt'
        self.display_positives_only = display_positives_only
        self.clfr = InsectClassifier(insect_names, model_file, classes_file)

        rospy.init_node('insect_classification_node')

        self.bridge = CvBridge()

        rospy.Subscriber('/theia/left/image_raw', Image, self.image_callback)
        self.result_pub = rospy.Publisher('/classification_result', InsectClassInfo, queue_size=5)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            res = self.clfr.image_classify(cv_image, seq=msg.header.seq)

            # Publish the classification result
            #  self.result_pub.publish(str(json.dumps(res)))
            insect_info_msg = InsectClassInfo()
            #  "Elapsed Seconds": elapsed_time,
            #  "Header.seq": seq,
            #  "Scientific Name": sciPred,
            #  "Common Name": cmnPred,
            #  "Confirmed": confirmed,
            #  "Order": order,
            #  "Other Classes": other_classes
            insect_info_msg.scientific_name = res["Scientific Name"]
            insect_info_msg.common_name = res["Common Name"]
            insect_info_msg.confirmed = res["Confirmed"]
            insect_info_msg.order = res["Order"]
            insect_info_msg.other_classes = res["Other Classes"]

            self.result_pub.publish(insect_info_msg)

            # Display the image if it is a positive result
            if self.display_positives_only and res['Confirmed']:
                cv.namedWindow("Insect Classification Results", cv.WINDOW_NORMAL)
                cv2.imshow('Insect Classification', cv_image)
                cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error processing the image: {str(e)}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = InsectClassifierNode(display_positives_only=True)
        node.run()
    except rospy.ROSInterruptException:
        pass

