#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

class ImageDisplayNode:
    def __init__(self):
        rospy.init_node('image_display_node')

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/image_topic', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Display the image
            cv2.imshow('Received Image', cv_image)
            cv2.waitKey(1)  # Wait for a short time to handle GUI events

        except Exception as e:
            rospy.logerr(f"Error processing the image: {str(e)}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ImageDisplayNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
