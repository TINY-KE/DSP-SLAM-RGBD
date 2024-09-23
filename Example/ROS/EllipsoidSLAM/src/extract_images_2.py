# coding:utf-8
#!/usr/bin/python
  
# Extract images from a bag file.
  
#PKG = 'beginner_tutorials'
import roslib   #roslib.load_manifest(PKG)
import rosbag
import rospy
import decimal
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
  
# Reading bag filename from command line or roslaunch parameter.
#import os
#import sys
  
camera0_path = '/media/lzw/Win-Soft/workspace/datasets/home/home_2020-10-19-15-48-35/rgb/'
camera1_path = '/media/lzw/Win-Soft/workspace/datasets/home/home_2020-10-19-15-48-35/depth/'
  
class ImageCreator():
  
  
    def __init__(self):
        self.bridge = CvBridge()
        with rosbag.Bag('/media/lzw/Win-Soft/workspace/datasets/home/home_2020-10-19-15-48-35.bag', 'r') as bag:  #要读取的bag文件；
            for topic,msg,t in bag.read_messages():
                if topic == "/kinect2/qhd/image_color_rect": #图像的topic；
                        try:
                            cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                        except CvBridgeError as e:
                            print e
                        timestr = msg.header.stamp.to_sec()
                        #%.6f表示小数点后带有6位，可根据精确度需要修改；
                        # timer = 100000000 * timestr
                        timer = 1 * timestr
                        image_name = "%.6f" % timer + ".png" #图像命名：时间戳.png
                        cv2.imwrite(camera0_path + image_name, cv_image)  #保存；
                        print("saving to", image_name);
                elif topic == "/kinect2/qhd/image_depth_rect": #图像的topic；
                        try:
                            cv_image = self.bridge.imgmsg_to_cv2(msg,"16UC1")
                        except CvBridgeError as e:
                            print e
                        timestr = msg.header.stamp.to_sec()
                        #%.6f表示小数点后带有6位，可根据精确度需要修改；
                        timer = 1 * timestr
                        image_name = "%.6f" % timer + ".png" #图像命名：时间戳.png
                        cv2.imwrite(camera1_path + image_name, cv_image)  #保存；
  
if __name__ == '__main__':
  
    #rospy.init_node(PKG)
  
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass