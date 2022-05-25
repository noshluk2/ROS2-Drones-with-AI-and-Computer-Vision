#!/usr/bin/env python3


'''
Process :
When you get an image from drone
you send velocity command to drone
'''


import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from sensor_msgs.msg import Range

import time

import math
import numpy as np

class vision_drive:
  def __init__(self):
    self.uav_camera_subscriber = rospy.Subscriber("/front_cam/front_cam/image",Image,self.video_feed_cb,10)
    self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    self.bridge  = CvBridge()
    self.vel_msg = Twist()

    self.sonar_subscriber = rospy.Subscriber("/sonar_height", Range, self.sonar_callback)
    self.attained_required_elevation = False
    self.attained_capture_elevation = False
    self.required_elevation = 2.2
    self.capture_elevation = 4.0
    self.time_elasped = 0

  def line_detect(self,gray,thresh = 450):
    # Apply edge detection method on the image
    dst = cv2.Canny(gray,50,150,apertureSize = 3)
    #dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(dst, 1, np.pi / 180, thresh, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    return cdst

  def sonar_callback(self,msg):
      sonar_data = msg.range
      if(sonar_data>=self.required_elevation):
          self.vel_msg.linear.z=0.0
          self.vel_pub.publish(self.vel_msg)
          self.attained_required_elevation = True
          print("Reached required height")
          if self.required_elevation==self.capture_elevation:
            self.attained_capture_elevation = True
            print("Reached Capture height")
          #self.sonar_subscriber.unregister()
      else:
          self.vel_msg.linear.z=0.5
          print("Reaching required height")

      self.vel_pub.publish(self.vel_msg)
      print("Sonar Values : " , sonar_data)


  def video_feed_cb(self,data,grbg):
    frame = self.bridge.imgmsg_to_cv2(data,'bgr8')

    if self.attained_required_elevation:

      self.vel_msg.linear.x = 0.00 # for very slow movement
      self.vel_msg.angular.z = 0.0
      self.vel_pub.publish(self.vel_msg)
    
      if not self.attained_capture_elevation:
        if self.time_elasped <30:
          self.time_elasped +=1
        else:
          self.required_elevation = self.capture_elevation
          self.attained_required_elevation = False
      else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_lines = self.line_detect(gray)

        edges = cv2.Canny(gray, 50, 150,None,3)
        #edges_lines = self.line_detect(edges)

        gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges_lap = cv2.Laplacian(gray, cv2.CV_64F)

        otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges_otsu = cv2.Canny(otsu_mask, 50, 150,None,3)

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hue = hls[:,:,0]
        #hue_lines = self.line_detect(hue)
        hue_edges = cv2.Canny(hue, 50, 150,None,3)

        otsu_hue_mask = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges_hue_otsu = cv2.Canny(otsu_hue_mask, 50, 150,None,3)

        cv2.imshow("(1) gray",gray)
        cv2.imshow("(1) gray_lines",gray_lines)
        cv2.imshow("(1) edges",edges)
        #cv2.imshow("(1) edges_lines",edges_lines)
        cv2.imshow("(1) edges_lap",edges_lap)


        cv2.imshow("(2) otsu_mask",otsu_mask)
        cv2.imshow("(2) edges_otsu",edges_otsu)
        cv2.imshow("(3) hue",hue)
        #cv2.imshow("(3) hue_lines",hue_lines)
        cv2.imshow("(3) hue_edges",hue_edges)
        cv2.imshow("(3) otsu_hue_mask",otsu_mask)
        cv2.imshow("(3) edges_hue_otsu",edges_otsu)

    
    
    cv2.imshow("UAV_video_feed",frame)
    cv2.waitKey(1)
    

        




def main(args=None):
  rospy.init_node('drive_drone')
  #rate = rospy.Rate(10) # 10hz
  drone_drive_obj = vision_drive()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")


if __name__ == '__main__':
  main()