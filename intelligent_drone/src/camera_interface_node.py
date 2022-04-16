#!/usr/bin/env python3

####
# 
#  Written by Muhammad Luqman
# 
#  17/11/21
#
###
import rospy
import cv2 
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image 

class video_recording():
  def __init__(self):
    self.uav_camera_subscriber = rospy.Subscriber("/front_cam/front_cam/image",Image,self.uav_video_feed,10)
    self.uav_out = cv2.VideoWriter('/home/luqman/uav_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,1080))
    self.bridge = CvBridge()


  def uav_video_feed(self, data,a):
      frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
      self.uav_out.write(frame) # saving the video
      cv2.imshow("UAV_video_feed",frame)
      cv2.waitKey(1) 

  
def main(args=None):
  rospy.init_node('Simulation_Guard')
  video_class_obj = video_recording()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")
  
if __name__ == '__main__':
  main()