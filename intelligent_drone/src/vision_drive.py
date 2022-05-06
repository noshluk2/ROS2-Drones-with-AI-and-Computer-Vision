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




class vision_drive:
  def __init__(self):
    self.uav_camera_subscriber = rospy.Subscriber("/front_cam/front_cam/image",Image,self.video_feed_cb,10)
    self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    self.bridge = CvBridge()
    self.vel_msg=Twist()

  def video_feed_cb(self,data,a):
    frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
    self.vel_msg.linear.x = 0.01 # for very slow movement
    self.vel_msg.angular.z = 0.0
    self.vel_pub.publish(self.vel_msg)
    cv2.imshow("UAV_video_feed",frame)
    cv2.waitKey(1)







def main(args=None):
  rospy.init_node('drive_drone')
  drone_drive_obj = vision_drive()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")

if __name__ == '__main__':
  main()