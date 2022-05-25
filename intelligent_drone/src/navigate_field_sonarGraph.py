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
import os


from matplotlib import pyplot as plt

from drone_motionplanning import bot_motionplanner
#from goal_tracker import TL_Tracker
#from goal_tracker_2 import TL_Tracker
from goal_tracker_3 import TL_Tracker

from itertools import count
from matplotlib.animation import FuncAnimation



import signal
import sys
import threading

from utilities import euler_from_quaternion 
from math import sin,cos,pi,degrees
from gazebo_msgs.srv import GetModelState
from numpy import interp



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
    
    cv2.namedWindow("UAV_video_feed",cv2.WINDOW_NORMAL)
    self.mini_goals = []
    self.start_navigating = False
    self.img_no = 0

    self.drone_motionplanner = bot_motionplanner()
    self.TL_Tracker_ = TL_Tracker()
    
    self.x_vals = []
    self.y_vals = []

    self.index = count()

    self.sonar_curr = 0

    self.occupncy_grd = np.zeros((540,960,3),np.uint8)


    ## rosservice call /gazebo/get_model_state "drone_model: 'chirya'"
    self.model_state_obj = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    self.drone_pose = []


  def animate_sonar(self,num):
    print("party")
    self.x_vals.append(next(self.index))
    self.y_vals.append(self.sonar_curr)

    plt.cla()
    plt.plot(self.x_vals,self.y_vals)

 
  def animate(self):
    
    plt.style.use('fivethirtyeight')    

    self.ani = FuncAnimation(plt.gcf(), self.animate_sonar,interval = 100)
    
    plt.tight_layout()
    
    plt.show()   


    

  def sonar_callback(self,msg):
      sonar_data = msg.range

      if not (self.attained_required_elevation and self.attained_capture_elevation):

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

      self.sonar_curr = sonar_data
      print("Sonar Values : " , sonar_data)



  # stores mouse position in global variables ix(for x coordinate) and iy(for y coordinate) 
  # on double click inside the image
  def select_point(self,event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDBLCLK: # captures left button double-click
        print("Taking mini-goal")
        self.mini_goals.append([x,y])
        if len(self.mini_goals)==1:
            self.start_navigating = True


  def get_centr_img(self,frame,perc = 20):
    rows = frame.shape[0]
    cols = frame.shape[1]
    centr = (int(rows/2),int(cols/2))

    crop_perc = perc/200
    img_cropped = frame[centr[0]-int(rows*(crop_perc)):centr[0]+int(rows*(crop_perc)),
                        centr[1]-int(cols*(crop_perc)):centr[1]+int(cols*(crop_perc))
                       ]

    return img_cropped

  @staticmethod
  def findstrt_pt(pta,ptb):

    x1,y1 = pta
    x2,y2 = ptb
    return (x1*2-x2,y1*2-y2)

  def get_drone_pose(self):

    response_variable = self.model_state_obj("chirya","ground_plane")
    
    x_pos = int((response_variable.pose.position.x)*10)
    y_pos = int((response_variable.pose.position.y)*10)

    quaternions = response_variable.pose.orientation
    (roll,pitch,yaw) = euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
    yaw_deg = degrees(yaw)
    
    self.drone_pose = [x_pos,y_pos,yaw_deg]

  def show_drone_on_map(self,frame,putText=False):

    self.get_drone_pose()

    x,y,orientation = self.drone_pose

    angle = orientation - 90

    if angle<-180:
      angle = angle + 360


    c_x = int(self.occupncy_grd.shape[1]/2)
    c_y = int(self.occupncy_grd.shape[0]/2)

    x_off = -x + c_x
    y_off = y + c_y

    cv2.circle(self.occupncy_grd,(x_off,y_off),2,(255,255,255),-1)

    length = 20
    P1 = (x_off,y_off)
    P2 = ( 
            int(P1[0] + length * sin(angle * (pi / 180.0) ) ),
            int(P1[1] + length * cos(angle * (pi / 180.0) ) ) 
          )

    occupncy_grd = cv2.arrowedLine(self.occupncy_grd.copy(), P1, P2, (0,140,255), 3,tipLength=0.6)
    
    if putText:
      pose_txt = "[X,Y,Orient] = ["+str(x_off)+","+str(y_off)+","+str(orientation)+"]"
      occupncy_grd = cv2.putText(occupncy_grd,pose_txt, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
        

    cv2.imshow("OccpancyGrid (Drone-Pose)",occupncy_grd)



  def video_feed_cb(self,data,grbg):
    
    frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
    frame_draw = frame.copy()

    # Get updated drone Pose and display to User
    self.show_drone_on_map(frame,True)

    if self.attained_required_elevation:
    
      if not self.attained_capture_elevation:
        if self.time_elasped <30:
          self.time_elasped +=1
        else:
          self.required_elevation = self.capture_elevation
          self.attained_required_elevation = False
      else:

        # bind select_point function to a window that will capture the mouse click
        cv2.setMouseCallback("UAV_video_feed", self.select_point)

        if self.start_navigating:
            self.img_no = self.img_no + 1
            img_dir = os.path.abspath("intelligent_drone/src/data/is_plant_data")
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)
            img_name = img_dir + "/" +str(self.img_no)+".png"
            
            img_cropped = self.get_centr_img(frame,perc=40)
            #cv2.imwrite(img_name,img_cropped)
            print("start navigating")

            self.TL_Tracker_.track_and_retrive_goal_loc(frame,frame_draw)


            # [Drone: MotionPlanning] Reach the (maze exit) by navigating the path previously computed
            drone_loc = (int(frame.shape[1]/2),int(frame.shape[0]/2))

            path = []
            for bbox in self.TL_Tracker_.tracked_bboxes:
              center_x = int(bbox[0] + (bbox[2]/2))
              center_y = int(bbox[1] + (bbox[3]/2))
              
              #path = [(center_x,center_y-250),(center_x,center_y)]
              #path = [(center_x,center_y)]
              path.append((center_x,center_y))

            if path!=[]:
              pt_c = self.findstrt_pt(path[0],path[1])
              path = [pt_c] + path



            max_dist = min(frame.shape[0],frame.shape[1])
            self.drone_motionplanner.nav_path(drone_loc, path, max_dist, self.vel_msg, self.vel_pub)

            goal = (self.drone_motionplanner.goal_pose_x,self.drone_motionplanner.goal_pose_y)
            frame_draw = cv2.circle(frame_draw, goal, 5, (0,255,0),3)


            cv2.putText(frame_draw, "[Angle,Speed] = ", (50,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
            cv2.putText(frame_draw, str(self.drone_motionplanner.curr_angle), (320,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
            cv2.putText(frame_draw, str(self.drone_motionplanner.curr_speed), (420,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)

            cv2.putText(frame_draw, str(self.TL_Tracker_.tracked_bboxes[0]), (250,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
            cv2.putText(frame_draw, self.TL_Tracker_.mode, (50,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
    
    frame_draw = cv2.resize(frame_draw,None, fx=0.5,fy=0.5)
    cv2.imshow("UAV_video_feed",frame_draw)
    
    k = cv2.waitKey(1)
    if k==27:
        plt.close("all")

    

def signal_handler(signal, frame):
    print('\nYou pressed Ctrl+C, keyboardInterrupt detected!')
    sys.exit(0)


def main(args=None):

  signal.signal(signal.SIGINT, signal_handler)
  forever_wait = threading.Event()

  
  rospy.init_node('drive_drone')

  drone_drive_obj = vision_drive()
  drone_drive_obj.animate()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")
  


if __name__ == '__main__':
  main()