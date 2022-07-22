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
from numpy import interp
import os

from motionplanners.drone_motionplanning import drone_motionplanner
from trackers.goal_tracker import Tracker

from utilities.utitlities_pi import euler_from_quaternion,calc_focalLen_h_pix
from math import sin,cos,pi,degrees,radians
from gazebo_msgs.srv import GetModelState


from navigation import navigator



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
    self.capture_elevation = 4.5
    self.time_elasped = 0
    
    cv2.namedWindow("UAV_video_feed",cv2.WINDOW_NORMAL)
    self.mini_goals = []
    self.start_navigating = False
    self.img_no = 0

    self.navigator_ = navigator()
    
    self.sonar_curr = 0

    self.occupncy_grd = np.zeros((540,960,3),np.uint8)
    ## rosservice call /gazebo/get_model_state "drone_model: 'chirya'"
    self.model_state_obj = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    self.drone_pose = []

    self.aspect_ratio = 0
    self.img_w = 0
    self.img_h = 0
    self.sensor_w = 0
    self.foclen_h_mm = 0
    self.foclen_v_mm = 0
    self.GSD = 0
    self.GSD_x_cm = 0
    self.GSD_y_cm = 0


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
              if self.img_w!=0:
                altitude_m = sonar_data
                altitude_mm = altitude_m * 1000
                self.GSD = (altitude_mm / self.foclen_h_mm) / 100
                self.GSD_x_cm = self.GSD * self.img_w
                self.GSD_y_cm = self.GSD * self.img_h
                print("GSD = {} , GSD_x_mm = {}, GSD_y_mm = {}".format(self.GSD,self.GSD_x_cm,self.GSD_y_cm))
            #self.sonar_subscriber.unregister()
        else:
            self.vel_msg.linear.z=0.5
            print("Reaching required height")

        self.vel_pub.publish(self.vel_msg)

      self.sonar_curr = sonar_data
      print("Sonar Values : " , sonar_data)
      print("focal_len = {}".format(self.foclen_h_mm))
      print("GSD = {} , GSD_x_deci_m = {}, GSD_y_deci_m = {}".format(self.GSD,self.GSD_x_cm,self.GSD_y_cm))



  def get_centr_img(self,frame,perc = 20):
    rows = frame.shape[0]
    cols = frame.shape[1]
    centr = (int(rows/2),int(cols/2))

    crop_perc = perc/200
    img_cropped = frame[centr[0]-int(rows*(crop_perc)):centr[0]+int(rows*(crop_perc)),
                        centr[1]-int(cols*(crop_perc)):centr[1]+int(cols*(crop_perc))
                       ]

    return img_cropped


  def get_drone_pose(self):

    response_variable = self.model_state_obj("chirya","ground_plane")
    
    x_pos = int((response_variable.pose.position.x)*10)
    y_pos = int((response_variable.pose.position.y)*10)

    quaternions = response_variable.pose.orientation
    (roll,pitch,yaw) = euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
    yaw_deg = degrees(yaw)
    
    self.drone_pose = [x_pos,y_pos,yaw_deg]
  
  def rotate_point(self,cx, cy, angle, p_tuple):
        
    #  * [Clockwise Rotation happens in Image as Y increases downwards]
    #  * .------------------------------------.
    #  * |            * -------.          img |
    #  * |    rotated_pt    -90 \             |
    #  * |                       \            |
    #  * |                        \           |
    #  * |            * ---------> *          |
    #  * |         ref_pt     given_pt        |
    #  * |                       /            |
    #  * |                      /             |
    #  * |    rotated_pt   +90 /              |
    #  * |            * ------'               |
    #  * '------------------------------------'

    p = list(p_tuple)
    
    s = sin(radians(angle))
    c = cos(radians(angle))
    
    # translate point back to origin:
    p[0] -= cx
    p[1] -= cy

    #rotate point
    # (Interestingly) the plot is modified from what the internet provided
    # AS we want to represent simulation (cartesian) angles in the image we will do (counterclockwise) rotation for +
    xnew = p[0] * c + p[1] * s
    ynew = - p[0] * s + p[1] * c

    # translate point back:
    p[0] = xnew + cx
    p[1] = ynew + cy
    
    return (int(p[0]),int(p[1]))
  
  def rotate_rectangle(self, theta_deg,cx,cy,pt0, pt1, pt2, pt3):
    # Counterclockwise rotation of 4 points
    theta = radians(theta_deg)
    # Point 0
    rotated_x = math.cos(theta) * (pt0[0] - cx) + math.sin(theta) * (pt0[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt0[0] - cx) + math.cos(theta) * (pt0[1] - cy) + cy
    point_0 = (rotated_x, rotated_y)

    # Point 1
    rotated_x = math.cos(theta) * (pt1[0] - cx) + math.sin(theta) * (pt1[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt1[0] - cx) + math.cos(theta) * (pt1[1] - cy) + cy
    point_1 = (rotated_x, rotated_y)

    # Point 2
    rotated_x = math.cos(theta) * (pt2[0] - cx) + math.sin(theta) * (pt2[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt2[0] - cx) + math.cos(theta) * (pt2[1] - cy) + cy
    point_2 = (rotated_x, rotated_y)

    # Point 3
    rotated_x = math.cos(theta) * (pt3[0] - cx) + math.sin(theta) * (pt3[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt3[0] - cx) + math.cos(theta) * (pt3[1] - cy) + cy
    point_3 = (rotated_x, rotated_y)

    return point_0, point_1, point_2, point_3

  def show_drone_on_map(self,frame,putText=False):

    self.get_drone_pose()

    x,y,orientation = self.drone_pose

    angle = orientation - 90

    if angle<-180:
      angle = angle + 360


    c_x = int(self.occupncy_grd.shape[1]/2)
    c_y = int(self.occupncy_grd.shape[0]/2)

    # a) Shifting Origin to Image center by subtracting
    # b) In Image to get identical effect y increases downwards so we inverse y so it increases now upwards
    # a) Inverting both x and y (based on the direction we want our drone to move.)
    #     e.g If x increases in simulation ===> We want it to decrease in image (go left)
    #         If y increases        //     ===> //        //  increase    //    (go down)
    #       [ Simulation/Normal Convention ]      [ Image ]
    #
    #                    Y                          -Y
    #                     |                           |
    #                  ___|___                   ____ | ____ 
    #                -X   |    X                X     |     -X
    #                     |                           |
    #                   -Y                           Y

    x_off = -(+x - c_x)
    y_off = -(-y - c_y)

    cv2.circle(self.occupncy_grd,(x_off,y_off),2,(255,255,255),-1)

    length = 20
    P1 = (x_off,y_off)
    P2 = ( 
            int(P1[0] + length * sin(angle * (pi / 180.0) ) ),
            int(P1[1] + length * cos(angle * (pi / 180.0) ) ) 
          )
    P1_rotated = self.rotate_point(P1[0], P1[1], orientation, (P1[0]-50,P1[1]))

    occupncy_grd = cv2.arrowedLine(self.occupncy_grd.copy(), P1, P2, (0,140,255), 3,tipLength=0.6)
    occupncy_grd = cv2.arrowedLine(occupncy_grd, P1, P1_rotated, (0,0,255), 3,tipLength=0.3)
    occupncy_grd = cv2.circle(occupncy_grd, (c_x,c_y), 2, (255,255,255),-1)

    if putText:
      pose_txt = "[X,Y,Orient] = ["+str(x_off)+","+str(y_off)+","+str(orientation)+"]"
      pose_orig_txt = "[XOrig,YOrig,Orient] = ["+str(x)+","+str(y)+","+str(orientation)+"]"
      #Shape_txt = "(Width(Cols),height(Rows)) = ["+str(self.img_w)+","+str(self.img_h)+"]"
      occupncy_grd = cv2.putText(occupncy_grd,pose_txt, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
      occupncy_grd = cv2.putText(occupncy_grd,pose_orig_txt, (50,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
      #occupncy_grd = cv2.putText(occupncy_grd,Shape_txt, (50,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
      

    fov_strt     = (x_off-int(self.GSD_x_cm/2),y_off-int(self.GSD_y_cm/2))
    fov_topright = (x_off+int(self.GSD_x_cm/2),y_off-int(self.GSD_y_cm/2))
    fov_btmleft  = (x_off-int(self.GSD_x_cm/2),y_off+int(self.GSD_y_cm/2))
    fov_end      = (x_off+int(self.GSD_x_cm/2),y_off+int(self.GSD_y_cm/2))

    st_rot,toprgt_rot,btmlft_rot,end_rot = self.rotate_rectangle(orientation+90,P1[0], P1[1],fov_strt, fov_topright, fov_btmleft, fov_end)

    occupncy_grd = cv2.putText(occupncy_grd,str(fov_strt), (fov_strt[0]-20,fov_strt[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
    occupncy_grd = cv2.putText(occupncy_grd,str(fov_btmleft), (fov_btmleft[0]-20,fov_btmleft[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)
    occupncy_grd = cv2.putText(occupncy_grd,str(fov_end), (fov_end[0]+20,fov_end[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
    occupncy_grd = cv2.putText(occupncy_grd,str(fov_topright), (fov_topright[0]+20,fov_topright[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
  

    # Default ROI (based on HFOV AND VFOV) (Not aligned with robot pose)
    pts = np.array([np.asarray(fov_strt),np.asarray(fov_btmleft),
                    np.asarray(fov_end),np.asarray(fov_topright)
                   ],np.int32)     
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(occupncy_grd, [pts], True, (0,165,255),3)
    

    # ROI representing the FOV of the drone in the simulation (Aligned with Robot Pose)
    pts_rot = np.array([np.asarray(st_rot),np.asarray(btmlft_rot),
                    np.asarray(end_rot),np.asarray(toprgt_rot)
                   ],np.int32)     
    pts_rot = pts_rot.reshape((-1, 1, 2))
    cv2.polylines(occupncy_grd, [pts_rot], True, (0,255,0),3)

    # What didnot work! Directly rotating the points using origin caused many problems
    #M = cv2.getRotationMatrix2D((int(occupncy_grd.shape[1]/2), int(occupncy_grd.shape[0]/2)),  orientation , 1.0)
    #field_mask_rot = cv2.warpAffine(field_mask_rot, M, (hue.shape[1], hue.shape[0]))
    #pts = cv2.warpAffine(pts, M, (pts.shape[1], pts.shape[0]))
    #pts = cv2.transform(pts, M)
    #cv2.polylines(occupncy_grd, [pts], True, (0,255,0),3)

    cv2.namedWindow("OccpancyGrid (Drone-Pose)",cv2.WINDOW_NORMAL)
    cv2.imshow("OccpancyGrid (Drone-Pose)",occupncy_grd)


  def video_feed_cb(self,data,grbg):
    
    frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
    frame_draw = frame.copy()

    if self.img_w == 0:
      self.img_w = frame.shape[1]
      self.img_h = frame.shape[0]
      self.aspect_ratio = self.img_w/self.img_h
      self.sensor_w = self.img_w
      focallen_h_pix = calc_focalLen_h_pix(self.img_w)
      self.foclen_h_mm = focallen_h_pix
      self.foclen_v_mm = self.foclen_h_mm / self.aspect_ratio

    # Get updated drone Pose and display to User
    self.show_drone_on_map(frame,True)

    if self.attained_required_elevation:
      if not self.attained_capture_elevation:
        if self.time_elasped <30:
          self.time_elasped +=1
        else:
          self.required_elevation = self.capture_elevation
          self.attained_required_elevation = False
          self.start_navigating = True
          
      else:

        if self.start_navigating:
            self.img_no = self.img_no + 1
            img_dir = os.path.abspath("intelligent_drone/src/data/is_plant_data")
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)
            img_name = img_dir + "/" +str(self.img_no)+".png"
            img_cropped = self.get_centr_img(frame,perc=40)
            #cv2.imwrite(img_name,img_cropped)
            print("start navigating")
            
            self.navigator_.navigate(frame, frame_draw,self.vel_msg,self.vel_pub,self.sonar_curr,self.occupncy_grd,self.drone_pose)



    frame_draw = cv2.resize(frame_draw,None, fx=0.5,fy=0.5)
    cv2.imshow("UAV_video_feed",frame_draw)
    
    k = cv2.waitKey(1)
    if k==27:
        plt.close("all")

    

def main(args=None):

  rospy.init_node('drive_drone')

  drone_drive_obj = vision_drive()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")
  


if __name__ == '__main__':
  main()