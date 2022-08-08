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
# import os

from utilities.utitlities_pi import euler_from_quaternion,calc_focalLen_h_pix
from utilities.utilities import ordrpts,imshow

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
    
    self.start_navigating = False
    # self.img_no = 0

    self.navigator_ = navigator()
    
    self.sonar_curr = 0

    # [CSLAM] Map generated while drone navigating on field
    self.c_slam_map = np.zeros((540,960,3),np.uint8)

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
              print("Reached Capture height",self.img_w)
              if self.img_w!=0:
                altitude_m = sonar_data
                altitude_mm = altitude_m * 1000
                # Ground Sampling distance(decimeter/pix)
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
      #print("focal_len = {}".format(self.foclen_h_mm))
      #print("GSD = {} , GSD_x_deci_m = {}, GSD_y_deci_m = {}".format(self.GSD,self.GSD_x_cm,self.GSD_y_cm))



  # def get_centr_img(self,frame,perc = 20):
  #   rows = frame.shape[0]
  #   cols = frame.shape[1]
  #   centr = (int(rows/2),int(cols/2))

  #   crop_perc = perc/200
  #   img_cropped = frame[centr[0]-int(rows*(crop_perc)):centr[0]+int(rows*(crop_perc)),
  #                       centr[1]-int(cols*(crop_perc)):centr[1]+int(cols*(crop_perc))
  #                      ]

  #   return img_cropped


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
    point_0 = (int(rotated_x), int(rotated_y))

    # Point 1
    rotated_x = math.cos(theta) * (pt1[0] - cx) + math.sin(theta) * (pt1[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt1[0] - cx) + math.cos(theta) * (pt1[1] - cy) + cy
    point_1 = (int(rotated_x), int(rotated_y))

    # Point 2
    rotated_x = math.cos(theta) * (pt2[0] - cx) + math.sin(theta) * (pt2[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt2[0] - cx) + math.cos(theta) * (pt2[1] - cy) + cy
    point_2 = (int(rotated_x), int(rotated_y))

    # Point 3
    rotated_x = math.cos(theta) * (pt3[0] - cx) + math.sin(theta) * (pt3[1] - cy) + cx
    rotated_y = -math.sin(theta) * (pt3[0] - cx) + math.cos(theta) * (pt3[1] - cy) + cy
    point_3 = (int(rotated_x), int(rotated_y))

    return point_0, point_1, point_2, point_3

  def show_drone_on_map(self,frame,putText=False):

    self.get_drone_pose()

    x,y,orientation = self.drone_pose

    angle = orientation - 90

    if angle<-180:
      angle = angle + 360

    # [CSLAM] Origin of custom slam map .
    c_x = int(self.c_slam_map.shape[1]/2)
    c_y = int(self.c_slam_map.shape[0]/2)

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
    
    # [CSLAM] Drone current location being shown on Map
    cv2.circle(self.c_slam_map,(x_off,y_off),2,(255,255,255),-1)

    # [CSLAM] Drone Pose being overlayed using two different methods
    length = 20
    P1 = (x_off,y_off)
    P2 = ( 
            int(P1[0] + length * sin(angle * (pi / 180.0) ) ),
            int(P1[1] + length * cos(angle * (pi / 180.0) ) ) 
          )
    P1_rotated = self.rotate_point(P1[0], P1[1], orientation, (P1[0]-50,P1[1]))

    c_slam_map_draw = cv2.arrowedLine(self.c_slam_map.copy(), P1, P2, (0,140,255), 3,tipLength=0.6)
    c_slam_map_draw = cv2.arrowedLine(c_slam_map_draw, P1, P1_rotated, (0,0,255), 3,tipLength=0.3)
    
    # [CSLAM] Map Origin ==> White circle.
    c_slam_map_draw = cv2.circle(c_slam_map_draw, (c_x,c_y), 2, (255,255,255),-1)

    # [CSLAM] Drone pose being overlayed on map in text 
    if putText:
      pose_txt = "[X,Y,Orient] = ["+str(x_off)+","+str(y_off)+","+str(int(orientation))+"]"
      pose_orig_txt = "[XOrig,YOrig,Orient] = ["+str(x)+","+str(y)+","+str(int(orientation))+"]"

      c_slam_map_draw = cv2.putText(c_slam_map_draw,pose_txt, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
      c_slam_map_draw = cv2.putText(c_slam_map_draw,pose_orig_txt, (50,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
      

    # [CSLAM] Default and Pose Aligned FOV being displayed on CSLAM map (CounterClockwise pts)
    fov_strt     = (x_off-int(self.GSD_x_cm/2),y_off-int(self.GSD_y_cm/2))
    fov_btmleft  = (x_off-int(self.GSD_x_cm/2),y_off+int(self.GSD_y_cm/2))
    fov_end      = (x_off+int(self.GSD_x_cm/2),y_off+int(self.GSD_y_cm/2))
    fov_topright = (x_off+int(self.GSD_x_cm/2),y_off-int(self.GSD_y_cm/2))
    
    # Default ROI (based on HFOV AND VFOV) (Not aligned with robot pose)
    pts = np.array([np.asarray(fov_strt),np.asarray(fov_btmleft),
                    np.asarray(fov_end),np.asarray(fov_topright)
                   ],np.int32)     
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(c_slam_map_draw, [pts], True, (0,165,255),3)

    
    st_rot, btmlft_rot, end_rot, toprgt_rot = self.rotate_rectangle(orientation+90,P1[0], P1[1],fov_strt,  fov_btmleft, fov_end, fov_topright)

    c_slam_map_draw = cv2.putText(c_slam_map_draw,str(fov_strt), (fov_strt[0]-20,fov_strt[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
    c_slam_map_draw = cv2.putText(c_slam_map_draw,str(fov_btmleft), (fov_btmleft[0]-20,fov_btmleft[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)
    c_slam_map_draw = cv2.putText(c_slam_map_draw,str(fov_end), (fov_end[0]+20,fov_end[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
    c_slam_map_draw = cv2.putText(c_slam_map_draw,str(fov_topright), (fov_topright[0]+20,fov_topright[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)


    # ROI representing the FOV of the drone in the simulation (Aligned with Robot Pose)
    fov_pts = np.array([np.asarray(st_rot),np.asarray(btmlft_rot),
                    np.asarray(end_rot),np.asarray(toprgt_rot)
                   ],np.int32)     
    fov_pts = fov_pts.reshape((-1, 1, 2))
    cv2.polylines(c_slam_map_draw, [fov_pts], True, (0,255,0),3)

    fov_pts_str = "[FOV-PoseAligned] = [ "+str(st_rot)+" , "+str(btmlft_rot)+" , "+str(end_rot)+" , "+str(toprgt_rot)+" ] "
    c_slam_map_draw = cv2.putText(c_slam_map_draw,fov_pts_str, (50,430), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)

    fov_pts_ordered = ordrpts(fov_pts,image_draw=c_slam_map_draw)

    cv2.imshow("C-SLAM (map)",self.c_slam_map)

    return c_slam_map_draw,pts,fov_pts


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
    c_slam_map_draw,fov_unrot_pts,fov_pts = self.show_drone_on_map(frame,True)

    if self.attained_required_elevation:
      if not self.attained_capture_elevation:
        if self.time_elasped <30:
          self.time_elasped +=1
        else:
          self.required_elevation = self.capture_elevation
          self.attained_required_elevation = False
          self.start_navigating = True
          
      else:

        # if self.start_navigating:
        #     self.img_no = self.img_no + 1
        #     img_dir = os.path.abspath("intelligent_drone/src/data/is_plant_data")
        #     if not os.path.isdir(img_dir):
        #         os.mkdir(img_dir)
        #     img_name = img_dir + "/" +str(self.img_no)+".png"
        #     img_cropped = self.get_centr_img(frame,perc=40)
        #     #cv2.imwrite(img_name,img_cropped)
        #     print("start navigating")
            
            self.navigator_.navigate(frame, frame_draw,self.vel_msg,self.vel_pub,self.sonar_curr,self.c_slam_map,c_slam_map_draw,fov_unrot_pts,fov_pts,self.drone_pose,self.GSD)



    imshow("UAV_video_feed",frame_draw)
    
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