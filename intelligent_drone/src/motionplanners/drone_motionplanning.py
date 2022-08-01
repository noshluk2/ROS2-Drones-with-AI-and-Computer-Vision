'''
> Purpose :
Module to perform motionplanning for helping the vehicle navigate to the desired destination

> Usage :
You can perform motionplanning by
1) Importing the class (bot_motionplanner)
2) Creating its object
3) Accessing the object's function of (nav_path). 
E.g ( self.bot_motionplanner.nav_path(bot_loc, path, self.vel_msg, self.velocity_publisher) )


> Inputs:
1) Robot Current location
2) Found path to destination
3) Velocity object for manipulating linear and angular component of robot
4) Velocity publisher to publish the updated velocity object

> Outputs:
1) speed              => Speed with which the car travels at any given moment
2) angle              => Amount of turning the car needs to do at any moment

Author :
Haider Abbasi

Date :
6/04/22
'''
import cv2
import numpy as np
from numpy import interp

from math import pow , atan2,sqrt , degrees,asin
import os

import pygame
pygame.mixer.init()
pygame.mixer.music.load(os.path.abspath('intelligent_drone/resource/aud_chomp.mp3'))


class drone_motionplanner():


    def __init__(self):

        # [Container] => Bot Angle [Image]
        self.bot_angle = 90

        # State Variable ==> (Maze Exit) Not Reached ?
        self.goal_not_reached_flag = True
        # [Containers] ==> Mini-Goal (X,Y)
        self.goal_pose_x = 0
        self.goal_pose_y = 0
        # [Iterater] ==> Current Mini-Goal iteration
        self.path_iter = 0

        self.curr_speed = 0
        self.curr_angle = 0


    @staticmethod
    def bck_to_orig(pt,transform_arr,rot_mat):

        st_col = transform_arr[0] # cols X
        st_row = transform_arr[1] # rows Y
        tot_cols = transform_arr[2] # total_cols / width W
        tot_rows = transform_arr[3] # total_rows / height H
        
        # point --> (col(x),row(y)) XY-Convention For Rotation And Translated To MazeCrop (Origin)
        #pt_array = np.array( [pt[0]+st_col, pt[1]+st_row] )
        pt_array = np.array( [pt[0], pt[1]] )
        
        # Rot Matrix (For Normal XY Convention Around Z axis = [cos0 -sin0]) But for Image convention [ cos0 sin0]
        #                                                      [sin0  cos0]                           [-sin0 cos0]
        rot_center = (rot_mat @ pt_array.T).T# [x,y]
        
        # Translating Origin If neccasary (To get whole image)
        rot_cols = tot_cols#tot_rows
        rot_rows = tot_rows#tot_cols
        rot_center[0] = rot_center[0] + (rot_cols * (rot_center[0]<0) ) + st_col  
        rot_center[1] = rot_center[1] + (rot_rows * (rot_center[1]<0) ) + st_row 
        return rot_center

    def display_control_mechanism_in_action(self,bot_loc,path,img_shortest_path,bot_localizer,frame_disp):
        Doing_pt = 0
        Done_pt = 0

        path_i = self.path_iter
        
        # Circle to represent car current location
        img_shortest_path = cv2.circle(img_shortest_path, bot_loc, 3, (0,0,255))

        if ( (type(path)!=int) and ( path_i!=(len(path)-1) ) ):
            curr_goal = path[path_i]
            # Mini Goal Completed
            if path_i!=0:
                img_shortest_path = cv2.circle(img_shortest_path, path[path_i-1], 3, (0,255,0),2)
                Done_pt = path[path_i-1]
            # Mini Goal Completing   
            img_shortest_path = cv2.circle(img_shortest_path, curr_goal, 3, (0,140,255),2)
            Doing_pt = curr_goal
        else:
            # Only Display Final Goal completed
            img_shortest_path = cv2.circle(img_shortest_path, path[path_i], 10, (0,255,0))
            Done_pt = path[path_i]

        if Doing_pt!=0:
            Doing_pt = self.bck_to_orig(Doing_pt, bot_localizer.transform_arr, bot_localizer.rot_mat_rev)
            frame_disp = cv2.circle(frame_disp, (int(Doing_pt[0]),int(Doing_pt[1])), 3, (0,140,255),2)   
            #loc_car_ = self.bck_to_orig(loc_car, bot_localizer_obj.transform_arr, bot_localizer_obj.rot_mat_rev)
            #frame_disp = cv2.circle(frame_disp, (int(loc_car_[0]),int(loc_car_[1])), 3, (0,0,255))
         
            
        if Done_pt!=0:
            Done_pt = self.bck_to_orig(Done_pt, bot_localizer.transform_arr, bot_localizer.rot_mat_rev)
            if ( (type(path)!=int) and ( path_i!=(len(path)-1) ) ):
                pass
                #frame_disp = cv2.circle(frame_disp, (int(Done_pt[0]),int(Done_pt[1])) , 3, (0,255,0),2)   
            else:
                frame_disp = cv2.circle(frame_disp, (int(Done_pt[0]),int(Done_pt[1])) , 10, (0,255,0))  

        st = "len(path) = ( {} ) , path_iter = ( {} )".format(len(path),self.path_iter)        
        
        frame_disp = cv2.putText(frame_disp, st, (bot_localizer.orig_X+50,bot_localizer.orig_Y-30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255))
        cv2.imshow("maze (Shortest Path + Car Loc)",img_shortest_path)


    @staticmethod
    def angle_n_dist(pt_a,pt_b):
        # Trignometric rules Work Considering.... 
        #
        #       [ Simulation/Normal Convention ]      [ Image ]
        #
        #                    Y                    
        #                     |                     
        #                     |___                     ____ 
        #                          X                  |     X
        #                                             |
        #                                           Y
        #
        # Solution: To apply same rules , we subtract the (first) point Y axis with (Second) point Y axis
        error_x = pt_b[0] - pt_a[0]
        error_y = pt_a[1] - pt_b[1]

        # Calculating distance between two points
        distance = sqrt(pow( (error_x),2 ) + pow( (error_y),2 ) )

        # Calculating angle between two points [Output : [-Pi,Pi]]
        angle = atan2(error_y,error_x)
        # Converting angle from radians to degrees
        angle_deg = degrees(angle)

        if (angle_deg>0):
            return (angle_deg),distance
        else:
            # -160 +360 = 200, -180 +360 = 180,  -90 + 360 = 270
            return (angle_deg + 360),distance
        
        #             Angle bw Points 
        #      (OLD)        =>      (NEW) 
        #   [-180,180]             [0,360]

    def move_to_next_row(self,bot_loc,angle_to_turn,velocity,velocity_publisher):
        
        # Setting steering angle of bot proportional to the amount of turn it is required to take
        angle = interp(angle_to_turn,[-360,360],[-1,1])

        velocity.angular.z = angle
        #print("angle = ",angle)
        # Move forward when aligned to the destination
        if abs(angle) < 0.01:
            velocity.linear.x = 0.1
        else:
            velocity.linear.x = 0.0
        velocity_publisher.publish(velocity)

    
    def go_to_goal(self,bot_loc,path,max_dist,velocity,velocity_publisher):

        # Finding the distance and angle between (current) bot location and the (current) mini-goal
        angle_to_goal,distance_to_goal = self.angle_n_dist(bot_loc, (self.goal_pose_x,self.goal_pose_y))

        # Computing the angle the bot needs to turn to align with the mini goal
        angle_to_turn = angle_to_goal - self.bot_angle
        if angle_to_turn>180:
            angle_to_turn = -360 + angle_to_turn
        elif angle_to_turn<-180:
            angle_to_turn =  360 + angle_to_turn
        
        # Setting speed of bot proportional to its distance to the goal
        speed = interp(distance_to_goal,[0,max_dist],[0.05,0.35])
        self.curr_speed = round(speed,1)
        # Setting steering angle of bot proportional to the amount of turn it is required to take
        angle = interp(angle_to_turn,[-360,360],[-1,1])
        #self.curr_angle = round(angle,1)
        self.curr_angle = round(angle_to_turn,1)

        # If car is far away , turn towards goal
        if (distance_to_goal>=2):
            velocity.angular.z = angle

        # In view of limiation of differential drive, adjust speed of car with the amount of turn
        # E.g [Larger the turn  ==> Less the speed]
        if abs(angle) < 0.02:
            velocity.linear.x = speed
        else:
            velocity.linear.x = 0.0

        # Keep publishing the updated velocity until Final goal not reached
        #if (self.goal_not_reached_flag) or (distance_to_goal<=1):
        if (self.goal_not_reached_flag) or (distance_to_goal<=1):
            velocity_publisher.publish(velocity)

        # If car is within reasonable distance of mini-goal
        if ((distance_to_goal<=8)):

            velocity.linear.x = 0.0
            velocity.angular.z = 0.0
            # final goal not yet reached, stop moving
            if self.goal_not_reached_flag:
                velocity_publisher.publish(velocity)

            # Reached the final goal
            if self.path_iter==(len(path)-1):
                # First Time?
                if self.goal_not_reached_flag:
                    
                    # Stop Drone in its tracks
                    velocity.linear.x = 0.0
                    velocity_publisher.publish(velocity)

                    # Set goal_not_reached_flag to False
                    self.goal_not_reached_flag = False

                    
                    # Play the party song, Mention that reached goal
                    pygame.mixer.music.load(os.path.abspath('intelligent_drone/resource/Goal_reached.wav'))
                    pygame.mixer.music.play()
            # Still doing mini-goals?
            else:
                # Iterate over the next mini-goal
                self.path_iter += 1
                self.goal_pose_x = path[self.path_iter][0]
                self.goal_pose_y = path[self.path_iter][1]
                #print("Current Goal (x,y) = ( {} , {} )".format(path[self.path_iter][0],path[self.path_iter][1]))
                
                if pygame.mixer.music.get_busy() == False:
                    pygame.mixer.music.play()



    def nav_path(self,bot_loc,path,max_dist,velocity,velocity_publisher,state):
        
        
        # If valid path Founds
        if ((type(path)!=int) and path!=[]):
            # Trying to reach first mini-goal
            #if (self.path_iter==0):
            self.goal_pose_x = path[self.path_iter][0]
            self.goal_pose_y = path[self.path_iter][1]

            # Traversing through found path to reach goal
            self.go_to_goal(bot_loc,path,max_dist,velocity,velocity_publisher)
        elif ((path==[]) and (state=="Navigating_row")):
            print("Current State at motionplanner = {}".format(state))
            velocity.linear.x = 0.10
            velocity_publisher.publish(velocity)

        elif ((path==[]) and (state=="changing_row")):
            print("Current State at motionplanner = {}".format(state))
            #velocity.linear.x = 0.0
            velocity_publisher.publish(velocity)
        else:
            print("Current State at motionplanner = {}".format(state))
            velocity.linear.x = 0.0
            velocity_publisher.publish(velocity)


