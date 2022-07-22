import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from utilities.utilities import get_centroid,remove_outliers,dist,find_line_parameters
from utilities.utilities import dist_pt_line,ret_centroid,imfill,estimate_pt

from trackers.goal_tracker import Tracker
from motionplanners.drone_motionplanning import drone_motionplanner

from collections import deque
class navigator():
    def __init__(self):
        self.state = "init"

        self.Tracker_ = Tracker()
        self.drone_motionplanner = drone_motionplanner()

        self.prev_elevation = 0

        self.plant_tagged = (0,0)
        self.plants_tagged = deque(maxlen=2)
        self.curr_goal = (0,0)

        self.goal_iter = 0
        self.entered_goal_vicinity = False



    cv2.namedWindow("(2_d) field_mask_rot",cv2.WINDOW_NORMAL)
    cv2.namedWindow("(2_d) field_mask_rot_draw",cv2.WINDOW_NORMAL)
    cv2.namedWindow("(2_e) hue_edges_centroids",cv2.WINDOW_NORMAL)

    @staticmethod
    def find_nav_strt(img_centroids,rot_rect,rows,cols,centroids):
        
        nav_col = 0
        nav_row = 0
        
        nav_start_crnr = (0,0)
        strt_idx = 0
        prevmin_dist = 100000
        for idx,pt in enumerate(rot_rect):
            min_dist = dist(pt,(int(cols/2),int(rows/2)))
            if(min_dist < prevmin_dist):
                prevmin_dist = min_dist
                nav_start_crnr = pt
                strt_idx = idx

        crnt = rot_rect[strt_idx]
        frst = rot_rect[(strt_idx+1)%4]
        lst = rot_rect[(strt_idx+3)%4]
        scnd = (0,0)
        mid_pt = (0,0)
        if (dist(crnt,frst) <= dist(crnt,lst)):
            # last to current is the longer side taking that as second crner and computing mid
            mid_pt =  (int((crnt[0]+lst[0])/2),int((crnt[1]+lst[1])/2))
            scnd = lst
        else:
            # next to current is the longer side taking that as second crner and computing mid
            mid_pt =  (int((crnt[0]+frst[0])/2),int((crnt[1]+frst[1])/2))
            scnd = frst

        cv2.circle(img_centroids, mid_pt, 16, 255,1)

        m,b = find_line_parameters(crnt,scnd)

        prevmin_dist = 100000
        nav_strt = (0,0)
        for plant_center in centroids:
            min_dist = dist(plant_center,nav_start_crnr)
            if(min_dist < prevmin_dist):
                prevmin_dist = min_dist
                nav_strt = plant_center

        prevmin_dist = 100_000
        closest_pt = (0,0)
        for plant_center in centroids:
            if plant_center!= nav_strt:
                min_dist = dist_pt_line(plant_center, m, b)
                if(min_dist < prevmin_dist):
                    closest_pt = plant_center
                    prevmin_dist = min_dist
        #print("prevmin_dist =",prevmin_dist)
        close_to_line_pts = [closest_pt]
        for plant_center in centroids:
            if ( (plant_center!= nav_strt) and (plant_center!= closest_pt) ):
                min_dist = dist_pt_line(plant_center, m, b)
                if (abs(min_dist - prevmin_dist) < 40):
                    # Close enough
                    close_to_line_pts.append(plant_center)


        nav_scnd_strt = closest_pt
        if len(close_to_line_pts)>1:
            prevmin_dist = 100000
            for plant_center in close_to_line_pts:
                min_dist = dist(plant_center,nav_strt)
                if(min_dist < prevmin_dist):
                    prevmin_dist = min_dist
                    nav_scnd_strt = plant_center   


        return nav_strt,nav_scnd_strt

    def identify_nav_strt(self,frame):
        # Converting to hls color space to make light invariant
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hue = hls[:,:,0]
        
        # Identifying objects with stronger edges
        hue_edges = cv2.Canny(hue, 50, 150,None,3)

        # Removing noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        hue_edges_closed = cv2.morphologyEx(hue_edges, cv2.MORPH_CLOSE, kernel)
        
        # Keeping only largercontours and retreiving their centroids
        hue_edges_closed,hue_edges_centroids,prob_regs,prob_cnts,prob_centroids = imfill(hue_edges_closed)
        
        # Rotate our image around an arbitrary point rather than the center
        field_mask_rot = hue_edges_closed.copy()

        field_mask_rot_draw = field_mask_rot.copy()

        # Drawing min_areaRect covering the rotated imaGe
        prob_cnts = cv2.findContours(field_mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnts = np.concatenate(prob_cnts)
        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # [Drawing]
        cv2.drawContours(field_mask_rot_draw,[box],0,255,3)
        rot_text = "rect (Rotation) = "+ str(int(rect[2]))
        field_mask_rot_draw = cv2.putText(field_mask_rot_draw, rot_text, (100,600), cv2.FONT_HERSHEY_PLAIN, 3, 255,2)


        prob_centroids,hue_edges_centroids = ret_centroid(field_mask_rot,prob_cnts)

        # [Drawing]
        rect = cv2.boundingRect(cnts)
        cv2.rectangle(field_mask_rot_draw, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), 255,3)
        cv2.rectangle(hue_edges_centroids, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), 255,3)   
        cv2.drawContours(hue_edges_centroids,[box],0,255,3)

        nav_strt_pt,nav_strt_pt2 = self.find_nav_strt(hue_edges_centroids,box,hue.shape[0],hue.shape[1],prob_centroids)
        
        # [Drawing]    
        hue_edges_centroids = cv2.circle(hue_edges_centroids, nav_strt_pt, 16, 255,1)
        hue_edges_centroids = cv2.circle(hue_edges_centroids, nav_strt_pt2, 16, 255,2)
        cv2.imshow("(2_d) field_mask_rot",field_mask_rot)
        cv2.imshow("(2_d) field_mask_rot_draw",field_mask_rot_draw)
        cv2.imshow("(2_e) hue_edges_centroids",hue_edges_centroids)
        cv2.waitKey(1)

        bboxes = []
        
        if ((nav_strt_pt!= (0,0)) and (nav_strt_pt2!= (0,0))):
            # Retreive bbox surrounding the conoutrs represented by the centroid
            for cnt in prob_cnts:
                if (cv2.pointPolygonTest(cnt, nav_strt_pt, False)==1):
                    bboxes.append(cv2.boundingRect(cnt))
            
            # Retreive bbox surrounding the conoutrs represented by the centroid
            for cnt in prob_cnts:
                if (cv2.pointPolygonTest(cnt, nav_strt_pt2, False)==1):
                    bboxes.append(cv2.boundingRect(cnt))

        return bboxes

    def navigate(self,frame,frame_draw,vel_msg,vel_pub,elevation,occupncy_grd,d_pose):

        drone_loc = (int(frame.shape[1]/2),int(frame.shape[0]/2))
        max_dist = min(frame.shape[0],frame.shape[1])
        path = []
        
        if self.state == "init":
            self.prev_elevation = elevation
            bboxes = self.identify_nav_strt(frame)
            if bboxes!=[]:
                # Found starting Points , Start traversing field
                self.state = "Start"
                self.Tracker_.track_multiple(frame, frame_draw, bboxes)
                self.goal_iter = 0

        elif self.state == "Start":

            self.Tracker_.track_multiple(frame, frame_draw)
            # [Drone: MotionPlanning] Reach the (maze exit) by navigating the path previously computed
            if ( len(self.Tracker_.tracked_bboxes)==2 ):
                for bbox in self.Tracker_.tracked_bboxes:
                    center_x = int(bbox[0] + (bbox[2]/2))
                    center_y = int(bbox[1] + (bbox[3]/2))             
                    path.append((center_x,center_y))
            
            if path!=[]:
                if self.drone_motionplanner.goal_not_reached_flag:
                    pt_c = estimate_pt(path[0],path[1])
                    path = [pt_c] + path
                else:
                    #Reached final Goal ===> PARRTTTYYY !!
                    self.state = "Navigating_row"
                    self.goal_iter = 0
        
        elif self.state == "Navigating_row":
            print("Navigating Row")


        self.drone_motionplanner.nav_path(drone_loc, path, max_dist, vel_msg, vel_pub,self.state)

        # [Drawing] Displaying Current goal as a green circle on the occupency Grid
        curr_goal = (self.drone_motionplanner.goal_pose_x,self.drone_motionplanner.goal_pose_y)
        frame_draw = cv2.circle(frame_draw, curr_goal, 5, (0,255,0),3)

        # [Drawing] Displaying current state (statistics) on the occupency grid
        cv2.putText(frame_draw, "[Angle,Speed] = ", (50,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, str(self.drone_motionplanner.curr_angle), (320,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, str(self.drone_motionplanner.curr_speed), (420,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, str(self.Tracker_.tracked_bboxes[0]), (250,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, self.Tracker_.mode, (50,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        
        # [Drawing] Computing distance to goal for drone in (drone_view)
        dist_to_goal = int(dist(drone_loc,self.curr_goal))
        occupncy_grd = cv2.rectangle(occupncy_grd,  (300,450),  (350+100,500+100), (0,0,0),-1)
        # [Drawing] Displaying distance to goal in blue text on occupancy grid
        cv2.putText(occupncy_grd, str(dist_to_goal), (350,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        
        # Determining the curr goal for drone.
        goal_vicinity = 60        
        if path!=[]:
            # Determining curr goal from path using goal_iter [member of navigator clasa]
            self.curr_goal = (path[self.goal_iter][0],path[self.goal_iter][1])
            # If drone is far from its goal but has already entered goal vicinity
            if ((dist(drone_loc,self.curr_goal)>goal_vicinity) and self.entered_goal_vicinity):
                # Drone has exited goal area , We are now moving to the next goal
                self.goal_iter +=1
                # Reset entered_goal_vicinity boolean
                self.entered_goal_vicinity = False
            # If drone is far from its goal
            elif (dist(drone_loc,self.curr_goal)>goal_vicinity):
                print("Still Far from GOall ......")
            # If drone is within goal vicinity , then set boolean to True.
            else:
                self.entered_goal_vicinity = True
        plants_tg_txt = "plants_tagged = "+ str(len(self.plants_tagged))
        occupncy_grd = cv2.rectangle(occupncy_grd,  (0,350),  (0+400,350+70), (0,0,0),-1)
        cv2.putText(occupncy_grd, plants_tg_txt, (50,400), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)

        #m,b = find_line_parameters(crnt,scnd)
        d_x,d_y,_ = d_pose
        c_x = int(occupncy_grd.shape[1]/2)
        c_y = int(occupncy_grd.shape[0]/2)
        if (self.state!="init"): 
            # If drone is within a goal vicinity - & - 
            #       detects an anomaly (sudden elevation change) ===> Probable Plant detected
            if ( (dist(drone_loc,self.curr_goal)<goal_vicinity) and (abs(elevation - self.prev_elevation) > 0.5) ):
                # Only proceed to identifying plant if it is not the same one as previously found
                #       (Based on distance to previously detected)
                if (dist((d_x,d_y),self.plant_tagged)>10):
                    prob_plant = (-d_x + c_x , d_y + c_y)
                    cv2.circle(occupncy_grd, prob_plant, 2, (255,0,0),6)
                    self.plant_tagged = (d_x,d_y)
                    self.plants_tagged.append(prob_plant)
                    if len(self.plants_tagged)==2:
                        prob_nxtplant = estimate_pt(self.plants_tagged[0],self.plants_tagged[1],"end")
                        cv2.putText(occupncy_grd, str(prob_nxtplant), (50,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)
                        cv2.circle(occupncy_grd, prob_nxtplant, 2, (0,255,0),6)
                else:
                    print("Close to previously Found Goal")

