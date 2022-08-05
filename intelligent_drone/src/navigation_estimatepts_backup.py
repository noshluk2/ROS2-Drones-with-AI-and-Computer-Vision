import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from utilities.utilities import get_centroid,remove_outliers,dist,find_line_parameters,imshow,imshow_stages,imshow_stage
from utilities.utilities import dist_pt_line,ret_centroid,imfill,estimate_pt,closest_node,rotate_point

from trackers.goal_tracker import Tracker
from motionplanners.drone_motionplanning import drone_motionplanner

from math import atan,degrees,tan,radians

from collections import deque
class navigator():
    def __init__(self):
        self.state = "init"

        self.Tracker_ = Tracker()
        self.drone_motionplanner = drone_motionplanner()

        self.prev_elevation = 0

        # Current visited plant location
        self.plant_tagged = (0,0)
        # Used to estimate the next plant in the same row
        self.plants_tagged = deque(maxlen=2)
        # Container to store locations of all plants in a row
        self.plants_in_a_row = []
        # Container to store location of all plants in the field
        self.plants_in_the_field = []

        self.curr_goal = (0,0)

        self.drone_strt_loc_map = (0,0)

        self.drone_loc_map = []
        self.est_nxtplant = []

        self.avg_plant_width = []
        self.cam_offset_img = None

        self.curr_row_slope = None
        self.curr_row_yintercept = None

        self.goal_iter = 0
        self.entered_goal_vicinity = False

        self.entered_goal_vicinity_map = False


        # Set to true when drone is navigating from one row to another
        self.nav_to_second_row = False
        # Set to true when drone is navigating to the start of the next row
        self.moving_to_row_start = False
        # Set to true when drone has already reached the row start
        self.reached_row_strt = False

        self.dist_from_bbox = None
        self.aligned_to_row_strt = False


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

    def identify_nav_strt(self,drone_view):
        # Converting to hls color space to make light invariant
        hls = cv2.cvtColor(drone_view, cv2.COLOR_BGR2HLS)
        hue = hls[:,:,0]
        
        # Identifying objects with stronger edges
        hue_edges = cv2.Canny(hue, 50, 150,None,3)

        # Removing noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        hue_edges_closed = cv2.morphologyEx(hue_edges, cv2.MORPH_CLOSE, kernel)
        
        # Keeping only largercontours and retreiving their centroids
        field_mask_rot,hue_edges_centroids,prob_regs,prob_cnts,prob_centroids = imfill(hue_edges_closed)
        
        ## Rotate our image around an arbitrary point rather than the center
        #field_mask_rot = hue_edges_closed.copy()

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

        image_list = [drone_view,hue_edges,hue_edges_closed,field_mask_rot,
                      field_mask_rot_draw,hue_edges_centroids]
        #imshow_stage(image_list,"identify_nav_strt")

        bboxes = []
        if ((nav_strt_pt!= (0,0)) and (nav_strt_pt2!= (0,0))):
            # Retreive bbox surrounding the conoutrs represented by the centroid
            for cnt in prob_cnts:
                _,_,w,h= cv2.boundingRect(cnt)
                self.avg_plant_width.append(max(w,h))
                if (cv2.pointPolygonTest(cnt, nav_strt_pt, False)==1):
                    bboxes.append(cv2.boundingRect(cnt))
            
            # Retreive bbox surrounding the conoutrs represented by the centroid
            for cnt in prob_cnts:
                if (cv2.pointPolygonTest(cnt, nav_strt_pt2, False)==1):
                    bboxes.append(cv2.boundingRect(cnt))
        self.avg_plant_width = sum(self.avg_plant_width)/len(self.avg_plant_width)
        print("self.avg_plant_width(pixels) = ",self.avg_plant_width)

        return bboxes

    def identify_row_strt(self,drone_view,mask,lst_plantloc_calc,corrected_slope):
        # Converting to hls color space to make light invariant
        hls = cv2.cvtColor(drone_view, cv2.COLOR_BGR2HLS)
        hue = hls[:,:,0]
        
        # Identifying objects with stronger edges
        hue_edges = cv2.Canny(hue, 40, 150,None,3)

        # Removing noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        hue_edges_closed = cv2.morphologyEx(hue_edges, cv2.MORPH_CLOSE, kernel)
        
        # Keeping only largercontours and retreiving their centroids
        field_mask_rot,hue_edges_centroids,prob_regs,prob_cnts,prob_centroids = imfill(hue_edges_closed)
        
        ## Rotate our image around an arbitrary point rather than the center
        #field_mask_rot = hue_edges_closed_filled.copy()

        # Drawing min_areaRect covering the rotated imaGe
        prob_cnts = cv2.findContours(field_mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnts = np.concatenate(prob_cnts)
        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        field_mask_rot_draw = field_mask_rot.copy()
        cv2.drawContours(field_mask_rot_draw,[box],0,255,3)
        rot_text = "rect (Rotation) = "+ str(int(rect[2]))
        field_mask_rot_draw = cv2.putText(field_mask_rot_draw, rot_text, (100,600), cv2.FONT_HERSHEY_PLAIN, 3, 255,2)


        prob_centroids,hue_edges_centroids = ret_centroid(field_mask_rot,prob_cnts)

    
        # [Drawing]
        hue_edges_centroids_draw = hue_edges_centroids.copy()        
        rect = cv2.boundingRect(cnts)
        cv2.rectangle(field_mask_rot_draw, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), 255,3)
        cv2.rectangle(hue_edges_centroids_draw, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), 255,3)   
        cv2.drawContours(hue_edges_centroids_draw,[box],0,255,3)

        # Only Showing those plants and centroid that are unvisited in fields
        field_mask_rot_Unvisited = cv2.bitwise_and(field_mask_rot,field_mask_rot,mask=mask)
        hue_edges_centroids_Unvisited = cv2.bitwise_and(hue_edges_centroids,hue_edges_centroids,mask=mask)

        prob_cnts_unvisited = cv2.findContours(field_mask_rot_Unvisited, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        prob_centroids_Unvisited,hue_edges_centroids_Unvisited = ret_centroid(field_mask_rot_Unvisited,prob_cnts_unvisited)
        
        # Checking correctioness in localization of Last plant of the previous row
        lst_plantloc_onimg_idx = closest_node(lst_plantloc_calc, prob_centroids)
        lst_plantloc_actual = prob_centroids[lst_plantloc_onimg_idx]

        # Assuming [ Unvisited Area would include plant_rows where the drone has not been +
        #            Plants are mostly aligned, Closest node would represent the adjacent plant in
        #            the next row to the last plant of the current row.But because of inconsistency in mask
        #                                                             we will use location identified to get 
        #                                                             its precise location using the centroids
        #                                                             extracted from the original drone_view (no_mask applied)
        closest_plant_idx_Unvisited = closest_node(lst_plantloc_calc, prob_centroids_Unvisited)
        closest_plant_idx = closest_node(prob_centroids_Unvisited[closest_plant_idx_Unvisited], prob_centroids)
        closest_plant_centroid = prob_centroids[closest_plant_idx]
        
        # Identifying the plant to visit next (Would be the first plant in the adjacent unvisited row)
        drone_view_nxtrow_plant = drone_view.copy()
        cv2.drawContours(drone_view_nxtrow_plant, prob_cnts, closest_plant_idx, (0,128,0),-1)
        cv2.circle(drone_view_nxtrow_plant, closest_plant_centroid, 4, (0,0,255),4) 
        cv2.circle(drone_view_nxtrow_plant, lst_plantloc_actual, 4, (0,255,0),4) 
        cv2.circle(drone_view_nxtrow_plant, lst_plantloc_calc, 4, (0,128,255),4) # last plant in the current row loc (More precise)

        lstplant_offset = (lst_plantloc_calc[0] - lst_plantloc_actual[0],lst_plantloc_calc[1] - lst_plantloc_actual[1])
        lstplant_absoffset = (abs(lst_plantloc_calc[0] - lst_plantloc_actual[0]),abs(lst_plantloc_calc[1] - lst_plantloc_actual[1]))
        lstplant_maxoffset = max(lstplant_absoffset[0],lstplant_absoffset[1])
        self.cam_offset_img = lstplant_offset[lstplant_absoffset.index(lstplant_maxoffset)]
        print("self.cam_offset_img = ",self.cam_offset_img)
        
        # [TESTING]: Line perpendicular to row slope is being calculated based on formula = (-1/plantrow_slope) [Recent change: Corrected from + to - of inverse of slope]
        # Target : Was looking to find the area to look for next rows based on [Last Plant On Img , plant_row_slope_perp, found_closestPlant]
        print("# Target : Was looking to find the area to look for next rows based on [Last Plant On Img , plant_row_slope_perp, found_closestPlant]")
        rows_offset    = (    closest_plant_centroid[0] - lst_plantloc_actual[0] ,    closest_plant_centroid[1] - lst_plantloc_actual[1] )
        rows_absoffset = (abs(closest_plant_centroid[0] - lst_plantloc_actual[0]),abs(closest_plant_centroid[1] - lst_plantloc_actual[1]))
        rows_absmaxoffset = max(rows_absoffset[0],rows_absoffset[1])
        
        # idx   [0 or 1] ==> [col or row] 
        rows_maxoffset_idx = rows_absoffset.index(rows_absmaxoffset)
        # value [+ or -] ==> [inc or dec]
        rows_maxoffset_value = rows_offset[rows_maxoffset_idx]

        # y = mx + b
        r,c = drone_view.shape[0:2]
    
        pt_a = lst_plantloc_actual
        row_slope_perp = -(1 / self.curr_row_slope)
        # b              =  y(row) - (      m       * x(col))
        yiCntercept_perp = pt_a[1] - (row_slope_perp*pt_a[0])

        print("row_slope_perp = ",row_slope_perp)
        
        if (abs(row_slope_perp)<1):
            # Horizontal line ===> Look in x diorection for next point
            if rows_maxoffset_value>=0:
                x = c
            else:
                x = 0
            y = int ( (row_slope_perp * x) + yiCntercept_perp )
            print("row_slope_perp is less then 1 ==> Horizontal Line")
        else:
            # vertical line ===> Look in y direction for next point
            if rows_maxoffset_value>=0:
                y = r
            else:
                y = 0
            x = int ( (y - yiCntercept_perp) / row_slope_perp ) 
            print("row_slope_perp is greater then 1 ==> Vertical Line")

        pt_b = (x,y)
        print("pt_b = ",pt_b)
        print("Computed in conditions = [rows_maxoffset_value] = {}".format(rows_maxoffset_value))
        cv2.line(drone_view_nxtrow_plant, pt_a, pt_b, (0,50,0),int(self.avg_plant_width*2))
        

        # [Double Check]: Based on the estimated row direction, 
        #                             we double checked area of the field that is unvisited...
        estimated_nxt_rows_mask = np.zeros_like(mask)
        cv2.line(estimated_nxt_rows_mask, pt_a, pt_b, 255,int(self.avg_plant_width*2))
        mask = cv2.bitwise_and(mask, estimated_nxt_rows_mask)
        estimated_nxt_rows_mask = cv2.bitwise_and(field_mask_rot,field_mask_rot,mask=mask)

        # Plant that wee need to track is present at this indiex
        if closest_plant_idx!=-1:
            p_bbox = cv2.boundingRect(prob_cnts[closest_plant_idx])
        else:
            p_bbox = []

        
        image_list = [drone_view,hue_edges,hue_edges_closed,field_mask_rot,field_mask_rot_Unvisited,
                      hue_edges_centroids,hue_edges_centroids_Unvisited,drone_view_nxtrow_plant,estimated_nxt_rows_mask]
        imshow_stage(image_list,"identify_row_strt",corrected_slope)

        return p_bbox


    def is_plant_present(self,c_slam_map_draw):
        # Determining the curr goal for drone.
        goal_vicinity = 5        

        # If drone is far from its goal but has already entered goal vicinity
        if ((dist(self.drone_loc_map,self.est_nxtplant)>goal_vicinity) and self.entered_goal_vicinity_map):
            # Reset entered_goal_vicinity boolean
            self.entered_goal_vicinity_map = False
            return False
        # If drone is far from its goal
        elif (dist(self.drone_loc_map,self.est_nxtplant)>goal_vicinity):
            print("Still Far from GOall ......")
        # If drone is within goal vicinity , then set boolean to True.
        else:
            self.entered_goal_vicinity_map = True

        txt = str(self.drone_loc_map)+" <-> "+str(self.est_nxtplant)+" = "+ str(int(dist(self.drone_loc_map,self.est_nxtplant)))
        c_slam_map_draw = cv2.rectangle(c_slam_map_draw, (300,140), (700,180), (0,0,0),-1)
        cv2.putText(c_slam_map_draw, txt, (350,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
        
        return True

    # Function to find what each detected corner represented in the initial fov points
    def find_closest_on_fov(self,corners_unvisited,fov_cnts):
        
        closest_pts_on_fov = []
        closest_pts_indices = [0] * len(corners_unvisited)
        for i,corner in enumerate(corners_unvisited):
            # Estimating start as the closest road to car
            closest_idx = closest_node(corner,fov_cnts[0],5)
            #closest_corner = (corners_unvisited[closest_idx].tolist())[0]

            closest_pts_indices[i] = closest_idx
            if closest_idx!=-1:
                closest_pts_on_fov.append((fov_cnts[0][closest_idx][0][0],fov_cnts[0][closest_idx][0][1]))

        print("closest_pts_on_fov = ",closest_pts_on_fov)

        return closest_pts_on_fov,closest_pts_indices

    
    def est_line(self,drone_view,c_slam_map_draw,orientation,gsd_dm,fov_unrot_pts,fov_pts):
        
        if(len(self.plants_in_a_row)>1):# Atleast 2 points needed to estimate a line

            x = np.array([plant_loc[0] for plant_loc in self.plants_in_a_row])
            y = np.array([plant_loc[1] for plant_loc in self.plants_in_a_row])
            print("x = {} \ny = {} ".format(x,y))
            parameters,residuals = (np.polyfit(x, y, 1,full=True))[0:2]
            slope = parameters[0]
            yiCntercept = parameters[1]
            
            corrected_slope = False
            if residuals>1000:
                print("residuals (Error) is toooo high !!!!= ",residuals)
                print("(Initial) slope",slope)

                print("#### Correcting Slope!")
                parameters,residuals = (np.polyfit(y, x, 1,full=True))[0:2]

                if residuals<1000:
                    print("residuals (Error) is less in the inverse case!!!!= ",residuals)
                    # Slope will now be the negative of inverse of the perpendicular slope as x and y were swapped
                    slope = -(1/parameters[0])
                    # Interestingly , if a abs(slope)>50 means angle >88deg we have a vertical line. Meaning
                    #                                 Meaning: Equation of line is x = a (Here a can be taken from any point column)
                    if abs(slope)>50:
                        # Vertical line found
                        # Reference (Slope-intercpt form of a vertical line): https://www.quora.com/How-do-you-write-a-vertical-line-in-slope-intercept-form
                        print("Vertical Line found ! .... Basicaly y intercept is not important anymore")
                    # b         = y    -      m*x
                    yiCntercept = y[0] - (slope*x[0])
                    print("#### (Corrected) Slope!",slope)
                    print("#### (Corrected) yiCntercept!",yiCntercept)
                    corrected_slope = True


            print("slope",slope)
            print("yiCntercept",yiCntercept)
            
            self.curr_row_slope         = slope
            # Value of drone orientation can be directly be passed to member variable of same name 
            #     as orientation is of type float which is an "immutable" object meaning modifying ones value would
            #     cause creation of a new reference of that new value without modifying the other pointers pointing to old reference
            self.curr_drone_orientation = orientation
            self.curr_row_yintercept    = yiCntercept

            h,w = c_slam_map_draw.shape[:2]
            if slope!='NA':
                ### here we are essentially extending the line to x=0 and x=width
                ### and calculating the y associated with it
                ##starting point
                #y = slope*x + b
                if slope <= 1:
                    px=0
                    py = (slope*px)+yiCntercept
                    ##ending point
                    qx=w
                    qy = (slope*qx)+yiCntercept
                elif abs(slope)>50:
                    # We have a vertical line [So it spans from top of image to bottom]
                    #                         [On same column as all other points.....]
                    px = x[0] # Same column as point_a
                    py = 0    # Starting from Image top (0 height)
                    qx = x[0] # Same column as point_a
                    qy = h    # Ending at Image bottom (full height)

                else:
                    # x = (y-b)/slope
                    py = 0
                    px = (py-yiCntercept)/slope
                    ## ending point
                    qy = h
                    qx = (qy-yiCntercept)/slope
            else:
                ### if slope is zero, draw a line with x=x1 and y=0 and y=height
                px,py=x,0
                qx,qy=x,h


            plant_wid_dm = self.avg_plant_width*gsd_dm
            print("Plant width in Fov on map is {} dm".format(plant_wid_dm))
            cv2.line(c_slam_map_draw, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), int(plant_wid_dm))
            
            drone_trajectory = np.zeros((c_slam_map_draw.shape[0],c_slam_map_draw.shape[1]),np.uint8)
            cv2.line(drone_trajectory, (int(px), int(py)), (int(qx), int(qy)), 255, int(plant_wid_dm))
            drone_trajectory=imfill(drone_trajectory,False,self.drone_strt_loc_map)[0]
            cv2.imshow("drone (trajectory) ",drone_trajectory)
            
            drone_fov = np.zeros_like(drone_trajectory)
            cv2.fillPoly(drone_fov, [fov_pts], 255)
            cv2.imshow("drone_fov ",drone_fov)

            fov_cnts = cv2.findContours(drone_fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

            #drone_fov_unvisited = cv2.bitwise_and(drone_trajectory, drone_trajectory,mask = drone_fov)
            drone_fov_unvisited = cv2.subtract(drone_fov,drone_trajectory)
            drone_fov_unvisited[drone_fov_unvisited>0] = 255
            cv2.imshow("drone_fov (Unvisited) ",drone_fov_unvisited)

            cnts = cv2.findContours(drone_fov_unvisited, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

            peri = cv2.arcLength(cnts[0], True)
            corners_unvisited = cv2.approxPolyDP(cnts[0], 0.04 * peri, True)
            print("corners_unvisited (ApproxPoly) = ",corners_unvisited)


            #[Clockwise closest pts]
            drone_fov_unvisited_pts_fov = cv2.cvtColor(drone_fov, cv2.COLOR_GRAY2BGR)
            #closest_pts_on_fov = self.find_closest_on_fov(corners_unvisited,fov_cnts)[0]
            
            # Finding the two corners that actually were part of initial fov of order
            # Idea: Drone would have followed a particular trajectory, we estimated that and masked out areas 
            #       that were already visited, this means the corners that we have are not the corners approx
            #       from the complete drone fov. but from drone fov (Unvisited) so likely half present. Meaning
            #       we would have one side with 2 corners of the original fov but the other are new corners made.
            #          [0,1,2,3]       =  [Top,b-left,end,top-right]
            # Correct Output: Two Unvisited corners matched
            closest_pts_on_fov,closest_pts_indices = self.find_closest_on_fov(corners_unvisited,[fov_pts])
            
            for pt in closest_pts_on_fov:
                cv2.circle(drone_fov_unvisited_pts_fov, (pt[0],pt[1]), 8, (255, 255, 0), -1)
            cv2.imshow("drone_fov (Unvisited) pts_fov!",drone_fov_unvisited_pts_fov)
            
            drone_fov_unvisited_pts_unrot_fov = drone_fov_unvisited_pts_fov.copy()
            cv2.polylines(drone_fov_unvisited_pts_unrot_fov, [fov_unrot_pts], True, (0,0,255),3)
            cv2.imshow("drone_fov (Unvisited) pts_unrot_fov!",drone_fov_unvisited_pts_unrot_fov)
            print("closest_pts_indices = ",closest_pts_indices)

            # list representing corners_unvisited(tuple) of the drone view in clockwise order [Top,b-left,end,top-right]
            drone_view_corner_list = [(0,0),(0,drone_view.shape[0]),(drone_view.shape[1],drone_view.shape[0]),(drone_view.shape[1],0)]


            corner_indices = [0,1,2,3]            
            notinlist = list(set(corner_indices)-set(closest_pts_indices)) 
            print("notinlist = ",notinlist)

            drone_fov_unvisited_pts_unrot_fov2 = drone_fov_unvisited_pts_fov.copy()

            corner_unrot_unknown = []
            for idx in notinlist:
                corner = corners_unvisited[idx]
                corner_unrot = rotate_point(0, 0, -orientation-90, (corner[0][0],corner[0][1]))
                print("corner_unrot = ",corner_unrot)
                corner_unrot_unknown.append(corner_unrot)
                cv2.circle(drone_fov_unvisited_pts_unrot_fov2, corner_unrot, 8, (0, 0, 255), -1)
            
            
            if corner_unrot_unknown[0][1]<corner_unrot_unknown[1][1]:
                pt_X_top = corner_unrot_unknown[0]
                pt_Y_btm = corner_unrot_unknown[1]

            else:
                pt_X_top = corner_unrot_unknown[1]
                pt_Y_btm = corner_unrot_unknown[0]

            cv2.putText(drone_fov_unvisited_pts_unrot_fov2, "pt_X", (pt_X_top[0]-60,pt_X_top[1]-30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
            cv2.putText(drone_fov_unvisited_pts_unrot_fov2, "pt_Y", (pt_Y_btm[0]-60,pt_Y_btm[1]+30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
            
            cv2.imshow("drone_fov (Unvisited) pts_unrot_fov2!",drone_fov_unvisited_pts_unrot_fov2)

            itr = 0
            for idx,pt_indx in enumerate(closest_pts_indices):
                if pt_indx!=-1:
                    print("pt we know (drone_view) [{}]  = {}".format(idx,drone_view_corner_list[idx]))
                else:
                    print("pt we need to find (drone_view) [{}]  = {}".format(idx,drone_view_corner_list[notinlist[itr]]))
                    itr+=1

            print("distance between TopLeft    <---> pt_X = {}, dist(image) = {}".format(dist(fov_unrot_pts[0][0], pt_X_top),int(dist(fov_unrot_pts[0][0], pt_X_top)/gsd_dm)))
            print("distance between BottomLeft <---> pt_Y = {}, dist(image) = {}".format(dist(fov_unrot_pts[1][0], pt_Y_btm),int(dist(fov_unrot_pts[1][0], pt_Y_btm)/gsd_dm)))
    
            top_dist = int(dist(fov_unrot_pts[0][0], pt_X_top)/gsd_dm)
            btm_dist = int(dist(fov_unrot_pts[1][0], pt_Y_btm)/gsd_dm)
            droneView_mask = np.zeros((drone_view.shape[0],drone_view.shape[1]),np.uint8)
            
            if notinlist[0] == (2 or 3):
                msk_toplft = drone_view_corner_list[0]
                msk_btmlft = drone_view_corner_list[1]
                msk_end    = (drone_view_corner_list[1][0]+btm_dist,drone_view_corner_list[1][1])
                msk_toprgt = (drone_view_corner_list[0][0]+top_dist,drone_view_corner_list[0][1])
            else:
                msk_toplft = (drone_view_corner_list[3][0]-top_dist,drone_view_corner_list[3][1])
                msk_btmlft = (drone_view_corner_list[2][0]-btm_dist,drone_view_corner_list[2][1])
                msk_end    = drone_view_corner_list[2]
                msk_toprgt = drone_view_corner_list[3]                

            # Clockwise polygon points creation
            droneView_mask_pts = np.array([np.asarray(msk_toplft),np.asarray(msk_btmlft),
                                           np.asarray(msk_end)   ,np.asarray(msk_toprgt)]
                                           ,np.int32)     
            droneView_mask_pts = droneView_mask_pts.reshape((-1, 1, 2))
            cv2.fillPoly(droneView_mask, [droneView_mask_pts], 255)
            cv2.imshow("droneView_mask (Unvisited) ",droneView_mask)
            
            lst_plant_in_row_unrotated = self.plants_in_a_row[len(self.plants_in_a_row)-1]
            fov_center = get_centroid(fov_cnts[0])
            lst_plant_in_row = rotate_point(fov_center[0], fov_center[1], orientation+90, (lst_plant_in_row_unrotated[0],lst_plant_in_row_unrotated[1]))
            print("Last plant Loc (Origin) = {} , (Drone-Pose_Aligned) = {}".format(lst_plant_in_row_unrotated,lst_plant_in_row))
            pt_test = cv2.pointPolygonTest(fov_pts,lst_plant_in_row,False)
            if pt_test>0:
                print("Last plant can be seen in the drone view")
                fov_center = get_centroid(fov_cnts[0])
                print("fov_center = ",fov_center)
                cv2.circle(c_slam_map_draw, fov_center, 4, (255,0,255),2)
                distance_from_center = dist(fov_center, lst_plant_in_row)
                print("fov_center = ",fov_center)
                lst_plant_offset = (lst_plant_in_row[0] - fov_center[0],lst_plant_in_row[1] - fov_center[1])
                print("lst_plant_offset = ",lst_plant_offset)
                lstplnt_x_off_img = int(lst_plant_offset[0]/gsd_dm)
                lstplnt_y_off_img = int(lst_plant_offset[1]/gsd_dm)
                img_cntr = (int(drone_view.shape[1]/2),int(drone_view.shape[0]/2))
                droneView_mask_bgr = cv2.cvtColor(droneView_mask, cv2.COLOR_GRAY2BGR)

                # Calculated location of last plant in the current row... Later we will compare this with the acutal location to estimate the error in our prediction
                lst_plantloc_calc = (img_cntr[0] + lstplnt_x_off_img,img_cntr[1] + lstplnt_y_off_img)
                cv2.circle(droneView_mask_bgr, lst_plantloc_calc, 8, (0,0,255),4)

                cv2.line(c_slam_map_draw, fov_center, lst_plant_in_row, (0,128,0),2)
                a = fov_center
                b = lst_plant_in_row
                dist_pt = (int((a[0]+b[0])/2),int((a[1]+b[1])/2))
                cv2.putText(c_slam_map_draw, str(int(distance_from_center)), dist_pt, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
                
                imshow("droneView_mask_bgr",droneView_mask_bgr)

            nxt_row_bbox = self.identify_row_strt(drone_view, droneView_mask,lst_plantloc_calc,corrected_slope)

        else:
            nxt_row_bbox = []

        return nxt_row_bbox


    def est_nxt_row(self):

        print("First row had following characteristics [Slope = {}, Y-intercept = {}, Drone Orientation = {}]".format(self.curr_row_slope,self.curr_row_yintercept,self.curr_drone_orientation))

        h,w = image.shape[:2]
        if slope!='NA':
            ### here we are essentially extending the line to x=0 and x=width
            ### and calculating the y associated with it
            ##starting point
            #y = slope*x + b
            if slope <= 1:
                px=0
                py = (slope*px)+yiCntercept
                ##ending point
                qx=w
                qy = (slope*qx)+yiCntercept
            elif abs(slope)>50:
                # We have a vertical line [So it spans from top of image to bottom]
                #                         [On same column as all other points.....]
                px = x[0] # Same column as point_a
                py = 0    # Starting from Image top (0 height)
                qx = x[0] # Same column as point_a
                qy = h    # Ending at Image bottom (full height)

            else:
                # x = (y-b)/slope
                py = 0
                px = (py-yiCntercept)/slope
                ## ending point
                qy = h
                qx = (qy-yiCntercept)/slope
        else:
            ### if slope is zero, draw a line with x=x1 and y=0 and y=height
            px,py=x,0
            qx,qy=x,h


        cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (128, 0, 255), 2)


    def est_nxt_row2(self,image_draw,bbox_cntr,orientation):
        
        r,c = image_draw.shape[0:2]
        drone_loc = (int(image_draw.shape[1]/2),int(image_draw.shape[0]/2))

        # [TESTING]: Line perpendicular to row slope is being calculated based on formula = (-1/plantrow_slope) [Recent change: Corrected from + to - of inverse of slope]
        # Target : Was looking to find the area to look for next rows based on [Last Plant On Img , plant_row_slope_perp, found_closestPlant]
        print("# Target : Was looking to find the area to look for next rows based on [Last Plant On Img , plant_row_slope_perp, found_closestPlant]")
        rows_offset    = (    drone_loc[0] - bbox_cntr[0] ,    drone_loc[1] - bbox_cntr[1] )
        rows_absoffset = (abs(drone_loc[0] - bbox_cntr[0]),abs(drone_loc[1] - bbox_cntr[1]))
        rows_absmaxoffset = max(rows_absoffset[0],rows_absoffset[1])
        # idx   [0 or 1] ==> [col or row] 
        rows_maxoffset_idx = rows_absoffset.index(rows_absmaxoffset)
        # value [+ or -] ==> [inc or dec]
        rows_maxoffset_value = rows_offset[rows_maxoffset_idx]

        pt_a = bbox_cntr
        
        row_init_orientation = degrees(atan(self.curr_row_slope))# row inital orientation (deg) 
        offset_init_orientation = row_init_orientation - self.curr_drone_orientation
        slope = tan( radians( offset_init_orientation  + orientation ) )
        y_inter_init = self.curr_row_yintercept
        
        # y intercept computed from point a
        # b              =  y(row) - (      m       * x(col))
        y_inter = pt_a[1] - (slope*pt_a[0])
        print("#############################  TESTING #####################################")
        print("Initial slope = ",self.curr_row_slope)
        print("Initial y_inter = ",self.curr_row_yintercept)

        print("slope = ",slope)
        print("y_inter = ",y_inter)
        print("#############################  TESTING #####################################")

        if (abs(slope)<1):
            # Horizontal line ===> Look in x diorection for next point
            if rows_maxoffset_value>=0:
                x = c
            else:
                x = 0
            y = int ( (slope * x) + y_inter )
            print("slope is less then 1 ==> Horizontal Line")
        else:
            # vertical line ===> Look in y direction for next point
            if rows_maxoffset_value>=0:
                y = r
            else:
                y = 0
            x = int ( (y - y_inter) / slope ) 
            print("slope is greater then 1 ==> Vertical Line")

        pt_b = (x,y)
        print("pt_b = ",pt_b)
        # Once drone starts moving to Start of next row then start distance from bbox should not change
        #                                                    Only its other component be computed on change of drone orientation
        if not self.aligned_to_row_strt:
            dist_strt = int( bbox_cntr[rows_maxoffset_idx] + (rows_maxoffset_value/2) )
            self.dist_from_bbox = dist_strt
        if rows_maxoffset_idx==0:
            # Distance is greater in cols So (MaxOffset) is in cols
            x_strt = self.dist_from_bbox# Set x_point as half of the distance in columns
            y_strt = int ( (slope * x_strt) + y_inter )
        else:
            # Distance is greater in rows So (MaxOffset) is in rows
            y_strt = self.dist_from_bbox
            # y = mx + b ==> x = (y-b)     /   m
            x_strt = int( (y_strt-y_inter) / slope )
        
        pt_strt = (x_strt,y_strt)

        # Finding the point of intersection between drone and the estimated next row, This will be our start loc
        x_intersect = drone_loc[0]# At this particular column , Where will the line be at which row?
        y_intersect = int ( (slope * x_intersect) + y_inter )
        pt_intersect = (x_intersect,y_intersect)
        print("[Test] drone_loc = ",drone_loc)
        print("[Test] pt_intersect = ",pt_intersect)
        print("[Test] pt_strt = ",pt_strt)

        print("Computed in conditions = [rows_maxoffset_value] = {}".format(rows_maxoffset_value))
        #cv2.line(image_draw, pt_a, pt_b, (0,50,0),int(self.avg_plant_width*2))
        cv2.line(image_draw, pt_a, pt_b, (0,0,255),4)
        txt = "rows_maxoffset_idx = "+str(rows_maxoffset_idx)
        cv2.putText(image_draw,txt,(50,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255),2)
        txt = "self.moving_to_row_start = "+str(self.moving_to_row_start)
        cv2.putText(image_draw,txt,(50,250), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255),2)
        imshow("Estimated Next Row", image_draw)

        return pt_strt


    def navigate(self,drone_view,frame_draw,vel_msg,vel_pub,elevation,c_slam_map,c_slam_map_draw,fov_unrot_pts,fov_pts,d_pose,gsd_dm):

        drone_loc = (int(drone_view.shape[1]/2),int(drone_view.shape[0]/2))
        max_dist = min(drone_view.shape[0],drone_view.shape[1])
        path = []
        
        if self.state == "init":
            self.prev_elevation = elevation
            bboxes = self.identify_nav_strt(drone_view)
            if bboxes!=[]:
                # Found starting Points , Start traversing field
                self.state = "Start"
                self.Tracker_.track_multiple(drone_view, frame_draw, bboxes)
                self.goal_iter = 0

        elif self.state == "Start":

            self.Tracker_.track_multiple(drone_view, frame_draw)
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
        
        elif self.state == "Start-Row":
            print("[State]: Start-Row!")

            self.Tracker_.track_multiple(drone_view, frame_draw)
            # [Drone: MotionPlanning] Reach the (maze exit) by navigating the path previously computed
            if ( len(self.Tracker_.tracked_bboxes)==1 ):
                for bbox in self.Tracker_.tracked_bboxes:
                    center_x = int(bbox[0] + (bbox[2]/2))
                    center_y = int(bbox[1] + (bbox[3]/2))             
                    path.append((center_x,center_y))

            if path!=[]:
                if not self.drone_motionplanner.goal_not_reached_flag:
                    #Reached final Goal ===> PARRTTTYYY !!
                    self.state = "Navigating_row"
                    self.goal_iter = 0       


        elif self.state == "Navigating_row":
            print("[State]: Navigating Row")
            if not self.is_plant_present(c_slam_map_draw):
                print("Plant not found at estimated location .... Search for next row!!!")
                self.state = "Searching"

        elif self.state == "Searching":
            print("[State]: Searching....")
            # Use prior plants to estimate the general row line
            nxt_row_bbox = self.est_line(drone_view,c_slam_map_draw,d_pose[2],gsd_dm,fov_unrot_pts,fov_pts)
            
            if nxt_row_bbox!=[]:
                # Found starting Points , Start traversing field
                self.state = "changing_row"
                # # Set to true when drone is navigating from one row to another
                # self.nav_to_second_row = False
                # # Set to true when drone is navigating to the start of the next row
                # self.moving_to_row_start = False
                # # Set to true when drone has already reached the row start
                # self.reached_row_strt = False
                # Initialize tracker with a single bbox
                self.Tracker_.mode = "Detection"
                self.Tracker_.track_multiple(drone_view, frame_draw, [nxt_row_bbox])

                self.goal_iter = 0
                # Append found plants row in the plants_in_the_field_container
                self.plants_in_the_field.append(self.plants_in_a_row)
        
        elif self.state == "changing_row":
            print("[State]: Changing row!")

            self.Tracker_.track_multiple(drone_view, frame_draw)

            # Estimating the second row using [tracked bbox_center as the point +
            #                                    using estimated slope of row]
            bbox = self.Tracker_.tracked_bboxes[0]
            # Centroid of bbox is at ((x+w)/2,(y+h)/2)
            bbox_w,bbox_h = bbox[2:4]
            bbox_cntr = ( int( bbox[0] + (bbox_w/2) ) , int( bbox[1] + (bbox_h/2) ) )

            pt_strt = self.est_nxt_row2(drone_view.copy(), bbox_cntr,d_pose[2])
            #cv2.waitKey(0)
            
            #angle_to_trn = int(degrees(atan(-(1 / self.curr_row_slope))))
            angle_to_goal = (degrees(atan(-(1 / self.curr_row_slope))))
            # Computing the angle the bot needs to turn to align with the mini goal
            drone_orientation   = d_pose[2]
            angle_to_turn = angle_to_goal - drone_orientation 
            #print("drone_orientation = ",drone_orientation)
            print("angle_to_turn = ",angle_to_turn)
            self.drone_motionplanner.move_to_next_row(drone_loc, angle_to_turn, vel_msg, vel_pub)
            # [Drone: MotionPlanning] Reach the (maze exit) by navigating the path previously computed
            if ( len(self.Tracker_.tracked_bboxes)==1 ):
                bbox = self.Tracker_.tracked_bboxes[0]
                # Centroid of bbox is at ((x+w)/2,(y+h)/2)
                bbox_w = bbox[2]
                bbox_h = bbox[3]
                bbox_loc = ( int( bbox[0] + (bbox_w/2) ) , int( bbox[1] + (bbox_h/2) ) )
                cv2.circle(frame_draw, bbox_loc, 4, (255,0,0),2)
                cv2.circle(frame_draw, drone_loc, 4, (255,255,255),2)
                cv2.circle(frame_draw, pt_strt, 4, (0,128,255),3)

                txt = "drone_loc = "+str(drone_loc)
                txt2 = "bbox_loc = "+str(bbox_loc)
                cv2.putText(frame_draw, txt, (50,300), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)             
                cv2.putText(frame_draw, txt2, (50,350), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)   
                print("angle_to_turn = {}, drone_loc[1] ={}, bbox_loc[1] = {}".format(angle_to_turn,drone_loc[1],bbox_loc[1]))
                
                

                if ( (angle_to_turn<5) or self.nav_to_second_row ):
                    self.nav_to_second_row = True

                    if self.reached_row_strt:
                        self.state = "Start-Row"
                        path = [bbox_loc]
                    # Row aligns with destination row (Almost)
                    elif ( (abs(drone_loc[1]-bbox_loc[1])<5) or self.moving_to_row_start ):
                        self.moving_to_row_start = True
                        path = [pt_strt]

                        # Testing
                        angle_to_goal_strt = self.drone_motionplanner.angle_n_dist(drone_loc, pt_strt)[0]
                        angle_to_turn_strt = angle_to_goal_strt - drone_orientation
                        if (angle_to_turn_strt<5): 
                            # We are now aligned to row start
                            self.aligned_to_row_strt = True

                        # About to reach the row-start
                        if dist(drone_loc, pt_strt)<=10:
                            # Check if we have reached goal
                            if not self.drone_motionplanner.goal_not_reached_flag:
                                print("Reached row start ... Proceed to first plant!")
                                self.drone_motionplanner.goal_not_reached_flag = True # Reason we set this to true so that next condition doesnot execute. 
                                                                                      #   Which basically if executed will reset everything
                                self.reached_row_strt = True

                        # Initially. We Switch goal_not_reached from False to True. To activate go-to-goal (start)
                        if not self.drone_motionplanner.goal_not_reached_flag:
                            # Reset motionplanner for new go-to-goal call
                            self.drone_motionplanner.path_iter = 0
                            # Set goal not reached to true so that gotogoal() leads to the new goal
                            self.drone_motionplanner.goal_not_reached_flag = True
                            self.goal_iter = 0
                            # Resetting plant tagged for start of new row
                            self.plant_tagged = (0,0)
                            self.plants_tagged.clear()
                            self.plants_in_a_row.clear()
                        
                    


        self.drone_motionplanner.nav_path(drone_loc, path, max_dist, vel_msg, vel_pub,self.state)

        # [Drawing] Displaying Current goal as a green circle on the occupency Grid
        curr_goal = (self.drone_motionplanner.goal_pose_x,self.drone_motionplanner.goal_pose_y)
        frame_draw = cv2.circle(frame_draw, curr_goal, 5, (0,255,0),3)

        # [Drawing] Displaying current state (statistics) on the occupency grid
        cv2.putText(frame_draw, "[Angle,Speed] = ", (50,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, str(self.drone_motionplanner.curr_angle), (320,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, str(self.drone_motionplanner.curr_speed), (420,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        if self.Tracker_.tracked_bboxes != []:
            cv2.putText(frame_draw, str(self.Tracker_.tracked_bboxes[0]), (250,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        cv2.putText(frame_draw, self.Tracker_.mode, (50,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        
        # [Drawing] Computing distance to goal for drone in (drone_view)
        dist_to_goal = int(dist(drone_loc,self.curr_goal))
        # [Drawing] Displaying distance to goal in blue text on occupancy grid
        cv2.putText(c_slam_map_draw, str(dist_to_goal), (350,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),3)
        
        # Determining the curr goal for drone.
        goal_vicinity = 60        
        if path!=[]:
            print("Current path = ",path)
            # Determining curr goal from path using goal_iter [member of navigator clasa]
            self.curr_goal = (path[self.goal_iter][0],path[self.goal_iter][1])
            # If drone is far from its goal but has already entered goal vicinity
            if ((dist(drone_loc,self.curr_goal)>goal_vicinity) and self.entered_goal_vicinity):
                # Drone has exited goal area , We are now moving to the next goal
                if (self.goal_iter<len(path)-1):
                    # we can only iterate to the next goal, if there is next goal present in path
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
        cv2.putText(c_slam_map_draw, plants_tg_txt, (50,400), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)

        d_x,d_y,_ = d_pose
        c_x = int(c_slam_map_draw.shape[1]/2)
        c_y = int(c_slam_map_draw.shape[0]/2)
        self.drone_loc_map = (-d_x + c_x , d_y + c_y)
        if (self.state!="init"): 
            # If drone is within a goal vicinity - & - 
            #       detects an anomaly (sudden elevation change) ===> Probable Plant detected
            if ( (dist(drone_loc,self.curr_goal)<goal_vicinity) and (abs(elevation - self.prev_elevation) > 0.5) ):
                # Only proceed to identifying plant if it is not the same one as previously found
                #       (Based on distance to previously detected)
                if (dist((d_x,d_y),self.plant_tagged)>10):
                    prob_plant = (-d_x + c_x , d_y + c_y)
                    # Displaying detected plant using Sonar on Map (Imp 4 future reference)
                    cv2.circle(c_slam_map, prob_plant, 2, (255,0,0),6)
                    cv2.circle(c_slam_map_draw, prob_plant, 2, (255,0,0),6)
                    self.plant_tagged = (d_x,d_y)
                    self.plants_tagged.append(prob_plant)
                    self.plants_in_a_row.append(prob_plant)
                    if len(self.plants_tagged)==2:
                        self.entered_goal_vicinity_map = False
                        self.est_nxtplant = estimate_pt(self.plants_tagged[0],self.plants_tagged[1],"end")
                        cv2.putText(c_slam_map_draw, str(self.est_nxtplant), (50,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),3)
                        cv2.circle(c_slam_map_draw, self.est_nxtplant, 2, (0,255,0),6)
                else:
                    print("Close to previously Found Goal")
        else:
            # Initial state 
            self.drone_strt_loc_map = self.drone_loc_map


        cv2.imshow("C-SLAM (map)[Draw]",c_slam_map_draw)




