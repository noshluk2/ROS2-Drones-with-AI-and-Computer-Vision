import cv2
import numpy as np

import os
import math

class TL_Tracker:

    def __init__(self):
        # Instance Variables
        print("Initialized Object of signTracking class")

    # Class Variables
    mode = "Detection"
    max_allowed_dist = 100
    feature_params = dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
    lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))  
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    known_centers = []
    known_centers_confidence = []
    old_gray = 0
    p0 = []
    Tracked_class = 0
    mask = 0
    Tracked_ROI=0
    CollisionIminent = False

    tracked_bbox = [50,50,50,50]

    def Distance(self,a,b):
        return math.sqrt( ( (float(a[1])-float(b[1]))**2 ) + ( (float(a[0])-float(b[0]))**2 ) )

    def MatchCurrCenter_ToKnown(self,center):
        match_found = False
        match_idx = 0
        for i in range(len(self.known_centers)):
            if ( self.Distance(center,self.known_centers[i]) < self.max_allowed_dist ):
                match_found = True
                match_idx = i
                return match_found, match_idx
        # If no match found as of yet return default values
        return match_found, match_idx

    def santitze_pts(self,pts_src,pts_dst):
        # Idea was to Order on Descending Order of Strongest Points [Strength here is 
        # considered when two points have minimum distance between each other]
        pt_idx = 0
        dist_list = []
        for pt in pts_src:
            pt_b = pts_dst[pt_idx]
            dist_list.append(self.Distance(pt,pt_b))
            pt_idx+=1

        pts_src_list = pts_src.tolist()
        pts_dst_list = pts_dst.tolist()

        pts_src_list = [x for _, x in sorted(zip(dist_list, pts_src_list))]
        pts_dst_list = [x for _, x in sorted(zip(dist_list, pts_dst_list))]

        pts_src = np.asarray(pts_src_list, dtype=np.float32)
        pts_dst = np.asarray(pts_dst_list, dtype=np.float32)

        return pts_src,pts_dst

    
    def EstimateTrackedRect(self,im_src,pts_src,pts_dst,img_draw):
        Tracking = "Tracking"
        im_dst = np.zeros_like(im_src)

        if(len(pts_src)>=3):
            pts_src,pts_dst = self.santitze_pts(pts_src,pts_dst)
            pts_src = pts_src[0:3][:]
            pts_dst = pts_dst[0:3][:]
            
            M = cv2.getAffineTransform(pts_src, pts_dst)
            im_dst = cv2.warpAffine(im_src, M ,(im_dst.shape[1],im_dst.shape[0]),flags=cv2.INTER_CUBIC)

            img_dst_2 = np.zeros_like(im_dst)

            kernel = np.ones((2,2), dtype=np.uint8)
            closing = cv2.morphologyEx(im_dst, cv2.MORPH_CLOSE, kernel)
            
            cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE )[0]
            cnt = max(cnts, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(cnt)
            if ( abs( (x+w) - im_src.shape[1] ) < (0.3*im_src.shape[1]) ):
                self.CollisionIminent = True
                
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            #cntr = rect[0]  # (x,y) or (col,row)
            #col = cntr[0]-int((rect[1][0])/2)
            #row = cntr[1]-int((rect[1][1])/2)
            #self.tracked_bbox = [col,row,rect[1][0],rect[1][1]]
            self.tracked_bbox = [x,y,w,h]

            cv2.drawContours(img_dst_2,[box],0,255,-1)

            # Drawing Tracked Traffic Light Rect On Original Image
            cv2.drawContours(img_draw,[box],0,(255,0,0),2)

            #https://stackoverflow.com/questions/39371507/image-loses-quality-with-cv2-warpperspective
            # Smoothing by warping is caused by interpolation
            #im_dst = cv2.warpAffine(im_src, M ,(im_dst.shape[1],im_dst.shape[0]))

        else:
            print("Points less then 3, Error!!!")
            #cv2.waitKey(0)
            Tracking = "Detection"
            # Set Img_dst_2 to Already saved Tracked Roi One last Time
            img_dst_2 = self.Tracked_ROI
            self.CollisionIminent = False # Reset
        
        return im_dst,img_dst_2,Tracking

    def Track(self,frame,frame_draw):

        Temp_Tracked_ROI = self.Tracked_ROI
        # 4a. Convert Rgb to gray
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        Text2display = "OpticalFlow ( " + self.mode + " )"
        cv2.putText(frame_draw,Text2display,(20,150),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

        # Localizing Potetial Candidates and Classifying them in SignDetection
        # 4b. Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None,**self.lk_params)
        
        # 4c. If no flow, look for new points
        if p1 is None:
            self.mode = "Detection"
            self.mask = np.zeros_like(frame_draw)
            self.Reset()

        # 4d. If points tracked, Display and Update SignTrack class    
        else:
            # Select good points
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            self.Tracked_ROI,Temp_Tracked_ROI,self.mode = self.EstimateTrackedRect(self.Tracked_ROI,good_old,good_new,frame_draw)
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = (int(x) for x in new.ravel())
                c, d = (int(x) for x in old.ravel())
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)
                frame_draw = cv2.circle(frame_draw, (a, b), 5, self.color[i].tolist(), -1)
            frame_draw_ = frame_draw + self.mask# Display the image with the flow lines
            np.copyto(frame_draw,frame_draw_)#important to copy the data to same address as frame_draw   
            self.old_gray = gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
        #cv2.imshow("frame_draw",frame_draw)
        return Temp_Tracked_ROI

    def Reset(self):
        
        self.known_centers = []
        self.known_centers_confidence = []
        self.old_gray = 0
        self.p0 = []


    def track_and_retrive_goal_loc(self,img,frame_draw):

        # 4. Checking if SignTrack Class mode is Tracking If yes Proceed
        if(self.mode == "Tracking"):

            Temp_Tracked_ROI = self.Track(img,frame_draw)
            
            #cv2.imshow("[Fetch_TL_State] (4) Tracked_ROI",self.Tracked_ROI)

            img_ROI_tracked = cv2.bitwise_and(img,img,mask=Temp_Tracked_ROI)
            
            #cv2.imshow('[Fetch_TL_State] (5) img_ROI_tracked_BoundedRect', img_ROI_tracked)

            #cv2.imshow('[Fetch_TL_State] (3) Traffic Light With State', frame_draw)


        # 3. If SignTrack is in Detection Proceed to intialize tracker
        elif (self.mode == "Detection"):

            # 3a. Select the ROI which u want to track
            r = cv2.selectROI("SelectROI",img)
            cv2.destroyWindow("SelectROI")
            TLD_Class = "Plant"

            if ((r!=np.array([0,0,0,0])).all()):
                # Traffic Light Detected ===> Initialize Tracker 
                # 3b. Convert Rgb to gray
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # 3c. creating ROI mask
                ROI_toTrack = np.zeros_like(gray)
                ROI_toTrack[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255

                self.Tracked_ROI = ROI_toTrack
                # 3d. Updating signtrack class with variables initialized
                self.mode = "Tracking" # Set mode to tracking
                self.Tracked_class = TLD_Class # keep tracking frame sign name
                #if Mask_ClrRmvd is None:
                self.p0 = cv2.goodFeaturesToTrack(gray, mask = ROI_toTrack, **self.feature_params)
                #else:
                #    self.p0 = cv2.goodFeaturesToTrack(gray, mask = Mask_ClrRmvd, **self.feature_params)
                self.old_gray = gray.copy()
                self.mask = np.zeros_like(frame_draw)
                self.CollisionIminent = False

