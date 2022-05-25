import cv2
import sys


class TL_Tracker:

    def __init__(self):
        # Instance Variables
        print("Initialized Object of Goal class")

        self.tracker = cv2.TrackerCSRT_create()
        self.mode = "Detection"
        self.Tracked_class = "Unknown"
        self.tracked_bbox = [50,50,50,50]


    def track_and_retrive_goal_loc(self,frame,frame_draw):

        # 4. Checking if SignTrack Class mode is Tracking If yes Proceed
        if(self.mode == "Tracking"):

            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = self.tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame_draw, p1, p2, (255,0,0), 2, 1)

                self.tracked_bbox = bbox
            else :
                self.mode = "Detection"
                # Tracking failure
                cv2.putText(frame_draw, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display tracker type on frame
            cv2.putText(frame_draw,"Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
            # Display FPS on frame
            cv2.putText(frame_draw, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        

        # 3. If SignTrack is in Detection Proceed to intialize tracker
        elif (self.mode == "Detection"):

            # 3a. Select the ROI which u want to track
            r = cv2.selectROI("SelectROI",frame)
            cv2.destroyWindow("SelectROI")
            
            # Initialize tracker with first frame and bounding box
            ok = self.tracker.init(frame, r)
            if ok:
                self.mode = "Tracking" # Set mode to tracking
                self.Tracked_class = "Plant" # keep tracking frame sign name

