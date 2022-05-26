import cv2
import sys
from random import randint


class Tracker:

    def __init__(self):

        print("Initialized Object of Goal class")
        # Create MultiTracker object
        self.multiTracker = cv2.MultiTracker_create()
        self.mode = "Detection"
        self.Tracked_class = "Unknown"
        self.tracked_bboxes = []
        self.colors =[]

    def track_and_retrive_goal_loc(self,frame,frame_draw):

        # 4. Checking if SignTrack Class mode is Tracking If yes Proceed
        if(self.mode == "Tracking"):

            # Start timer
            timer = cv2.getTickCount()

            # get updated location of objects in subsequent frames
            success, boxes = self.multiTracker.update(frame)
            if success:
                self.tracked_bboxes = []
                for rct in boxes:
                    #rct = boxes[0]
                    self.tracked_bboxes.append((round(rct[0],1),round(rct[1],1),round(rct[2],1),round(rct[3],1)))

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            # draw tracked objects
            if success:
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame_draw, p1, p2, self.colors[i], 3, 1)
            else:
                self.mode = "Detection"
                # Tracking failure
                cv2.putText(frame_draw, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        
            # Display FPS on frame
            cv2.putText(frame_draw, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        

        # 3. If SignTrack is in Detection Proceed to intialize tracker
        elif (self.mode == "Detection"):
            
            # Select boxes
            bboxes = []

            # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
            # So we will call this function in a loop till we are done selecting all objects
            while True:
                # draw bounding boxes over objects
                # selectROI's default behaviour is to draw box starting from the center
                # when fromCenter is set to false, you can draw box starting from top left corner
                bbox = cv2.selectROI('MultiTracker', frame)
                bboxes.append(bbox)
                self.colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                k = cv2.waitKey(0) & 0xFF
                print (k)
                if (k == 113):  # q is pressed
                    cv2.destroyWindow('MultiTracker')
                    break

            print('Selected bounding boxes {}'.format(bboxes))
            
            # Initialize MultiTracker
            for bbox in bboxes:
                if bbox != (0,0,0,0):
                    print("bbox = ", bbox)
                    tracker = cv2.TrackerKCF_create()
                    self.multiTracker.add(tracker, frame, bbox)
                    self.mode = "Tracking" # Set mode to tracking
                    self.Tracked_class = "Plant" # keep tracking frame sign name

