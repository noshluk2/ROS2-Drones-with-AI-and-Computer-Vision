import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from utilities.utilities import get_centroid,remove_outliers,dist,find_line_parameters,dist_pt_line

rot_angle = 0

def OnRotChange(val):
    global rot_angle
    rot_angle = val

trans_X = 0

def OnTransXChange(val):
    global trans_X
    trans_X = val - 700

cv2.namedWindow("(2_d) field_mask_rot",cv2.WINDOW_NORMAL)
cv2.createTrackbar("Rot(Testing)","(2_d) field_mask_rot",0,360,OnRotChange)
cv2.createTrackbar("Trans_X(Testing)","(2_d) field_mask_rot",700,1400,OnTransXChange)

cv2.namedWindow("(2_d) field_mask_rot_draw",cv2.WINDOW_NORMAL)
cv2.namedWindow("(2_e) hue_edges_centroids",cv2.WINDOW_NORMAL)

def findextremas(cnts,img):

    if cnts:
        c = np.concatenate(cnts)
        
        #cnts = np.array(c)
        #ProjectedLane = np.zeros_like(img)
        #cv2.fillConvexPoly(ProjectedLane, cnts, 255)
        #cv2.imshow("Field (Estimated)",ProjectedLane)

        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        cv2.circle(img, extLeft, 9,(0,0,255))
        cv2.circle(img, extTop, 9,(0,255,0))
        cv2.circle(img, extRight, 9,(255,0,0))
        cv2.circle(img, extBot, 9,(255,0,255))

        return extLeft,extTop,extRight,extBot
    else:
        print("Empty Contours")
        return (0,0),(0,0),(0,0),(0,0)

def imfill(im_th,display_intermediates = False):
    #th, im_th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV);
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Display images.
    if display_intermediates:
        cv2.imshow("Thresholded Image", im_th)
        cv2.imshow("Floodfilled Image", im_floodfill)
        cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        cv2.imshow("Foreground", im_out)
        cv2.waitKey(0)

    im_out = remove_outliers(im_out,400)

    cnts = cv2.findContours(im_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    im_out_centroids = np.zeros_like(im_out)
    #im_out_bgr = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)

    prob_regs = []
    prob_cnts = []
    prob_centroids = []
    for idx,cnt in enumerate(cnts):
        if cv2.contourArea(cnt)<200:
            im_out = cv2.drawContours(im_out, cnts, idx, 0,-1)
            #cnt_center = get_centroid(cnt)
            #im_out_bgr = cv2.circle(im_out_bgr, cnt_center, 10, (0,0,255))
            #im_out_bgr = cv2.drawContours(im_out_bgr, cnts, idx, (0,0,255),-1)
        else:
            cnt_center = get_centroid(cnt)
            im_out_centroids = cv2.circle(im_out_centroids, cnt_center, 5, 255,-1)
            x,y,w,h = cv2.boundingRect(cnt)
            prob_regs.append([x,y,w,h])
            prob_cnts.append(cnt)
            prob_centroids.append(cnt_center)

    
    #cv2.imshow("im_out_bgr", im_out_bgr)
    #cv2.waitKey(0)
    
    return im_out,im_out_centroids,prob_regs,prob_cnts,prob_centroids

def find_nav_strt_bckup(img_centroids,rot_rect,rows,cols,centroids):
    
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
    nav_scnd_strt = (0,0)
    close_to_line_pts = []
    for plant_center in centroids:
        if plant_center!= nav_strt:
            #min_dist = dist(plant_center,nav_start_crnr) + dist(plant_center,mid_pt)
            min_dist = dist_pt_line(plant_center, m, b)
            if (prevmin_dist == 100_000 or (abs(min_dist - prevmin_dist) < 20) ):
                # Close enough
                close_to_line_pts.append(plant_center)
                prevmin_dist = min_dist
                nav_scnd_strt = plant_center

    if len(close_to_line_pts)>1:
        prevmin_dist = 100000
        for plant_center in close_to_line_pts:
            min_dist = dist(plant_center,nav_strt)
            if(min_dist < prevmin_dist):
                prevmin_dist = min_dist
                nav_scnd_strt = plant_center        


    return nav_strt,nav_scnd_strt


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



def ret_centroid(img,cnts):

    img_centroids = np.zeros_like(img)
    prob_centroids = []
    for idx,cnt in enumerate(cnts):
        cnt_center = get_centroid(cnt)
        img_centroids = cv2.circle(img_centroids, cnt_center, 5, 255,-1)
        prob_centroids.append(cnt_center)
    return prob_centroids,img_centroids

def identify_nav_strt(frame):
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
    M = cv2.getRotationMatrix2D((int(hue.shape[1]/2), int(hue.shape[0]/2)), rot_angle, 1.0)
    
    M[:,2] = [trans_X,0]
    #M[:,2] = [0,trans_X]
    field_mask_rot = cv2.warpAffine(field_mask_rot, M, (hue.shape[1], hue.shape[0]))
    field_mask_rot_draw = field_mask_rot.copy()

    # Drawing min_areaRect covering the rotated imaGe
    prob_cnts = cv2.findContours(field_mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cnts = np.concatenate(prob_cnts)
    rect = cv2.minAreaRect(cnts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(field_mask_rot_draw,[box],0,255,3)

    rot_text = "rect (Rotation) = "+ str(int(rect[2]))
    field_mask_rot_draw = cv2.putText(field_mask_rot_draw, rot_text, (100,600), cv2.FONT_HERSHEY_PLAIN, 3, 255,2)

    # Finding Extremas of the Field (Rotated)
    hue_edges_closed_bgr = cv2.cvtColor(field_mask_rot, cv2.COLOR_GRAY2BGR)
    
    extLeft,extTop,extRight,extBot = findextremas(prob_cnts,hue_edges_closed_bgr)
    rect = cv2.boundingRect(cnts)
    
    cv2.rectangle(field_mask_rot_draw, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), 255,3)
    cv2.rectangle(field_mask_rot_draw, (extLeft[0],extTop[1]), (extRight[0],extBot[1]), 255,3)
    
    prob_centroids,hue_edges_centroids = ret_centroid(field_mask_rot,prob_cnts)
    
    cv2.rectangle(hue_edges_centroids, (extLeft[0],extTop[1]), (extRight[0],extBot[1]), 255,3)
    cv2.drawContours(hue_edges_centroids,[box],0,255,3)

    nav_strt_pt,nav_strt_pt2 = find_nav_strt(hue_edges_centroids,box,hue.shape[0],hue.shape[1],prob_centroids)
    hue_edges_centroids = cv2.circle(hue_edges_centroids, nav_strt_pt, 16, 255,1)
    hue_edges_centroids = cv2.circle(hue_edges_centroids, nav_strt_pt2, 16, 255,2)

    cv2.imshow("(2_d) field_mask_rot",field_mask_rot)
    cv2.imshow("(2_d) field_mask_rot_draw",field_mask_rot_draw)
    cv2.imshow("(2_e) hue_edges_centroids",hue_edges_centroids)
 
    cv2.waitKey(1)

    return nav_strt_pt,nav_strt_pt2

def main():

    frame = cv2.imread("drone_view.png")

    while(1):
        identify_nav_strt(frame)


if __name__ == '__main__':
  main()
