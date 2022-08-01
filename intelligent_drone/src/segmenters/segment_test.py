import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from utilities import get_centroid,remove_outliers,dist,find_line_parameters,dist_pt_line,imshow

th1 = 50

def On_th1_Change(val):
    global th1
    th1 = val

th2 = 150

def On_th2_Change(val):
    global th2
    th2 = val

aperture = 3

def On_aperture_Change(val):
    global aperture
    aperture = (val*2) + 1

k_size = 5

def On_ksize_Change(val):
    global k_size
    k_size = (val*2) + 1

sigmaX = 0

def On_sigmaX_Change(val):
    global sigmaX
    sigmaX = val

cv2.namedWindow("hue_edges",cv2.WINDOW_NORMAL)
cv2.createTrackbar("th_1","hue_edges",0,255,On_th1_Change)
cv2.createTrackbar("th_2","hue_edges",0,255,On_th2_Change)
cv2.createTrackbar("ap_size","hue_edges",1,3,On_aperture_Change)

cv2.namedWindow("hue_blur",cv2.WINDOW_NORMAL)
cv2.createTrackbar("k_size","hue_blur",1,3,On_ksize_Change)
cv2.createTrackbar("sigmaX","hue_blur",1,3,On_sigmaX_Change)


def findextremas(cnts,img):

    if cnts:
        c = np.concatenate(cnts)
        
        #cnts = np.array(c)
        #ProjectedLane = np.zeros_like(img)
        #cv2.fillConvexPoly(ProjectedLane, cnts, 255)
        #imshow("Field (Estimated)",ProjectedLane)

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
        imshow("Thresholded Image", im_th)
        imshow("Floodfilled Image", im_floodfill)
        imshow("Inverted Floodfilled Image", im_floodfill_inv)
        imshow("Foreground", im_out)
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

    
    #imshow("im_out_bgr", im_out_bgr)
    #cv2.waitKey(0)
    
    return im_out,im_out_centroids,prob_regs,prob_cnts,prob_centroids

def ret_centroid(img,cnts):

    img_centroids = np.zeros_like(img)
    prob_centroids = []
    for idx,cnt in enumerate(cnts):
        cnt_center = get_centroid(cnt)
        img_centroids = cv2.circle(img_centroids, cnt_center, 5, 255,-1)
        prob_centroids.append(cnt_center)
    return prob_centroids,img_centroids



def segment(frame):
    hue_blur = cv2.GaussianBlur(frame, (k_size,k_size), sigmaX)
    imshow("frame_blur", hue_blur)

    # Converting to hls color space to make light invariant
    hls = cv2.cvtColor(hue_blur, cv2.COLOR_BGR2HLS)
    hue = hls[:,:,0]
    lig = hls[:,:,1]
    sat = hls[:,:,2]
    imshow("hue", hue)
    imshow("lig", lig)
    imshow("sat", sat)
    #hue_blur = cv2.GaussianBlur(hue, (k_size,k_size), sigmaX)
    #imshow("hue_blur", hue_blur)

    # Identifying objects with stronger edges
    hue_edges = cv2.Canny(hue, th1, th2,None,aperture)
    imshow("hue_edges", hue_edges)

    # Removing noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    hue_edges_closed = cv2.morphologyEx(hue_edges, cv2.MORPH_CLOSE, kernel)
    imshow("hue_edges_closed", hue_edges_closed)
    
    # Keeping only largercontours and retreiving their centroids
    field_mask_rot,hue_edges_centroids,prob_regs,prob_cnts,prob_centroids = imfill(hue_edges_closed)

    field_mask_rot_draw = field_mask_rot.copy()

    # Drawing min_areaRect covering the rotated imaGe
    prob_cnts = cv2.findContours(field_mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(prob_cnts)!=0:
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
        imshow("field_mask_rot_draw",field_mask_rot_draw)
        imshow("hue_edges_centroids",hue_edges_centroids)

    imshow("field_mask_rot",field_mask_rot)
 
    cv2.waitKey(1)


def main():

    frame = cv2.imread("drone_view.png")
    r = cv2.selectROI("SelectROI",frame)
    frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    while(1):
        segment(frame)


if __name__ == '__main__':
  main()
