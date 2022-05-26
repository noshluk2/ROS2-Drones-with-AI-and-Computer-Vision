import cv2
import time
import math
import numpy as np
from matplotlib import pyplot as plt
import os

import tensorflow as tf # tensorflow imported to check installed tf version
from tensorflow.keras.models import load_model # import load_model function to load trained CNN model for Sign classification

from utilities.utilities import get_centroid,remove_outliers,dist

model_loaded = False
model = 0
#sign_classes = ["speed_sign_30","speed_sign_60","speed_sign_90","stop","left_turn","No_Sign"] # Trained CNN Classes
is_plant = [False,True]

l_thrsh = 50
h_thrsh = 150
aperture = 3

rot_angle = 0
def OnLowThreshChange(val):
    global l_thrsh
    l_thrsh = val

def OnHighThreshChange(val):
    global h_thrsh
    h_thrsh = val

def OnApertureChange(val):
    global aperture
    aperture = (val*2)+3

#cv2.namedWindow("(2_c) hue_edges_closed",cv2.WINDOW_NORMAL)

#cv2.createTrackbar("l_thrsh","(2_c) hue_edges_closed",l_thrsh,255,OnLowThreshChange)
#cv2.createTrackbar("h_thrsh","(2_c) hue_edges_closed",h_thrsh,255,OnHighThreshChange)
#cv2.createTrackbar("aperture","(2_c) hue_edges_closed",0,2,OnApertureChange)

def OnRotChange(val):
    global rot_angle
    rot_angle = val


cv2.namedWindow("(2_d) field_mask_rot",cv2.WINDOW_NORMAL)
cv2.createTrackbar("Rot(Testing)","(2_d) field_mask_rot",0,360,OnRotChange)

cv2.namedWindow("(2_d) field_mask_rot_draw",cv2.WINDOW_NORMAL)

cv2.namedWindow("(2_e) hue_edges_centroids",cv2.WINDOW_NORMAL)


def plot_multiple_hists(img_list,Block = False,Pause = 0.0001):

    no_of_images = len(img_list)
    fig = plt.figure()

    for i in range(no_of_images):
        image = img_list[i]
        ax = fig.add_subplot(no_of_images, 1, i+1)
        ax.hist(image.ravel(),256,[0,256])
    
    plt.show(block = Block)
    if not Block:
        plt.pause(0.0001)
    
def stretch_histogram(image,method="LUT"):

    image_hist_stretched = image.copy()

    if method == "LUT":
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        image_hist_stretched = cv2.LUT(image_hist_stretched, table)

    else:
        # normalize float versions
        norm_img1 = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = cv2.normalize(image, None, alpha=0, beta=3, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # scale to uint8
        norm_img1 = (255*norm_img1).astype(np.uint8)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)

        # display input and both output images
        cv2.imshow('original',image)
        cv2.imshow('normalized1',norm_img1)
        cv2.imshow('normalized2',norm_img2)
        cv2.waitKey(0)


    return norm_img1,norm_img2

def line_detect(gray,thresh = 350,lower_thresh = 5,upper_thresh= 40):
    #gray_blur = cv2.GaussianBlur(gray,(5,5),sigmaX=0)
    # Apply edge detection method on the image
    dst = cv2.Canny(gray,lower_thresh,upper_thresh,apertureSize = 3)
    #dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(dst, 1, np.pi / 180, thresh, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    # linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 150, None, 50, 10)
    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    return cdst
    
def segment_crop(frame):

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hue = hls[:,:,0]

    #hue_hist_stretched,hue_hist_stretched_2 = stretch_histogram(hue,"No")
    #img_hist_streching = [hue,hue_hist_stretched,hue_hist_stretched_2]
    
    hue_hist_equalized = cv2.equalizeHist(hue)

    img_his_equalized = [hue,hue_hist_equalized]
    plot_multiple_hists(img_his_equalized)

    hue_lines = line_detect(hue)
    hue_equalized_lines = line_detect(hue_hist_equalized,lower_thresh=50,upper_thresh=180)
    hue_edges = cv2.Canny(hue, 50, 150,None,3)
    otsu_hue_mask = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges_hue_otsu = cv2.Canny(otsu_hue_mask, 50, 150,None,3)

    cv2.imshow("(2) hue",hue)
    cv2.imshow("(2) hue_hist_equalized",hue_hist_equalized)
    cv2.imshow("(2) hue_lines",hue_lines)
    cv2.imshow("(2) hue_equalized_lines",hue_equalized_lines)
    cv2.imshow("(2) hue_edges",hue_edges)
    cv2.imshow("(2) otsu_hue_mask",otsu_hue_mask)
    cv2.imshow("(2) edges_hue_otsu",edges_hue_otsu)
    cv2.waitKey(1)


def image_forKeras(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)# Image everywher is in rgb but Opencv does it in BGR convert Back
    image = cv2.resize(image,(30,30)) #Resize to model size requirement
    image = np.expand_dims(image, axis=0) # Dimension of model is [Batch_size, input_row,inp_col , inp_chan]
    return image

def get_plants(image,prob_masks,prob_regs,prob_cnts):

    plant_cnts = []
    plant_masks = prob_masks.copy()
    for i,rect in enumerate(prob_regs):
        # 4c. Detection (Localization) Extracting Roi from localized circle
        assumed_plant = image[rect[0]:rect[0]+rect[2],rect[1]:rect[1]+rect[3],:]
        is_plant = np.argmax(model(image_forKeras(assumed_plant)))
        
        if is_plant == 0:
            plant_masks = cv2.rectangle(plant_masks, (rect[1],rect[0]), (rect[1]+rect[3],rect[0]+rect[2]), 0,-1)
        else:
            plant_cnts.append(prob_cnts[i])

    return plant_masks,plant_cnts

def find_pattern(image,prob_regs,prob_cnts):
    mask = image.copy()


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

    mid_pt = (0,0)
    if (dist(crnt,frst) <= dist(crnt,lst)):
        # last to current is the longer side taking that as second crner and computing mid
        mid_pt =  (int((crnt[0]+lst[0])/2),int((crnt[1]+lst[1])/2))
    else:
        # next to current is the longer side taking that as second crner and computing mid
        mid_pt =  (int((crnt[0]+frst[0])/2),int((crnt[1]+frst[1])/2))

    
    cv2.circle(img_centroids, mid_pt, 16, 255,1)


    prevmin_dist = 100000
    nav_strt = (0,0)
    for plant_center in centroids:
        min_dist = dist(plant_center,nav_start_crnr)
        if(min_dist < prevmin_dist):
            prevmin_dist = min_dist
            nav_strt = plant_center

    prevmin_dist = 100000
    nav_scnd_strt = (0,0)
    for plant_center in centroids:
        if plant_center!= nav_strt:
            min_dist = dist(plant_center,nav_start_crnr) + dist(plant_center,mid_pt)
            if (min_dist < prevmin_dist):
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

def segment_crop_(frame):

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    hue = hls[:,:,0]

    hue_edges = cv2.Canny(hue, l_thrsh, h_thrsh,None,aperture)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    hue_edges_closed = cv2.morphologyEx(hue_edges, cv2.MORPH_CLOSE, kernel)
    
    hue_edges_closed,hue_edges_centroids,prob_regs,prob_cnts,prob_centroids = imfill(hue_edges_closed)
    

    center_col = int(hue.shape[1]/2)
    center_row = int(hue.shape[0]/2)
    # Rotate our image around an arbitrary point rather than the center
    field_mask_rot = hue_edges_closed.copy()
    M = cv2.getRotationMatrix2D((center_col, center_row), rot_angle, 1.0)
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
    cv2.rectangle(field_mask_rot_draw, (extLeft[0],extTop[1]), (extRight[0],extBot[1]), 255,3)
    
    prob_centroids,hue_edges_centroids = ret_centroid(field_mask_rot,prob_cnts)
    
    cv2.rectangle(hue_edges_centroids, (extLeft[0],extTop[1]), (extRight[0],extBot[1]), 255,3)
    cv2.drawContours(hue_edges_centroids,[box],0,255,3)

    nav_strt_pt,nav_strt_pt2 = find_nav_strt(hue_edges_centroids,box,hue.shape[0],hue.shape[1],prob_centroids)
    hue_edges_centroids = cv2.circle(hue_edges_centroids, nav_strt_pt, 16, 255,1)
    hue_edges_centroids = cv2.circle(hue_edges_centroids, nav_strt_pt2, 16, 255,2)


    #cv2.imshow("(2_a) hue",hue)
    #cv2.imshow("(2_b) hue_edges",hue_edges)
    #cv2.imshow("(2_c) hue_edges_closed",hue_edges_closed)
    #cv2.imshow("(2_d) hue_edges_closed_bgr",hue_edges_closed_bgr)
    cv2.imshow("(2_d) field_mask_rot",field_mask_rot)
    cv2.imshow("(2_d) field_mask_rot_draw",field_mask_rot_draw)
    cv2.imshow("(2_e) hue_edges_centroids",hue_edges_centroids)
 
    cv2.waitKey(1)

def main():

    global model_loaded
    if not model_loaded:
        print(tf.__version__)#2.4.1
        print("************ LOADING MODEL **************")

        # 1. Load CNN model
        global model
        model = load_model(os.path.join(os.getcwd(),"intelligent_drone/src/data/saved_model_Ros2_5_Sign.h5"),compile=False)

        # summarize model.
        model.summary()
        model_loaded = True

    frame = cv2.imread("drone_view.png")
    #cv2.imshow("UAV_video_feed",frame)

    while(1):
        segment_crop_(frame)


if __name__ == '__main__':
  main()
