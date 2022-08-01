
import cv2
import numpy as np
import math
from math import sin,cos,pi,degrees,radians




def ordrpts(pts,order="Counter-clockwise",image_draw=[]):

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = np.sum(pts,axis = 2)
    toplft = pts[np.argmin(s)][0]
    btmrgt = pts[np.argmax(s)][0]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 2)
    toprgt = pts[np.argmin(diff)][0]
    btmlft = pts[np.argmax(diff)][0]

    if image_draw!=[]:
        cv2.putText(image_draw,str(toplft), (toplft[0]-40,toplft[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
        cv2.putText(image_draw,str(btmlft), (btmlft[0]-40,btmlft[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)
        cv2.putText(image_draw,str(btmrgt), (btmrgt[0]+40,btmrgt[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
        cv2.putText(image_draw,str(toprgt), (toprgt[0]+40,toprgt[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)

    if order == "Counter-clockwise":
         return([toplft,btmlft,btmrgt,toprgt])
    else:
         return([toplft,toprgt,btmrgt,btmlft])
      

def estimate_corners(pts,image_draw=[],method="sum-diff",shape="rect"):

    if shape == "rect":

        if method =="sum-diff":

            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            s = np.sum(pts,axis = 2)
            crnr_a = pts[np.argmin(s)][0]
            crnr_b = pts[np.argmax(s)][0]
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
            diff = np.diff(pts, axis = 2)
            crnr_c = pts[np.argmin(diff)][0]
            crnr_d = pts[np.argmax(diff)][0]

        elif method == "approx-poly":
            peri = cv2.arcLength(cnts[0], True)
            corners = cv2.approxPolyDP(cnts[0], 0.04 * peri, True)

            crnr_a = (corners[0][0][0],corners[0][0][1])
            crnr_b = (corners[1][0][0],corners[1][0][1])
            crnr_c = (corners[2][0][0],corners[2][0][1])
            crnr_d = (corners[3][0][0],corners[3][0][1])


        if image_draw!=[]:

            cv2.circle(image_draw, (crnr_a[0][0],crnr_a[0][1]), 8, (0, 0, 255), -1)
            cv2.circle(image_draw, (crnr_b[0][0],crnr_b[0][1]), 8, (0, 255, 0), -1)
            cv2.circle(image_draw, (crnr_c[0][0],crnr_c[0][1]), 8, (255, 0, 0), -1)
            cv2.circle(image_draw, (crnr_d[0][0],crnr_d[0][1]), 8, (255, 255, 0), -1)

            cv2.putText(image_draw,str(crnr_a), (crnr_a[0]-40,crnr_a[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            cv2.putText(image_draw,str(crnr_d), (crnr_d[0]-40,crnr_d[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)
            cv2.putText(image_draw,str(crnr_b), (crnr_b[0]+40,crnr_b[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv2.putText(image_draw,str(crnr_c), (crnr_c[0]+40,crnr_c[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)


        return([crnr_a,crnr_d,crnr_b,crnr_c])

      



def nothing():
    pass

def imshow_stage(image_list,function = "",debug=False):
    if function!="":
        function = "[ "+function+" ] "
    img_name = function
    cv2.namedWindow(img_name,cv2.WINDOW_NORMAL)
    trackbar_name = "Stage"
    cv2.createTrackbar(trackbar_name, img_name,0,len(image_list)-1,nothing)

    if debug == True:
        while(1):
            Current_Stage = cv2.getTrackbarPos(trackbar_name,img_name)
            imshow(img_name, image_list[Current_Stage])
            k = cv2.waitKey(1)
            if k==27:
                break
    else:
        Current_Stage = cv2.getTrackbarPos(trackbar_name,img_name)
        imshow(img_name, image_list[Current_Stage])       


def imshow_stages(image_list,function = ""):
    
    if function!="":
        function = "[ "+function+" ] "    

    for idx,image in enumerate(image_list):
        img_name = function + "Stage_" + str(idx)
        imshow(img_name, image)

def imshow(img_name,img):
    # Function to display complete image on the screen
    img_disp = img.copy()
    cv2.namedWindow(img_name,cv2.WINDOW_NORMAL)
    # If size is greater then hd resolution (most monitors) resize image
    if img_disp.shape[1]>=720:
        img_disp = cv2.resize(img_disp, None,fx=0.5,fy=0.5)
    cv2.imshow(img_name,img_disp)    


# [NEW]: Find closest point in a list of point to a specific position
def closest_node(node, nodes,max_sanedist=None):
    nodes = np.asarray(nodes)
    if type(node)!='numpy.ndarray':
        node = np.asarray(node)
    dist_2 = np.sum((nodes - node)**2, axis=(nodes.ndim-1))
    if max_sanedist!= None:
        # if max allowed dist is provided then if the closest node is farther from max_sanedist we return -1
        if min(dist_2)>max_sanedist:
            return -1
    return np.argmin(dist_2)

def rotate_point(cx, cy, angle, p_tuple):
        
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
  

def moving_average(numbers,window_size=3):

    i = 0
    moving_averages = []
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i : i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    print(moving_averages)

def get_centroid(cnt):
    M = cv2.moments(cnt)
    if (M['m00']==0):
        pt_a = cnt[0][0]
        return (pt_a[0],pt_a[1])
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx,cy)

def ret_centroid(img,cnts):

    img_centroids = np.zeros_like(img)
    prob_centroids = []
    for idx,cnt in enumerate(cnts):
        cnt_center = get_centroid(cnt)
        img_centroids = cv2.circle(img_centroids, cnt_center, 5, 255,-1)
        prob_centroids.append(cnt_center)
    return prob_centroids,img_centroids


def find_line_parameters(a,b):
    m = 999
    y_inter = 0
    
    # For horizontal line (m = 0 ) then equation is y = b (y_intercept)

    # If not a vertical line , has a defined slope ,Else equation would be x = k 
    if (a[0]-b[0])!=0:
        m = (a[1]-b[1])/(a[0]-b[0])

    # b = y_0 - m*x_0
    y_inter = a[1]-(m*a[0])

    return m,y_inter
    
def dist(a,b):
    return math.sqrt( ( (a[1]-b[1])**2 ) + ( (a[0]-b[0])**2 ) )

def dist_pt_line(pt,m,b):
    # Todo: Handle Exceptional Cases [Horizontal + Vertical Lines]
    # Start by finding the point on line where perpendicular drawn from point intersects the original line
    (x1,y1) = pt
    x = ( (-b*m + x1 + m*y1) / (1 + (m*m)) )
    y = m*x + b

    return dist(pt,(x,y))

def estimate_pt(pta,ptb,case ="start"):
    x1,y1 = pta
    x2,y2 = ptb
    if case == "start":
        return (x1*2-x2,y1*2-y2)
    else:
        return (x2*2-x1,y2*2-y1)


def imfill(im_th,display_intermediates = False,fill_loc = (0,0)):
    #th, im_th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV);
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, fill_loc, 255)

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
    prob_regs = []
    prob_cnts = []
    prob_centroids = []
    for idx,cnt in enumerate(cnts):
        if cv2.contourArea(cnt)<200:
            im_out = cv2.drawContours(im_out, cnts, idx, 0,-1)
        else:
            cnt_center = get_centroid(cnt)
            im_out_centroids = cv2.circle(im_out_centroids, cnt_center, 5, 255,-1)
            x,y,w,h = cv2.boundingRect(cnt)
            prob_regs.append([x,y,w,h])
            prob_cnts.append(cnt)
            prob_centroids.append(cnt_center)
    
    return im_out,im_out_centroids,prob_regs,prob_cnts,prob_centroids

def ApproxDistBWCntrs(cnt,cnt_cmp):
    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # compute the center of the contour
    M_cmp = cv2.moments(cnt_cmp)
    cX_cmp = int(M_cmp["m10"] / M_cmp["m00"])
    cY_cmp = int(M_cmp["m01"] / M_cmp["m00"])
    minDist=dist((cX,cY),(cX_cmp,cY_cmp))
    Centroid_a=(cX,cY)
    Centroid_b=(cX_cmp,cY_cmp)
    return minDist,Centroid_a,Centroid_b

def RetLargestContour(gray):
    LargestContour_Found = False
    thresh=np.zeros(gray.shape,dtype=gray.dtype)
    _,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    #Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    Max_Cntr_area = 0
    Max_Cntr_idx= -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > Max_Cntr_area:
            Max_Cntr_area = area
            Max_Cntr_idx = index
            LargestContour_Found = True
    if (Max_Cntr_idx!=-1):
        thresh = cv2.drawContours(thresh, cnts, Max_Cntr_idx, (255,255,255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
    return thresh, LargestContour_Found

def remove_outliers(BW,MaxDistance,display_connections = False):

    #cv2.namedWindow("BW_zero",cv2.WINDOW_NORMAL)
    BW_zero= cv2.cvtColor(BW,cv2.COLOR_GRAY2BGR)

    # 1. Find the two Contours for which you want to find the min distance between them.
    cnts= cv2.findContours(BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]#3ms
    
    # 2. Keep Only those contours that are not lines 
    MinArea=1
    cnts_Legit=[]
    for index, _ in enumerate(cnts):
        area = cv2.contourArea(cnts[index])
        if area > MinArea:
            cnts_Legit.append(cnts[index])
    cnts=cnts_Legit

    # Cycle through each point in the Two contours & find the distance between them.
    # Take the minimum Distance by comparing all other distances & Mark that Points.
    CntIdx_BstMatch = []# [BstMatchwithCnt0,BstMatchwithCnt1,....]
    
    # 3. Connect each contous with its closest 
    for index, cnt in enumerate(cnts):
        prevmin_dist = MaxDistance ; Bstindex_cmp = 0 ; BstCentroid_a=0  ; BstCentroid_b=0      
        for index_cmp in range(len(cnts)-index):
            index_cmp = index_cmp + index
            cnt_cmp = cnts[index_cmp]
            if (index!=index_cmp):
                min_dist,Centroid_a,Centroid_b  = ApproxDistBWCntrs(cnt,cnt_cmp)

                #Closests_Pixels=(cnt[min_dstPix_Idx[0]],cnt_cmp[min_dstPix_Idx[1]])
                if(min_dist < prevmin_dist):
                    if (len(CntIdx_BstMatch)==0):
                        prevmin_dist = min_dist
                        Bstindex_cmp = index_cmp
                        #BstClosests_Pixels = Closests_Pixels
                        BstCentroid_a=Centroid_a
                        BstCentroid_b=Centroid_b   

                    else:
                        Present= False
                        for i in range(len(CntIdx_BstMatch)):
                            if ( (index_cmp == i) and (index == CntIdx_BstMatch[i]) ):
                                Present= True
                        if not Present:
                            prevmin_dist = min_dist
                            Bstindex_cmp = index_cmp
                            #BstClosests_Pixels = Closests_Pixels
                            BstCentroid_a=Centroid_a
                            BstCentroid_b=Centroid_b   
        #if ((prevmin_dist!=100000 ) and (prevmin_dist>MaxDistance)):
        #    print("party")
        #    break
        if (type(BstCentroid_a)!=int):
            CntIdx_BstMatch.append(Bstindex_cmp)
            cv2.line(BW_zero,BstCentroid_a,BstCentroid_b,(0,255,0),thickness=2)

    if display_connections:
        cv2.namedWindow("BW_zero(Connected)",cv2.WINDOW_NORMAL)
        cv2.imshow("BW_zero(Connected)", BW_zero)
        cv2.waitKey(0)

    BW_zero = cv2.cvtColor(BW_zero,cv2.COLOR_BGR2GRAY)

    # 4. Get estimated midlane by returning the largest contour 
    BW_Largest,Largest_found = RetLargestContour(BW_zero)#3msec

    BW_Largest = cv2.bitwise_and(BW, BW_Largest)

    # 5. Return Estimated Midlane if found otherwise send original
    if(Largest_found):
        return BW_Largest
    else:
        return BW