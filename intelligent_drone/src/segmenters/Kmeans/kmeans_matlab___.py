# Based on https://www.mathworks.com/help/images/ref/imsegkmeans.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import preprocessing
import cProfile

import time
import concurrent.futures
from utilities import ret_largest_cnt,imfill

from multiprocessing import Array,Value

#start = time.time()
#end = time.time()
#print("end - start = {}.sec".format(end - start))

# Building some gabor kernels to filter image
orientations = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
wavelengths = [3, 6, 12, 24, 48, 96]

def build_gabor_kernels():
    filters = []
    ksize = 40
    for rotation in orientations:
        for wavelength in wavelengths:
            kernel = cv2.getGaborKernel((ksize, ksize), 4.25, rotation, wavelength, 0.5, 0, ktype=cv2.CV_32F)
            filters.append(kernel)

    return filters

def extract_gabor_features(image):
    
    rows, cols = image.shape[0:2]
    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    gaborKernels = build_gabor_kernels()

    gaborFilters = []

    for (i, kernel) in enumerate(gaborKernels):
        filteredImage = cv2.filter2D(gray, cv2.CV_8UC1, kernel)

        # Blurring the image
        sigma = int(3*0.5*wavelengths[i % len(wavelengths)])

        # Sigma needs to be odd
        if sigma % 2 == 0:
            sigma = sigma + 1

        blurredImage = cv2.GaussianBlur(filteredImage,(int(sigma),int(sigma)),0)
        gaborFilters.append(blurredImage)

    # numberOfFeatures = 1 (gray color) + number of gabor filters + 2 (x and y)
    numberOfFeatures = 1  + len(gaborKernels) + 2

    # Empty array that will contain all feature vectors
    # Some example results:
    # featureVectors[0] = [164, 3, 10, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 3, 10, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 1, 1]
    # featureVectors[1] = [163, 3, 17, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 3, 17, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 1, 2]
    featureVectors = []
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            vector = [gray[i][j]]

            for k in range(0, len(gaborKernels)):
                vector.append(gaborFilters[k][i][j])

            vector.extend([i+1, j+1])

            featureVectors.append(vector)

    # Normalizing the feature vectors
    scaler = preprocessing.StandardScaler()
    scaler.fit(featureVectors)
    featureVectors = scaler.transform(featureVectors)

    return featureVectors,numberOfFeatures

def kmeans(image):
    # Step 2: Extract gabor features 
    featureVectors,numberOfFeatures = extract_gabor_features(image)
    ##############################################################################
    # Compute clustering with KMeans    
    kmeans = KMeans(n_clusters=2, random_state=170)
    kmeans.fit(featureVectors)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return labels

def batch_kmeans(image):

    # Step 2: Extract gabor features 
    featureVectors,numberOfFeatures = extract_gabor_features(image)
    ##############################################################################
    # Compute clustering with MiniBatchKMeans
    # initialization method ===> kmean++ is selected for smart initialization of cluster centers
    # n_clusters ==> 2 because i want to use it as a segmenting tool
    # max_no_improvement ==> 15 (increased from 10) to get more reliable inertial stopping
    # reassignment_ratio ==> 0.2 (increased from 0.01) low count centers are more easily reasigned [Done to tackle getting stuck in local optima]
    # Current accuracy seems to be around 70 - 80 % which is not the best but gives us relatively good results
    mbk = MiniBatchKMeans(n_clusters=2,init="k-means++", n_init=10, max_no_improvement=15,reassignment_ratio=0.02)
    mbk.fit(featureVectors)
    mbk_labels = mbk.labels_

    return mbk_labels

def segment_kmeans(image,batch_process = False):

    # Step 1: PreProcessing 
    # Keeping only the hue information seems to be critical for kmean to cluster what we truly intend to...
    #              (Plants from everything else) ==> Otherwise its just a big mess that gets returned
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    hue = hls[:,:,0]

    # Resizing the image. => Full image is taking to much time to process
    rows, cols = hue.shape[0:2]
    scale = 0.5
    hue = cv2.resize(hue, (int(cols * scale), int(rows * scale)))

    rows, cols = hue.shape[0:2]
    
    # Step 3: Use batch_kmeans to segment image into clusters

    if batch_process:
        labels = batch_kmeans(hue)
    else:
        labels = kmeans(hue)
    
    ##############################################################################
    # Step 4: Post-Processing
    outt = (labels.reshape(rows, cols)) 
    outt = cv2.normalize(outt, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask = outt.astype(np.uint8)

    corners_sum = outt[0,0] + outt[0,cols-1] + outt[rows-1,0] + outt[rows-1,cols-1]
    if (corners_sum>510):
        mask = cv2.bitwise_not(mask)

    mask_largest, cnt_largest = ret_largest_cnt(mask)

    return mask_largest,cnt_largest,scale




def segment_canny(image):
    # Step 1: PreProcessing 
    # Keeping only the hue information seems to be critical for kmean to cluster what we truly intend to...
    #              (Plants from everything else) ==> Otherwise its just a big mess that gets returned
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    hue = hls[:,:,0]

    # Identifying objects with stronger edges
    hue_edges = cv2.Canny(hue, 50, 150,None,3)

    # Removing noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    hue_edges_closed = cv2.morphologyEx(hue_edges, cv2.MORPH_CLOSE, kernel)
    
    # Keeping only largercontours and retreiving their centroids
    hue_edges_filled,hue_edges_centroids,prob_regs,prob_cnts,prob_centroids = imfill(hue_edges_closed)


    mask_largest, cnt_largest = ret_largest_cnt(hue_edges_filled,prob_cnts)

    return mask_largest,cnt_largest






def segment(image,method="kmeans"):
    
    if method == "kmeans":
    
        segment_kmeans(image)

    elif method == "Canny":
        
        segment_canny()

def detect(image):
    """
    (detect visually and structurally identifiable objects.)
  
    Function takes in a bgr image and utilizez multiple methods (Canny || kmeans) 
                            To segment out foreground and return its bounding box
  
    Parameters:
    image (numpy 3d array): Colored image
  
    Returns:
    List: Rect bounding the segmented object
  
    """

    b_rect = [0,0,0,0] # Rect bounding the segmented object (Output)

    used_kmeans = False
    seg_rect = [0,0,0,0] # Rect bounding the segmented object (Needs adjustments: if preprocessing done
                         #                                      on image for segmentation)

    
    plant_mask,plant_cnt = segment_canny(image)

    # if plant was not segmented using Canny (faster) method . Resort to kmeans for segmentaion.
    if len(plant_cnt)==0:
        plant_mask,plant_cnt,scale = segment_kmeans(image)
        used_kmeans = True

    if len(plant_cnt)!=0:
        seg_rect = cv2.boundingRect(plant_cnt) 
        if used_kmeans:
            # Need to adjust for the precprocesing (resizing) done indicated by scale
            xA, yA, w, h = seg_rect
            xB, yB = xA + w, yA + h
            rect_pts = np.array([[[xA, yA]], [[xB, yA]], [[xA, yB]], [[xB, yB]]], dtype=np.float32)
            affine_warp = np.array([[1/scale,       0,  0],
                                    [      0, 1/scale,  0],
                                    [      0,       0,  1]], dtype=np.float32)
            pts_trans = cv2.perspectiveTransform(rect_pts, affine_warp)# (4,2) => 4 rows 2 columns
            x_t = int(pts_trans[0][0][0]) # 1 row 1 col  => x_start (transformed)
            y_t = int(pts_trans[0][0][1]) # 1 row 2 col  => y_start (transformed)
            w_t = int(pts_trans[3][0][0] - pts_trans[0][0][0]) # x_end - x_start
            h_t = int(pts_trans[3][0][1] - pts_trans[0][0][1]) # x_end - x_start

            b_rect = [x_t,y_t,w_t,h_t]
        else:
            b_rect = seg_rect

    #plant_mask = cv2.rectangle(plant_mask,(seg_rect[0],seg_rect[1]), (seg_rect[0]+seg_rect[2],seg_rect[1]+seg_rect[3]), 255,2)

    #cv2.namedWindow("plant_mask",cv2.WINDOW_NORMAL)
    #cv2.imshow("plant_mask",plant_mask)
    
    return b_rect
        
    

def main():
    image = cv2.imread('/home/haiderabbasi/Development/r1_workspace/src/ROS2-Drones-with-AI-and-Computer-Vision/drone_view.png')
    #r = cv2.selectROI("SelectROI",image)
    r =  [763, 160, 253, 238]
    p_rect = Array('i', 4)
    #updated = Value('i',False)
    updated = False
    #print("updated b4 = ", updated.value )
    print("updated b4 = ", updated )
    # Crop image
    image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    #mask = segment(image)
    
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(detect,image)]
    #     for f in concurrent.futures.as_completed(results):
    #         print("results = ",f.result())
    #         p_rect = f.result()
    #         updated = True
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.submit(detect,image)
    #for f in concurrent.futures.as_completed(results):
    #    print("results = ",f.result())
    #    p_rect = f.result()
    #    updated = 1            
    #print("results = ",results.result())
    #print("Updated = ",updated.value)     
    #print("results = ",results.result())
    print("results.isrunning? = ",results.running())
    #p_rect = detect(image)
    while(1):
        if not results.running():
            # Done executing , Extract result
            updated = True
            p_rect = results.result()
        if updated:
            mask = cv2.rectangle(image,(p_rect[0],p_rect[1]), (p_rect[0]+p_rect[2],p_rect[1]+p_rect[3]), (0,255,0),2)

            cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
            cv2.imshow("mask",mask)
            cv2.waitKey(0)
            break
        else:
            
            print("Not updated yet")

if __name__ == "__main__":
    #cProfile.run('main()',sort='cumtime')
    main()
