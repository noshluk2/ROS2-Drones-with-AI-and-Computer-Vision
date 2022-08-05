# Based on https://www.mathworks.com/help/images/ref/imsegkmeans.html

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import preprocessing
import cProfile

import time
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
            kernel = cv.getGaborKernel((ksize, ksize), 4.25, rotation, wavelength, 0.5, 0, ktype=cv.CV_32F)
            filters.append(kernel)

    return filters


def kmeans(featureVectors,numberOfFeatures):

    kmeans = KMeans(n_clusters=2, random_state=170)
    kmeans.fit(featureVectors)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    result = centers[labels]
    # Only keep first 3 columns to make it easy to plot as an RGB image
    result = np.delete(result, range(3, numberOfFeatures), 1)

    return result


def batch_kmeans(featureVectors,numberOfFeatures):
    ##############################################################################
    # Compute clustering with MiniBatchKMeans

    mbk = MiniBatchKMeans(n_clusters=2)
    mbk.fit(featureVectors)
    mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_
    mbk_means_labels_unique = np.unique(mbk_means_labels)
    result_mbk = mbk_means_cluster_centers[mbk_means_labels]
    # Only keep first 3 columns to make it easy to plot as an RGB image
    result_mbk = np.delete(result_mbk, range(3, numberOfFeatures), 1)

    return result_mbk

def extract_gabor_features(image):
    rows, cols, channels = image.shape

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gaborKernels = build_gabor_kernels()

    gaborFilters = []

    for (i, kernel) in enumerate(gaborKernels):
        filteredImage = cv.filter2D(gray, cv.CV_8UC1, kernel)

        # Blurring the image
        sigma = int(3*0.5*wavelengths[i % len(wavelengths)])

        # Sigma needs to be odd
        if sigma % 2 == 0:
            sigma = sigma + 1

        blurredImage = cv.GaussianBlur(filteredImage,(int(sigma),int(sigma)),0)
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


def segment(image):
    # Step 1: Processing 
    # Keeping only the hue information seems to be critical for kmean to cluster what we truly intend to...
    #              (Plants from everything else) ==> Otherwise its just a big mess that gets returned
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    image = cv.cvtColor(hls[:,:,0], cv.COLOR_GRAY2BGR)
    # Resizing the image. => Full image is taking to much time to process
    rows, cols, channels = image.shape
    image = cv.resize(image, (int(cols * 0.5), int(rows * 0.5)))

    rows, cols, channels = image.shape
    
    # Step 2: Extract gabor features 
    featureVectors,numberOfFeatures = extract_gabor_features(image)

    # Step 3: Use batch_kmeans to segment image into clusters
    result = batch_kmeans(featureVectors,numberOfFeatures)
    #result = kmeans(featureVectors,numberOfFeatures)

    ##############################################################################
    # Step 4: Post-Processing
    outt = (result.reshape(rows, cols, 3)) 
    outt = cv.normalize(outt, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    mask = outt.astype(np.uint8)

    return mask

def main():
    image = cv.imread('/home/haiderabbasi/Development/r1_workspace/src/ROS2-Drones-with-AI-and-Computer-Vision/drone_view.png')
    #r = cv.selectROI("SelectROI",image)
    r =  [763, 160, 253, 238]

    # Crop image
    image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    mask = segment(image)

    print("min(mask) ",mask.min())
    print("max(mask) ",mask.max())
    print("mask.dtype ",mask.dtype)

    plt.figure(figsize = (15,8))
    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()',sort='cumtime')
    #start = time.time()
    main()
    #end = time.time()
    #print("end - start = {}.sec".format(end - start))