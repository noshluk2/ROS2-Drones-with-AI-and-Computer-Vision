# Based on https://www.mathworks.com/help/images/ref/imsegkmeans.html

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import preprocessing

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

image = cv.imread('drone_view.png')
r = cv.selectROI("SelectROI",image)
# Crop image
image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
hue = hls[:,:,0]
plt.imshow(hue)

image = cv.cvtColor(hue, cv.COLOR_GRAY2BGR)

#image = cv.imread('kuta.png')
rows, cols, channels = image.shape

# Resizing the image. 
# Full image is taking to much time to process
image = cv.resize(image, (int(cols * 0.5), int(rows * 0.5)))
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
featureVectors = []

for i in range(0, rows, 1):
    for j in range(0, cols, 1):
        vector = [gray[i][j]]

        for k in range(0, len(gaborKernels)):
            vector.append(gaborFilters[k][i][j])

        vector.extend([i+1, j+1])

        featureVectors.append(vector)

# Some example results:
# featureVectors[0] = [164, 3, 10, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 3, 10, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 1, 1]
# featureVectors[1] = [163, 3, 17, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 3, 17, 255, 249, 253, 249, 2, 43, 255, 249, 253, 249, 1, 2]

# Normalizing the feature vectors
scaler = preprocessing.StandardScaler()

scaler.fit(featureVectors)
featureVectors = scaler.transform(featureVectors)

kmeans = KMeans(n_clusters=2, random_state=170)
kmeans.fit(featureVectors)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

result = centers[labels]

# Only keep first 3 columns to make it easy to plot as an RGB image
result = np.delete(result, range(3, numberOfFeatures), 1)

#outt = (result.reshape(rows, cols, 3)) * 100
outt = (result.reshape(rows, cols, 3)) 

outt = cv.normalize(outt, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)

outt = outt.astype(np.uint8)

print("min(outt) ",outt.min())
print("max(outt) ",outt.max())
print("outt.dtype ",outt.dtype)
plt.figure(figsize = (15,8))
#plt.imsave('test.jpg',outt)
plt.imshow(outt)
plt.show()