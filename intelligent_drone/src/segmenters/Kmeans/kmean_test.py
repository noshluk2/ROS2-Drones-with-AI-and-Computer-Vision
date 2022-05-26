import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_cluster(cluster_no=2):
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = cluster_no
    masked_image[labels == cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    # show the image
    #plt.imshow(masked_image)
    #plt.show()
    
    masking_cluster_text = "Masked {} custer".format(cluster_no)

    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(masking_cluster_text, masked_image)

def show_cluster(cluster_no=2):
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.zeros_like(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = cluster_no
    masked_image[labels == cluster] = colors[cluster]
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    # show the image
    #plt.imshow(masked_image)
    #plt.show()
    
    masking_cluster_text = "Masked {} custer".format(cluster_no)

    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(masking_cluster_text, masked_image)

# read the image
image = cv2.imread("drone_view.png")

r = cv2.selectROI("SelectROI",image)

# Crop image
image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#image = cv2.GaussianBlur(image,(5,5),0)
#cv2.imshow("image_blurred",image)

hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
hue = hls[:,:,0]
cv2.imshow("hue",hue)
hue = cv2.equalizeHist(hue)
cv2.imshow("hue(hist_equalized)",hue)

#hue = cv2.GaussianBlur(hue,(3,3),2)
hue = cv2.medianBlur(hue, 3)
cv2.imshow("hue_blurred",hue)

hue_edges = cv2.Canny(hue, 50, 150,None,3)
cv2.imshow("hue(edges)",hue_edges)

image = cv2.cvtColor(hue, cv2.COLOR_GRAY2BGR)

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

print(pixel_values.shape)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 2

_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

#cv2.waitKey(0)
# flatten the labels array
labels = labels.flatten()


colors = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,255]],dtype=np.uint8)

# convert all pixels to the color of the centroids
#segmented_image = centers[labels.flatten()]
segmented_image = colors[labels.flatten()]

#print("segmented_image = ",segmented_image[0])
#cv2.waitKey(0)

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image
#plt.imshow(segmented_image)
#plt.show()

segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
cv2.imshow("segmented_Image", segmented_image)

for i in range(k):
    show_cluster(i)

cv2.waitKey(0)

