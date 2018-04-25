print("Assignment 6: Part 3 - Image Quantization")

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

K = 4

img = cv2.imread('input-images//image1.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
output_image = res.reshape((img.shape))
cv2.imwrite('clusteredImages//output1_'+str(K)+'.png', output_image);

print("Image1 has been clustered")


img = cv2.imread('input-images//image2.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
output_image = res.reshape((img.shape))
cv2.imwrite('clusteredImages//output2_'+str(K)+'.png', output_image);
print("Image2 has been clustered")





img = cv2.imread('input-images//image3.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
output_image = res.reshape((img.shape))
cv2.imwrite('clusteredImages//output3_'+str(K)+'.png', output_image);
print("Image3 has been clustered")



img = cv2.imread('input-images//image4.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
output_image = res.reshape((img.shape))
cv2.imwrite('clusteredImages//output4_'+str(K)+'.png', output_image);
print("Image4 has been clustered")




img = cv2.imread('input-images//image5.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
output_image = res.reshape((img.shape))
cv2.imwrite('clusteredImages//output5_'+str(K)+'.png', output_image);
print("Image5 has been clustered")