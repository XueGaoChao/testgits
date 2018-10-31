#-*-coding:utf-8-*-
import cv2
import numpy as np
from matplotlib import  pyplot as plt

img = cv2.imread('img.png')
kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])











