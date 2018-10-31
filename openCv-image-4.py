#-*-coding:utf-8-*-
import cv2
import numpy as np

img = cv2.imread('football.jpg')
print(img.shape)

ball = img[240:280,260:300]
img[273:313,110:150] = ball

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

















