#-*-coding:utf-8-*-
import numpy as np
import cv2
img = cv2.imread('img.png',cv2.WINDOW_NORMAL)
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




