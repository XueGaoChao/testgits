#-*-coding:utf-8-*-
import cv2
import numpy as np

img = cv2.imread('football.jpg',0)

rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,0.6)

dst = cv2.warpAffine(img,M,(2*cols,2*rows))
while(1):
    cv2.imshow('img',dst)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()