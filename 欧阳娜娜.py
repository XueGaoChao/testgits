#-*-coding:utf-8-*-
import cv2
import numpy as np


img = cv2.imread('1.jpg')
# px=img[100,100]
# print(px)
# blue = img[100,100,0]
# print(blue)
#
# img[100,100] = [255,255,255]
# print(img[100,100])
#
# print(img.item(10,10,2))
# img.itemset((10,10,2),100)
# print(img.item(10,10,2))

print(img.shape)
print(img.size)
print(img.dtype)

ball = img[50:450,500:800,2]
img[10:410,1000:1300,2] = ball

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()











