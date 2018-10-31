#-*-coding:utf-8-*-
import cv2

img = cv2.imread('56.jpg')

print(img.shape)
print(img.size)
print(img.dtype)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()











