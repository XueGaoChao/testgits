#-*-coding:utf-8-*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = np.zeros((512,512,3),np.uint8)
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.circle(img,(445,60),65,(0,0,255))
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
cv2.line(img,(0,0),(511,511),(0,0,255),2)
pts = np.array([[10,5],[20,30],[70,23],[50,20]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500),font,4,(255,255,255),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


