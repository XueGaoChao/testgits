#-*-coding:utf-8-*-
import cv2
# （二）评分要求
# 1.通过OpenCV函数读入附加文件视频shendu001.avi。 （8分）
cap=cv2.VideoCapture("E:\Python深度学习\人工智能第5个月\深度学习\周考01/bwcs.mp4")
# 2.将读入的shendu001.avi保存成灰度视频文件。（8分）
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc,20.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    # print(frame.shape)
    if ret == True:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # print(frame.shape)
        out.write(frame)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break

cap.release()
out.release()
# 3.从视频文件shendu001.avi中随机截取20张图片并保存起来，文件分别命名为p1.jpg，p2.jpg,  … ,  p20.jpg。（8分）
import numpy as np
# cap=cv2.VideoCapture("E:\Python深度学习\人工智能第5个月\深度学习\周考01/bwcs.mp4")
# num = 1
# while cap.isOpened() and num < 21:
#     pic = np.random.randint(1,1000)
#     ret,frame = cap.read()
#     if pic < 20:
#         name = 'p{}.jpg'.format(str(num))
#         print(name)
#         cv2.imwrite(name,frame)
#         num += 1
# cap.release()
# 4.读取图片p2.jpg，将其上下翻转，保存为p21_rotate_90.jpg（8分）
img_p2 = cv2.imread('p2.jpg')
img_p2=cv2.flip(src=img_p2,flipCode=0)
cv2.imwrite('p21_rotate_90.jpg',img_p2)
# 5.读取图片p2.jpg，将其并左右翻转，保存为p22_rotate_90.jpg（8分）
p2_img = cv2.imread('p2.jpg')
p2_img = cv2.flip(src=p2_img,flipCode=1)
cv2.imwrite('p22_rotate_90.jpg',p2_img)
# 6.读取图片p1.jpg并将其转换为灰度图像后保存为ph1_gray.jpg（8分）
p1_img = cv2.imread('p1.jpg')
p1_img = cv2.cvtColor(src=p1_img,code=cv2.COLOR_BGR2GRAY)
cv2.imwrite('ph1_gray.jpg',p1_img)
# 7.对p5.jpg进行边缘检测，并将结果保存为p5_edges.jpg（8分）
p5_img = cv2.imread('p5.jpg')
p5_img = cv2.Canny(image=p5_img,threshold1=768,threshold2=1366)
cv2.imwrite('p5_edges.jpg',p5_img)
# 8.对p3.jpg进行均值滤波，并将结果分别保存为p31.jpg（7分）
# p3_img = cv2.imread('p3.jpg')
# p3_img = cv2.blur(p3_img,(7,7))
# cv2.imwrite('p3_blur.jpg',p3_img)

# 9.对p4.jpg进行高斯滤波，并将结果分别保存为p41.jpg（7分）
p4_img = cv2.imread('p4.jpg')
p4_img = cv2.GaussianBlur(p4_img,(7,7),0)
cv2.imwrite('p41.jpg',p4_img)
# 10.对p6.jpg利用sift算法提取其特征，并将图片保存为p6_sift.jpg（4分）
# p6_img = cv2.imread('p6.jpg')
# gray = cv2.cvtColor(p6_img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d_SIFT.create()
# kp = sift.detect(gray,None)
# cv2.drawKeypoints(image=p6_img,keypoints=kp,outImage=p6_img)
# cv2.imwrite('p6_sift.jpg')
# 11.读取p7.jpg，通过Harris算法加工，将图片保存为p7_Harris.jpg（4分）
# p7_img = cv2.imread('p7.jpg')
# gray=cv2.cvtColor(p7_img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# harris = cv2.cornerHarris(src=gray,blockSize=5,ksize=5,k=0.06)
# p7_img[harris>0.015*harris.max()]=[0,0,255]#选取角点
# cv2.imwrite('p7_Harris.jpg',p7_img)
# 12.OpenCV 中的 Gui 特性是什么?（4分）
# 13.为什么使用 Python-OpenCV ?（4分）
# 14.写出OpenCV图像处理编程常调用的库语句（4分）