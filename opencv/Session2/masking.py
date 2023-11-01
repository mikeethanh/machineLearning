
import cv2 as cv
import numpy as np

img = cv.imread('./Photos/cats 2.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)

# img.shape[1] và img.shape[0] là chiều rộng và chiều cao của 
# hình ảnh img. Tùy thuộc vào kích thước của hình ảnh, tâm của 
# hình tròn được đặt ở giữa hình ảnh, và sau đó, 45 pixel được 
# thêm vào phía ngang (trục X) để tạo sự dịch chuyển của tâm theo 
# chiều ngang.
# 100: Đây là bán kính của hình tròn.
circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45,img.shape[0]//2), 100, 255, -1)
cv.imshow('Circle', circle)

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
cv.imshow('Rectangle', rectangle)

weird_shape = cv.bitwise_and(circle,rectangle)
cv.imshow('Weird Shape', weird_shape)

masked = cv.bitwise_and(img,img,mask=weird_shape)
cv.imshow('Weird Shaped Masked Image', masked)

cv.waitKey(0)