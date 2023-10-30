#pylint:disable=no-member

import cv2 as cv

# Read in an image
img = cv.imread('./Photos/park.jpg')
cv.imshow('Park', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur 
# 7,7 xác định độ mờ 
# cv.BORDER_DEFAULT: Tham số này đại diện cho cách xử lý viền của hình 
# ảnh khi thực hiện phép biến đổi. 
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
# ngướng dưới , ngưỡng trên x
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Dilating the image
# iterations=3: Điều này chỉ định số lần lặp (iterations) để thực hiện phép Dilation. 
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7,7), iterations=3)
cv.imshow('Eroded', eroded)

# Resize
# INTER_CUBIC, một trong những phương pháp nội suy cao cấp, giúp làm mượt 
# hình ảnh sau khi thay đổi kích thước.
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)

