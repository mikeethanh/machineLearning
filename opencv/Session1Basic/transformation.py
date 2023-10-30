#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('./Photos/park.jpg')
cv.imshow('Park', img)

# Translation
def translate(img, x, y):
    # np.float32: Đây là hàm của thư viện NumPy để tạo một mảng NumPy
    # có kiểu dữ liệu là float32 (32-bit floating-point).
    # [[1, 0, x], [0, 1, y]]: Đây là nội dung của ma trận chuyển đổi.
    #  Ma trận này là một ma trận 2x3, tức là có 2 hàng và 3 cột. Ma 
    # trận này sử dụng để biểu diễn phép dịch chuyển trên hình ảnh
    transMat = np.float32([[1,0,x],[0,1,y]])
    # dimensions = (img.shape[1], img.shape[0]): Dòng này lấy kích thước 
    # ban đầu của hình ảnh img và lưu trữ chúng trong một tuple dimensions.
    dimensions = (img.shape[1], img.shape[0])
    # eturn cv.warpAffine(img, transMat, dimensions): Cuối cùng, hàm sử dụng 
    # hàm cv.warpAffine() của OpenCV để áp dụng phép dịch chuyển theo ma trận 
    #  transMat lên hình ảnh img, với kích thước đã được xác định trong dimensions
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    # Nếu rotPoint chưa được cung cấp (None), dòng này thiết lập rotPoint 
    # thành tâm của hình ảnh. Điều này có nghĩa là hình ảnh sẽ xoay quanh 
    # tâm của nó nếu rotPoint không được chỉ định.
    if rotPoint is None:
        rotPoint = (width//2,height//2)
    
    #  Dòng này tạo ma trận biến đổi xoay (rotation transformation matrix) rotMat
    #  bằng cách sử dụng hàm cv.getRotationMatrix2D. Ma trận này chứa thông tin về
    #  phép xoay sẽ được thực hiện. Tham số rotPoint là tâm xoay, angle là góc xoay, 
    # và 1.0 là tỷ lệ giãn (scale), trong trường hợp này là không giãn.
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

rotated_rotated = rotate(img, -90)
cv.imshow('Rotated Rotated', rotated_rotated)

# Resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping
# Tham số này xác định cách lật hình ảnh. Khi bạn sử dụng -1, hình ảnh
#  sẽ được lật cả theo trục ngang và trục dọc. Nói cách khác, hình ảnh 
# sẽ bị đảo ngược cả chiều ngang và chiều dọc.
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)