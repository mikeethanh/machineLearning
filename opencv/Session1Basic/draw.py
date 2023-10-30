#pylint:disable=no-member

import cv2 as cv
import numpy as np

# : Ở đây, bạn tạo một mảng NumPy có kích thước 500x500 
# pixel với 3 kênh màu (RGB) và kiểu dữ liệu uint8 (unsigned 8-bit integer). 
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# 1. Paint the image a certain colour
# red square in thís image
# Dòng này thiết lập màu của một vùng hình vuông 
# trên hình ảnh blank thành màu đỏ. Cụ thể, đoạn 
# mã này đặt giá trị của các pixel trong phạm vi 
# từ hàng 200 đến 299 và cột từ 300 đến 399 thành 
# (0, 0, 255), trong đó (0, 0, 255) biểu thị cho màu đỏ 
blank[200:300, 300:400] = 0,0,255
cv.imshow('Red', blank)

# 2. Draw a Rectangle
# hàm cv.rectangle để vẽ hình chữ nhật lên hình ảnh blank.
# (0, 0): Là tọa độ góc trái trên của hình chữ nhật (x, y).
# blank.shape trả về một tuple có ba phần tử, trong đó phần
#  tử đầu tiên (index 0) là chiều cao, phần tử thứ hai (index 1)
#  là chiều rộng, và phần tử thứ ba (index 2) là số kênh màu (3 trong trường hợp này).

cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
cv.imshow('Rectangle', blank)

# 3. Draw A circle
# 40: Đây là bán kính (radius) của hình tròn, trong trường hợp này là 40 pixel.
# blank.shape[1]//2, blank.shape[0]//2): Đây là tọa độ của tâm (center) của hình tròn. 
# Bằng cách sử dụng blank.shape[1]//2 và blank.shape[0]//2, bạn đã đặt tâm của hình tròn
#  ở giữa hình ảnh blank
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
cv.imshow('Circle', blank)

# 4. Draw a line
cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
cv.imshow('Line', blank)

# 5. Write text
cv.putText(blank, 'Hello, my name is Jason!!!', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
cv.imshow('Text', blank)

cv.waitKey(0)