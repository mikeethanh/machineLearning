#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('./opencv/Photos/park.jpg')
cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian

# Kết quả của dòng mã này là bạn có một hình ảnh mới (lap) 
# chứa biểu đồ của đạo hàm bậc hai (Laplace) của hình ảnh xám
#  gốc. Toán tử Laplace thường được sử dụng để tìm các điểm 
# đột ngột trong hình ảnh, đại diện cho sự thay đổi đột ngột 
# trong cường độ mức sáng. Nó có thể được sử dụng để làm nổi 
# bật các biên, cạnh hoặc các đặc điểm quan trọng trong hình
#  ảnh.
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel 

# Toán tử Sobel được sử dụng để phát hiện biên, cạnh hoặc thay
#  đổi đột ngột trong hình ảnh. Khi bạn tính toán đạo hàm theo
#  hướng ngang (sử dụng sobelx) và đạo hàm theo hướng dọc (sử
#  dụng sobely), bạn có thể kết hợp chúng để tạo ra thông tin
#  về biên toàn bộ hình ảnh. Thông thường, sau khi tính toán 
# đạo hàm theo hướng ngang và đạo hàm theo hướng dọc, bạn có
#  thể tính độ lớn của gradient tại mỗi điểm ảnh để hiển thị 
# biên hoặc thông tin về cường độ biến đổi trong hình ảnh.
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)


canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)
cv.waitKey(0)