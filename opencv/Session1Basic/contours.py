#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('./Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# contours: Biến này lưu trữ danh sách các đường viền mà hàm 
# cv.findContours() tìm thấy. Mỗi đường viền được biểu diễn 
# bằng một danh sách các điểm, mỗi điểm là một cặp tọa độ (x, y)
#  trên hình ảnh. Biến contours chứa danh sách các danh sách,
#  mỗi danh sách biểu diễn một đường viền.
# Hierarchies là một mảng nhiều chiều (thường là một mảng 2D) 
# chứa thông tin về mối quan hệ giữa các đường viền. Hierarchies 
# có thể được sử dụng để biết được cấu trúc hình viền nằm trong nhau, 
# nằm bên trong hoặc cách các viền tương tác với nhau

# ham số này xác định cách hiển thị các đường viền tìm được. 
# Trong trường hợp này, cv.RETR_LIST yêu cầu hàm trả về tất
#  cả các đường viền tìm thấy trong danh sách (LIST). 

# cv.CHAIN_APPROX_SIMPLE sử dụng phương pháp biểu diễn đường
#  viền đơn giản, chỉ lưu trữ các điểm đầu cuối của đoạn đường 
# thay vì tất cả các điểm trên đường viền. Điều này giúp tiết 
# kiệm bộ nhớ và làm giảm độ phức tạp của đường viền.
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# su dung f-string
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)