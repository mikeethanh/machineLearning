# convert to black , white , 255 , 
import cv2 as cv 

img = cv.imread('./opencv/Photos/cats.jpg')
cv.imshow('Cats',img)

# simple thresholding

# cvt : convert
#  việc chuyển đổi hình ảnh sang không gian màu xám trước khi
#  áp dụng ngưỡng thường giúp đơn giản hóa xử lý hình ảnh và 
# làm cho quá trình xử lý hiệu quả hơn.
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Pixel (viết tắt của "picture element") là đơn vị cơ bản của
#  hình ảnh số hoặc hình ảnh kỹ thuật số. Nó là một điểm ảnh
#  nhỏ nhất trong hình ảnh và chứa thông tin về mức sáng hoặc
#  màu sắc tại một vị trí cụ thể trong hình ảnh.

# Trong hình ảnh xám (grayscale), mỗi pixel chứa một giá trị
#  duy nhất đại diện cho mức sáng tại vị trí tương ứng trong
#  hình ảnh

# threshold = 127
threshold,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)
cv.imshow('Thresh',thresh)


# invert
threshold,thresh_inv = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
cv.imshow('Thresh_invert',thresh_inv)


# adaptive thresholding

# Đây là kích thước của khu vực lân cận (neighborhood) được 
# sử dụng để tính giá trị trung bình cho phép ngưỡng thích 
# ứng. Kích thước này xác định cỡ của khung chữ nhật (hoặc 
# hình vuông) xung quanh mỗi pixel để tính giá trị trung bình. 
# Trong trường hợp này, khu vực lân cận có kích thước 11x11 pixel.

# 3: Đây là giá trị C (Constant) được sử dụng để điều chỉnh 
# ngưỡng. Sau khi tính giá trị trung bình của khu vực lân cận,
# giá trị C sẽ được trừ đi từ giá trị trung bình để xác định 
# giá trị ngưỡng cuối cùng.
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('Adaptive Thresholding',adaptive_thresh)


cv.waitKey(0)