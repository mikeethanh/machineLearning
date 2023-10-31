#pylint:disable=no-member

import cv2 as cv

img = cv.imread('./opencv/Photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging
# uá trình trung bình hóa hoạt động bằng cách tính giá trị trung bình 
# của các pixel trong vùng cụ thể của hình ảnh (kích thước kernel) và 
# sau đó gán giá trị trung bình này cho pixel tại tâm của vùng đó. 
# Điều này dẫn đến việc làm mờ hình ảnh để giảm đi chi tiết và làm 
# cho nó trở nên mượt hơn.
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

# Gaussian Blur
# Gaussian Blur thường được sử dụng để giảm nhiễu và làm mờ hình ảnh
#  trước khi thực hiện các phân tích hình ảnh khác, như trích xuất đặc 
# trưng hoặc nhận dạng đối tượng.
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)

# Median Blur
# Việc sử dụng Median Blur thường rất hiệu quả trong việc loại bỏ nhiễu 
# bởi nó không bị ảnh hưởng bởi giá trị ngoại lai (outlier) hay giá trị 
# cực đoan (extreme value), trong khi các phương pháp làm mờ trung bình 
# có thể dễ dàng bị ảnh hưởng bởi những giá trị này. Do đó, Median Blur
#  thường được ưa chuộng trong các trường hợp nơi bạn muốn làm mờ hình 
# ảnh mà vẫn giữ lại các cạnh và chi tiết quan trọng.
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral
# Bilateral Blur là một phương pháp làm mờ hình ảnh được sử dụng để làm 
# mờ hình ảnh mà vẫn duy trì rất nhiều chi tiết cạnh (edge) quan trọng.
#  Phương pháp này kết hợp cả việc làm mờ không gian và làm mờ mức độ xám 
# (intensity) của pixel trong hình ảnh. Nó được sử dụng để giảm nhiễu 
# trong hình ảnh trong khi vẫn giữ lại sự tương phản và định dạng của 
# các cạnh trong hình ảnh.
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)