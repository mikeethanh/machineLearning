import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

# masked = cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Mask', masked)

# GRayscale histogram

# Mục đích chính của việc tính toán histogram là phân tích phân bố của 
# các mức sáng (intensity levels) trong hình ảnh. Histogram có thể giúp 
# bạn hiểu rõ hình ảnh bằng cách biểu thị tần suất xuất hiện của các mức
#  sáng khác nhau.
# gray: Đây là hình ảnh nguồn mà bạn muốn tính toán histogram. 
# Hàm này mong đợi một danh sách các hình ảnh đầu vào, và trong 
# trường hợp này, bạn đưa gray vào một danh sách (list) gồm một phần tử.
#  Mặc dù bạn chỉ có một hình ảnh, nhưng bạn vẫn phải đưa nó vào một danh
#  sách để tuân theo cú pháp của hàm cv.calcHist().

# [0]: Đây là một danh sách (list) các chỉ số của kênh mà bạn muốn tính
#  histogram. Trong hầu hết các trường hợp, hình ảnh xám (grayscale) chỉ 
# có một kênh, nên bạn sử dụng [0] để chỉ định kênh duy nhất đó. Đối với
#  hình ảnh màu (RGB), bạn có thể sử dụng [0], [1], và [2] để tham khảo 
# các kênh màu đỏ, xanh lá cây và xanh dương.

# None: Đây là mặt nạ (mask), bạn có thể sử dụng mặt nạ để chỉ tính 
# histogram cho một vùng cụ thể của hình ảnh, nhưng nếu bạn muốn tính 
# toán histogram cho toàn bộ hình ảnh, bạn có thể truyền None để bỏ qua mặt nạ.

# [256]: Đây là số lượng bins (ngăn) cho histogram. Histogram được chia
#  thành các bins, và bạn có thể xác định số lượng bins mà bạn muốn. 
# Trong trường hợp này, bạn chọn 256 bins để đo tần suất của 256 mức sáng
#  khác nhau (từ 0 đến 255) trong hình ảnh xám.
gray_hist = cv.calcHist([gray], [0], None, [256], [0,256] )

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
# Dòng mã plt.xlim([0, 256]) được sử dụng để đặt giới hạn trục x (hoặc trục ngang) 
plt.xlim([0,256])
plt.show()

# Colour Histogram

plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
# ỗi phần tử của colors được gán cho biến col, đồng thời biến i sẽ 
# lưu trữ chỉ số của phần tử đó trong danh sách.

#  enumertate: thường được sử dụng để lặp qua danh sách và thực hiện các tác vụ 
# dựa trên cả giá trị của phần tử và chỉ số của nó.
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)