import cv2 as cv
import numpy as np

img = cv.imread('./opencv/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Dòng mã ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) trong OpenCV được sử dụng để thực hiện một phép xác định ngưỡng (thresholding) trên một hình ảnh xám 
# (grayscale image). Phép xác định ngưỡng là một phép toán xử lý hình ảnh thường được sử dụng để chuyển đổi hình ảnh xám thành hình ảnh nhị phân, trong đó mỗi pixel chỉ có hai giá trị: 
# 0 hoặc 255 (hoặc 0 hoặc 1).

# Các tham số trong dòng mã có ý nghĩa như sau:

# gray: Hình ảnh xám đầu vào mà bạn muốn thực hiện phép xác định ngưỡng lên.
# 125: Ngưỡng, là một giá trị cường độ pixel. Tất cả các pixel trong hình ảnh có giá trị lớn hơn ngưỡng sẽ được đặt thành 255 (hoặc 1, tùy thuộc vào định dạng hình 
# ảnh nhị phân), và tất cả các pixel có giá trị nhỏ hơn ngưỡng sẽ được đặt thành 0.
# 255: Giá trị tương ứng với các pixel vượt qua ngưỡng. Ở đây, giá trị 255 thường được sử dụng để biểu thị vùng trắng.
# cv.THRESH_BINARY: Tham số này chỉ định cách mà phép xác định ngưỡng hoạt động. Trong trường hợp này, cv.THRESH_BINARY có nghĩa là mọi pixel có giá trị lớn hơn ngưỡng
#  sẽ được đặt thành 255, và mọi pixel có giá trị nhỏ hơn hoặc bằng ngưỡng sẽ được đặt thành 0.
# Kết quả của dòng mã này là hai biến ret và thresh. Biến ret chứa giá trị ngưỡng thực sự được sử dụng, và biến thresh là hình ảnh nhị phân sau khi xác định ngưỡng. 
# Trong hình ảnh nhị phân, pixel có giá trị 255 thường đại diện cho vùng trắng (đối tượng hoặc vùng quan tâm), trong khi pixel có giá trị 0 đại diện cho nền hoặc các vùng không quan trọng.
# Thường được sử dụng khi bạn muốn làm rõ vùng quan tâm và nền trong hình ảnh, chẳng hạn trong việc nhận dạng vật thể, phân đoạn vùng quan tâm, hay xác định các khu vực quan tâm.
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
# contours: Một danh sách (list) chứa các đường biên tìm thấy. Mỗi đường biên là một danh sách các điểm pixel (các đỉnh của đường biên)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# su dung f-string
print(f'{len(contours)} contour(s) found!')

# Dòng mã cv.drawContours(blank, contours, -1, (0, 0, 255), 1) trong OpenCV được sử dụng để vẽ các đường biên (contours) lên một hình ảnh trắng hoặc hình ảnh nền (blank). Điều này cho phép 
# bạn hiển thị các đường biên đã tìm thấy trên một hình ảnh trắng hoặc hình ảnh khác.
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)