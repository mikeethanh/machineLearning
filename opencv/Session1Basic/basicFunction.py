# Hình ảnh được tạo ra bằng cách sắp xếp các điểm ảnh (pixels) trên một mặt phẳng hoặc không gian để biểu thị các đối tượng, cảnh quan, hoặc dữ liệu trực quan. Mỗi điểm ảnh trong hình ảnh
#  biểu thị một giá trị cường độ (intensity) hoặc màu sắc tại một vị trí cụ thể trong không gian 2D hoặc 3D. Hình ảnh màu thường sử dụng ba kênh màu (Red, Green, Blue - RGB), trong
#  khi hình ảnh đen trắng (grayscale) chỉ sử dụng một kênh cường độ.

# Hình ảnh được biểu diễn bằng ma trận (hoặc mảng đa chiều) bởi vì cách này thuận tiện cho việc xử lý dữ liệu số học và tính toán trên hình ảnh. Mỗi phần tử trong ma trận tương ứng với 
# một pixel trên hình ảnh và chứa giá trị cường độ (trong trường hợp hình ảnh đen trắng) hoặc các giá trị màu sắc (trong trường hợp hình ảnh màu).

# Cách hình ảnh được biểu diễn bằng ma trận thường như sau:

# Trong trường hợp hình ảnh grayscale (đơn kênh):

# Mỗi phần tử trong ma trận là một số thực, thể hiện cường độ tại một pixel cụ thể. Giá trị này thường nằm trong khoảng từ 0 đến 255, với 0 biểu thị màu đen và 255 biểu thị màu trắng.
# Trong trường hợp hình ảnh màu (nhiều kênh):

# Mỗi phần tử trong ma trận là một vectơ 3 chiều hoặc 4 chiều (tùy thuộc vào định dạng màu sắc, thường là RGB hoặc BGR). Mỗi chiều của vectơ đại diện cho giá trị màu tại một pixel cụ thể.
# Việc biểu diễn hình ảnh bằng ma trận cho phép dễ dàng thực hiện các phép toán số học, xử lý hình ảnh và phân tích dữ liệu hình ảnh bằng cách sử dụng các thư viện xử lý hình ảnh như OpenCV 
# hoặc NumPy. Ma trận là một cấu trúc dữ liệu phổ biến trong lĩnh vực xử lý hình ảnh và khoa học dữ liệu.
import cv2 as cv

# Read in an image
# Hàm imread trả về một ma trận NumPy biểu diễn hình ảnh (hoặc None nếu việc đọc hình ảnh không thành công). Bạn có thể sử dụng biến img để 
# thực hiện các thao tác xử lý hình ảnh khác nhau, như làm sáng tối, lọc, cắt, chuyển đổi màu sắc và nhiều công việc khác liên quan đến xử lý hình ảnh.
img = cv.imread('./opencv/Photos/park.jpg')
cv.imshow('Park', img)

# Converting to grayscale
# Cụ thể, dòng mã này thực hiện việc chuyển đổi hình ảnh img từ không gian màu BGR sang không gian màu grayscale và kết quả được lưu trữ trong biến gray.
#  Sau khi chuyển đổi, biến gray sẽ chứa hình ảnh grayscale tương ứng với hình ảnh ban đầu.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur 
# 7,7 xác định độ mờ 
# cv.BORDER_DEFAULT: Tham số này đại diện cho cách xử lý viền của hình 
# ảnh khi thực hiện phép biến đổi. 

#  Lọc Gaussian là một phương pháp xử lý hình ảnh để làm mờ hình ảnh bằng cách sử dụng một hàm Gauss để tính trung bình cường độ pixel trong vùng xung quanh 
# của mỗi pixel. Các tham số trong dòng mã có ý nghĩa như sau:

# img: Hình ảnh gốc bạn muốn áp dụng lọc Gaussian lên.
# (7, 7): Kích thước của kernel (bộ lọc) Gaussian. Trong trường hợp này, kernel có kích thước là 7x7, nghĩa là nó sẽ tính trung bình cường độ của các pixel trong một
#  vùng 7x7 pixel xung quanh mỗi pixel trong hình ảnh.
# Kết quả của dòng mã này là hình ảnh blur sau khi áp dụng lọc Gaussian. Hình ảnh blur sẽ được làm mờ và giảm nhiễu, giúp cải thiện khả năng phát hiện cạnh và làm cho
#  hình ảnh trở nên mượt hơn. Lọc Gaussian thường được sử dụng trong xử lý hình ảnh để chuẩn bị dữ liệu trước khi thực hiện các công việc khác như phát hiện cạnh hoặc trích xuất đặc trưng.
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
# Trong ngữ cảnh xử lý hình ảnh, "biên" (edge) là sự thay đổi đột ngột trong cường độ pixel hoặc màu sắc giữa các vùng trên hình ảnh. Biên đại diện cho ranh giới giữa các đối 
# tượng hoặc cấu trúc trong hình ảnh. Các biên thường xuất hiện ở những nơi mà một vật thể kết thúc hoặc bắt đầu, hoặc nơi có sự thay đổi đột ngột về cường độ pixel hoặc màu sắc.

# Phân đoạn vùng ảnh: Biên có thể được sử dụng để tách các vùng khác nhau trên hình ảnh. Điều này hữu ích trong việc nhận dạng các đối tượng trong hình ảnh.

# Phát hiện đối tượng: Phát hiện biên có thể giúp xác định hình dạng và kích thước của các đối tượng trong hình ảnh, giúp trong việc nhận dạng và đếm đối tượng.

# Nhận dạng khuôn mặt và các đặc trưng của khuôn mặt: Trong việc nhận dạng khuôn mặt, phát hiện biên có thể giúp xác định các đặc điểm quan trọng như biên mắt, mũi, miệng, vv.

# Giảm nhiễu: Biên thường cách biệt giữa vùng nhiễu và vùng đối tượng trong hình ảnh. Nó có thể được sử dụng để giảm nhiễu và tăng độ chính xác của xử lý hình ản

# Dòng mã canny = cv.Canny(blur, 125, 175) trong OpenCV được sử dụng để thực hiện phát hiện biên bằng thuật toán Canny trên hình ảnh đã được làm mờ blur. 
# Thuật toán Canny là một phương pháp phát hiện biên trong xử lý hình ảnh và thường được sử dụng để tạo ra một hình ảnh nhị phân (hình ảnh chỉ chứa hai giá trị: 0 hoặc 255) 
# để biểu thị các đường biên trên hình ảnh.

# Các tham số trong dòng mã có ý nghĩa như sau:

# blur: Hình ảnh đã được làm mờ (thường là bằng lọc Gaussian) trước khi áp dụng phát hiện biên.
# 125: Ngưỡng dưới, dùng để xác định các điểm pixel có cường độ thấp hơn ngưỡng này không được coi là biên.
# 175: Ngưỡng trên, dùng để xác định các điểm pixel có cường độ cao hơn ngưỡng này sẽ được coi là điểm trên đường biên.
# Kết quả của dòng mã này là hình ảnh canny, trong đó các pixel trên đường biên sẽ có giá trị 255, còn lại là 0. Hình ảnh canny thường được sử dụng để tạo ra một biểu đồ biên 
# hoặc để tìm các cạnh quan trọng trong hình ảnh.
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Dilating the image
# iterations=3: Điều này chỉ định số lần lặp (iterations) để thực hiện phép Dilation. 
# Hàm dilate trong OpenCV thực hiện phép biển đạt (dilation) trên hình ảnh. Phép biển đạt hoạt động như sau:

# Hàm dilate nhận một hình ảnh đầu vào, thường là một hình ảnh nhị phân (binary image), trong đó có các vùng trắng (255) và các vùng đen (0). Hình ảnh này thường được tạo ra từ các phép
#  phát hiện biên như Canny hoặc Laplacian.

# Hàm dilate sử dụng một kernel (bộ lọc) có kích thước xác định để quét qua hình ảnh. Kernel này thường là một ma trận vuông với kích thước và hình dạng được xác định trước. Kernel 
# được áp dụng lên từng pixel của hình ảnh đầu vào.

# Tại mỗi vị trí của kernel trên hình ảnh, nếu có ít nhất một pixel trắng nào đó nằm trong vùng tương ứng với kernel, thì pixel tại vị trí đó trên hình ảnh kết quả sẽ được đặt thành
#  trắng (255). Ngược lại, nếu không có pixel trắng nào, pixel tại vị trí đó sẽ được đặt thành đen (0).

# Quá trình này được lặp lại trên toàn bộ hình ảnh. Số lần lặp (được xác định bởi tham số iterations trong hàm) quyết định mức độ mở rộng của biển đạt. Mỗi lần lặp tiếp theo sẽ làm
#  to biển hơn, mở rộng vùng trắng trên hình ảnh.

# Phép biển đạt thường được sử dụng trong xử lý hình ảnh để thực hiện các công việc như:

# Mở rộng các đối tượng trên hình ảnh.
# Loại bỏ các lỗ hoặc khoảng trống nhỏ giữa các đối tượng.
# Tạo ra các vùng liên thông để dễ dàng xác định và phân tích các đối tượng riêng lẻ.
# Điền các lỗ hoặc các chi tiết nhỏ vào đối tượng.
# Quá trình biển đạt là một trong các phép toán xử lý hình ảnh cơ bản và quan trọng.
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
# Trong ngữ cảnh xử lý hình ảnh và xử lý tín hiệu, "nhiễu" (noise) đề cập đến các biến đổi hoặc thành phần ngẫu nhiên được thêm vào tín hiệu hoặc dữ liệu, gây ra sự biến đổi ngẫu 
# nhiên và không mong muốn. Nhiễu có thể xuất hiện trong nhiều loại dữ 
# liệu, bao gồm hình ảnh, âm thanh, tín hiệu điện tử, và nhiều loại tín hiệu khác.

# Dòng mã eroded = cv.erode(dilated, (7,7), iterations=3) trong OpenCV được sử dụng để thực hiện phép co ngược lại (erosion) trên hình ảnh đã được mở rộng (dilated). Phép erosion là
#  một phép toán xử lý hình ảnh ngược lại với phép dilation.

# Phép erosion hoạt động như sau:

# Hàm erode nhận một hình ảnh đầu vào, thường là một hình ảnh nhị phân, trong đó có các vùng trắng (255) và các vùng đen (0).

# Hàm erode sử dụng một kernel (bộ lọc) có kích thước xác định để quét qua hình ảnh. Kernel này thường là một ma trận vuông với kích thước và hình dạng được xác định trước.

# Tại mỗi vị trí của kernel trên hình ảnh, để pixel tại vị trí đó trên hình ảnh kết quả trở thành trắng (255), thì tất cả các pixel trong vùng tương ứng với kernel trên hình ảnh đầu 
# vào cũng phải là trắng. Nếu có ít nhất một pixel đen (0) trong vùng tương ứng với kernel, pixel tại vị trí đó trên hình ảnh kết quả sẽ được đặt thành đen (0).

# Quá trình này được lặp lại trên toàn bộ hình ảnh. Số lần lặp (được xác định bởi tham số iterations trong hàm) quyết định mức độ co lại của phép erosion. Mỗi lần lặp tiếp theo sẽ
#  làm nhỏ vùng trắng hơn.

# Phép erosion thường được sử dụng trong xử lý hình ảnh để thực hiện các công việc như:

# Loại bỏ nhiễu và các điểm nhiễu nhỏ trên hình ảnh.
# Co lại các vùng trắng, làm nhỏ đối tượng trên hình ảnh.
# Loại bỏ các khoảng trống hoặc độ dày của biên đã được mở rộng.
# Quá trình erosion là một phần quan trọng của việc xử lý và tiền xử lý hình ảnh, đặc biệt khi cần làm sạch và xử lý các vùng trắng hoặc đối tượng trên hình ảnh.
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

