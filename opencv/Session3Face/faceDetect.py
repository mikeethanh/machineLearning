# Chính xác, bộ phân loại Haar Cascade là một phương pháp nhận diện đối 
# tượng trong hình ảnh dựa trên các đặc trưng Haar. Nó hoạt động bằng cách 
# sử dụng các tập hợp các bộ lọc Haar để nhận diện các đặc trưng quan trọng 
# trong hình ảnh, sau đó sử dụng các đặc trưng này để phân loại đối tượng.

# Một bộ lọc Haar là một loại bộ lọc dựa trên hình dạng, thường là các hình dạng
#  như cạnh, góc, hoặc đặc trưng quan trọng khác. Bộ lọc Haar có thể phát hiện 
# các đặc trưng bằng cách so sánh tổng giá trị pixel trên các vùng con hình ảnh khác nhau.

# Bộ phân loại Haar Cascade bao gồm nhiều bộ lọc Haar khác nhau và sử dụng chúng để xác định
#  xem một phần cụ thể của hình ảnh có chứa đối tượng cần nhận diện không. Nó thường được sử dụng
#  để phát hiện khuôn mặt trong hình ảnh, nhưng cũng có thể được áp dụng cho việc nhận diện các đối tượng khác
import cv2 as cv

img = cv.imread('./opencv/Photos/group 1.jpg')
cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)
# Dòng này tạo một đối tượng CascadeClassifier sử dụng tệp XML 
# "haar_face.xml" để phát hiện khuôn mặt. 
# Tính năng Haar (Haar feature) là một loại tính năng dùng để mô tả cụ thể vùng quan tâm trên hình ảnh. Tính năng Haar thường được sử dụng trong các thuật toán phát hiện
#  đối tượng trong xử lý hình ảnh, đặc biệt là trong mô hình Haar Cascade.

# Tính năng Haar là một loại bộ lọc đặc biệt được sử dụng để xác định các biểu đồ hoặc cấu trúc cụ thể trong hình ảnh. Để hiểu cách tính năng Haar hoạt động, hãy xem xét các ví dụ sau:

# Tính năng Haar đơn giản (Simple Haar Feature): Một tính năng Haar đơn giản là một hình chữ nhật có một phần trắng và một phần đen. Tính năng này được di chuyển trên hình ảnh
#  để tính toán sự khác biệt giữa tổng giá trị pixel trong vùng trắng và tổng giá trị pixel trong vùng đen. Điều này cho phép tính toán đặc trưng vùng cụ thể trên hình ảnh.

# Kết hợp các tính năng Haar: Tính năng Haar đơn giản có thể được kết hợp lại với nhau để tạo ra các tính năng phức tạp hơn. Ví dụ: một tính năng Haar có thể bao gồm một chuỗi
#  các hình chữ nhật có thể nối tiếp hoặc chồng lên nhau để tạo ra một đặc trưng mạng lưới.

# Sử dụng trong mô hình Haar Cascade: Mô hình Haar Cascade sử dụng các tính năng Haar để phát hiện các đối tượng, chẳng hạn như khuôn mặt, trong hình ảnh. Các tính năng Haar 
# được huấn luyện để phát hiện các đặc trưng quan trọng của đối tượng (ví dụ: mắt, mũi, miệng trong trường hợp phát hiện khuôn mặt) và sau đó được sử dụng để quét qua hình
#  ảnh và xác định vị trí của các đối tượng.

# Tính năng Haar là một phần quan trọng của phương pháp Haar Cascade và đã được chứng minh là hiệu quả trong việc phát hiện các đối tượng trên hình ảnh.

# CascadeClassifier là một lớp trong OpenCV được sử dụng để tạo một đối tượng phát hiện đối tượng sử dụng mô hình Haar Cascade. Đối tượng CascadeClassifier được tạo ra để phát
#  hiện các đối tượng cụ thể, chẳng hạn như khuôn mặt, trong hình ảnh bằng cách sử dụng mô hình Haar Cascade.

# Phương thức chính mà bạn có thể sử dụng với đối tượng CascadeClassifier bao gồm:

# detectMultiScale(): Phương thức này cho phép bạn phát hiện các đối tượng cụ thể trong hình ảnh và trả về một danh sách (list) các hình chữ nhật (bounding boxes) bao quanh đối
#  tượng đã phát hiện. Các hình chữ nhật này đại diện cho vị trí và kích thước của đối tượng.
# Phương thức detectMultiScale() có thể được sử dụng để phát hiện khuôn mặt (hoặc bất kỳ đối tượng cụ thể nào mà mô hình Haar Cascade đã được huấn luyện để phát hiện) trong hình 
# ảnh. Khi mô hình được tạo và đối tượng CascadeClassifier được khởi tạo với mô hình, bạn có thể sử dụng nó để phát hiện đối tượng trong hình ảnh.
haar_cascade = cv.CascadeClassifier('opencv/Session3Face/haar_face.xml')

#  Dòng này sử dụng phương thức detectMultiScale của CascadeClassifier
#  để tìm khuôn mặt trong hình ảnh xám (gray). Các tham số đi kèm bao gồm:

# scaleFactor: Tham số này xác định mức độ co giãn của hình ảnh khi quét.
#  Giá trị 1.1 cho biết hình ảnh sẽ được co giãn 10% mỗi lần khi quét.
# minNeighbors: Tham số này xác định số lượng hàng xóm cần phát 
# hiện trước khi một vùng được xem là một khuôn mặt.

# Hàm này trả về một danh sách (hoặc mảng) các hộp chữ nhật (rectangles) 
# mô tả vị trí và kích thước của các đối tượng được phát hiện trong hình ảnh.

# Danh sách faces_rect chứa các hộp chữ nhật, mỗi hộp chữ nhật biểu diễn vị
#  trí và kích thước của một đối tượng được phát hiện trong hình ảnh. Mỗi hộp 
# chữ nhật được biểu diễn bằng bốn giá trị (x, y, w, h)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

# Dòng này lặp qua từng khuôn mặt được tìm thấy và trích xuất tọa
#  độ và kích thước của mỗi khuôn mặt
for (x,y,w,h) in faces_rect:
    #  Dòng này vẽ một hộp chữ nhật màu xanh lá cây xung quanh mỗi khuôn mặt được 
    # tìm thấy bằng cách sử dụng hàm cv.rectangle
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)


cv.waitKey(0)