# ./: Đại diện cho thư mục hiện tại, nghĩa là thư mục mà mã hoặc tệp đang thực thi trong đó.
# ../: Đại diện cho thư mục cha, nghĩa là thư mục cấp trên so với thư mục hiện tại.
import os
import cv2 as cv
import numpy as np

# LBPHF (Local Binary Pattern Histograms with Fourier Transform) là một phương
#  pháp nhận diện khuôn mặt sử dụng biểu đồ các đặc trưng mô tả các mẫu cục bộ
#  trong khuôn mặt. Đây là một biến thể của LBPH (Local Binary Pattern Histograms), 
# mà đã được cải tiến bằng cách áp dụng biến đổi Fourier để cải thiện hiệu suất của mô hình. 
# Dưới đây là cách hoạt động của mô hình nhận diện khuôn mặt LBPHF:

# Trích xuất đặc trưng cục bộ: Mô hình LBPHF bắt đầu bằng việc chia khuôn mặt
#  thành các ô cục bộ (local patches). Cho mỗi ô cục bộ, mô hình sử dụng Local Binary Pattern (LBP) 
# để trích xuất một biểu đồ nhị phân mô tả sự biến đổi của cường độ pixel trong ô so với các điểm lân cận.

# Biểu đồ LBP và Fourier Transform: Sau khi trích xuất biểu đồ LBP cho mỗi ô cục bộ, 
# mô hình áp dụng biến đổi Fourier cho các biểu đồ này. Biến đổi Fourier cho phép biểu đồ 
# LBP được biểu diễn trong miền tần số, giúp giảm chi phí tính toán và cải thiện hiệu suất của mô hình.

# Tạo histogram tần số: Dựa trên biểu đồ LBP sau biến đổi Fourier, mô hình tạo ra 
# một histogram tần số. Histogram này mô tả phân phối của các giá trị tần số trong 
# biểu đồ LBP và cung cấp thông tin quan trọng về mẫu cục bộ trên khuôn mặt.

# Huấn luyện mô hình: Trước khi sử dụng, mô hình cần được huấn luyện trên
#  một tập dữ liệu lớn chứa các khuôn mặt và các nhãn tương ứng. Huấn luyện 
# mô hình giúp nó học cách phân biệt giữa các người khác nhau dựa trên các đặc
#  trưng trích xuất từ biểu đồ LBP và Fourier.

# Nhận diện khuôn mặt: Khi mô hình đã được huấn luyện, nó có thể được sử dụng để
#  nhận diện khuôn mặt trong các hình ảnh mới. Mô hình trích xuất các đặc trưng từ 
# khuôn mặt trong hình ảnh mới, sau đó so sánh chúng với các đặc trưng đã học từ 
# tập huấn luyện để xác định xem khuôn mặt thuộc về ai.

# LBPHF là một trong những phương pháp nhận diện khuôn mặt phổ biến và hiệu quả, 
# đặc biệt là trong những tình huống mà bạn cần một phương pháp đơn giản và hiệu
#  quả. Tuy nhiên, nó có thể không đạt được hiệu suất cao trong các tình huống 
# phức tạp hoặc khi có biến đổi lớn về ánh sáng, góc độ hoặc che khuất.

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

DIR = r'F:\machineLearning\opencv\Faces\train'

haar_cascade = cv.CascadeClassifier('./opencv/Session3Face/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        # sử dụng hàm os.path.join() để kết hợp đường dẫn DIR (đường dẫn đến thư mục gốc chứa các
        #  thư mục của các người) và chuỗi person (tên của người cụ thể). Kết quả là biến path 
        # sẽ lưu trữ đường dẫn đến thư mục con chứa hình ảnh của người đó
        path = os.path.join(DIR, person)
        label = people.index(person)
        # Hàm os.listdir() trả về một danh sách chứa tên của tất cả
        #  các tệp và thư mục nằm trong thư mục path. Sau đó, bạn có 
        # thể sử dụng danh sách này để duyệt qua từng tệp hình ảnh trong 
        # thư mục của người cụ thể và xử lý chúng trong quá trình đào tạo.
        for img in os.listdir(path):
            # s.path.join(path, img): Hàm os.path.join()
            #  được sử dụng để nối đường dẫn đến thư mục chứa 
            # hình ảnh (path) và tên của tệp hình ảnh (img). Kết
            #  quả là một đường dẫn đầy đủ đến tệp hình ảnh cụ thể.
            img_path = os.path.join(path,img)
            # cv.imread(img_path): Hàm cv.imread() của thư viện OpenCV
            #  được sử dụng để đọc tệp hình ảnh từ đường dẫn đã đượ
            # c tạo trước đó và lưu trữ nó trong biến img_array. Sau 
            # khi dòng này thực hiện, img_array sẽ chứa dữ liệu hình 
            # ảnh từ tệp hình ảnh đã được đọc.
            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                # faces_roi = gray[y:y+h, x:x+w]: Dòng này trích xuất một vùng
                #  ảnh con (ROI - Region of Interest) từ hình ảnh xám (gray) bằng 
                # cách sử dụng tọa độ x, y, w, và h của hộp chữ nhật. Điều này có nghĩa 
                # là nó cắt ra vùng chứa khuôn mặt từ hình ảnh xám và lưu trữ nó trong 
                # biến faces_roi.
                faces_roi = gray[y:y+h, x:x+w]
                # Sau khi trích xuất thành công vùng khuôn mặt (faces_roi), nó được thêm 
                # vào danh sách features. Danh sách này chứa các vùng khuôn mặt của tất 
                # cả các hình ảnh đào tạo.
                features.append(faces_roi)
                # Nhãn tương ứng của người được thêm vào danh sách labels. Nhãn này 
                # cho biết người trong vùng khuôn mặt đã được trích xuất. Danh sách labels
                #  chứa các nhãn tương ứng với từng vùng khuôn mặt trong danh sách features.
                labels.append(label)

# Quá trình này được thực hiện cho tất cả các hộp chữ nhật đã được phát hiện trong 
# hình ảnh đào tạo, và sau khi hoàn thành, danh sách features sẽ chứa các vùng khuôn mặt
#  và danh sách labels sẽ chứa các nhãn tương ứng với người trong các vùng khuôn mặt này. 
# Điều này làm cho dữ liệu đào tạo sẵn sàng cho việc xây dựng mô hình nhận diện khuôn mặt.

# Gọi hàm create_train() để đào tạo hệ thống nhận diện và xây dựng danh sách đặc trưng và nhãn.
create_train()
print('Training done ---------------')

# Danh sách (List): Lưu trữ các phần tử khác loại (các kiểu dữ liệu khác nhau) và 
# không yêu cầu tất cả các phần tử có cùng kiểu dữ liệu.
# Mảng NumPy: Yêu cầu tất cả các phần tử trong mảng có cùng kiểu dữ liệu, điều 
# này giúp làm cho việc thực hiện các phép toán trên mảng nhanh chóng hơn.

# Lưu trữ hiệu quả dữ liệu: Mảng NumPy có cách lưu trữ dữ liệu hiệu quả hơn
#  so với danh sách Python thông thường. Việc chuyển đổi danh sách thành mảng
#  NumPy giúp tiết kiệm bộ nhớ và cải thiện hiệu suất khi làm việc với dữ liệu lớn.

# Tích hợp với thư viện OpenCV: Một số phần của thư viện OpenCV yêu cầu dữ 
# liệu phải có định dạng mảng NumPy. Chuyển đổi danh sách thành mảng NumPy làm
#  cho dữ liệu dễ dàng sử dụng với các hàm và phương thức của OpenCV.

# Đồng nhất hóa dữ liệu: Chuyển đổi danh sách thành mảng NumPy giúp đảm bảo
#  rằng dữ liệu có cùng định dạng, loại dữ liệu và kích thước. Điều này quan
#  trọng khi bạn làm việc với các thuật toán máy học hoặc tiền xử lý dữ liệu.

# Tương thích với nhiều thư viện khác nhau: Mảng NumPy là một đối tượng phổ
#  biến và phù hợp với nhiều thư viện và công cụ trong ngữ cảnh xử lý dữ liệu
# và học máy.

# xác định kiểu dữ liệu của mảng là "object," có thể chứa bất kỳ loại dữ liệu nào.
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer.create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

# Dòng này sử dụng phương thức .save() của mô hình face_recognizer để lưu trữ mô
#  hình đã được đào tạo vào một tệp có tên 'face_trained.yml'. Tệp này chứa thông
#  tin về mô hình đã được huấn luyện và sẽ được sử dụng sau này để nhận diện khuôn
#  mặt trên các hình ảnh mới.
face_recognizer.save('face_trained.yml')
# Dòng này sử dụng thư viện NumPy để lưu trữ danh sách features (các vùng khuôn mặt)
#  vào một tệp có tên 'features.npy'. Tệp này sẽ chứa thông tin về các đặc trưng của 
# các khuôn mặt đã được trích xuất và sẽ được sử dụng để huấn luyện mô hình sau này.
np.save('features.npy', features)
# np.save('labels.npy', labels): Tương tự như trên, dòng này sử dụng NumPy để lưu trữ 
# danh sách labels (các nhãn tương ứng với người trong các vùng khuôn mặt) vào tệp có tên
#  'labels.npy'. Tệp này chứa thông tin về các nhãn và sẽ được sử dụng trong quá trình đào 
# tạo mô hình và nhận diện khuôn mặt
np.save('labels.npy', labels)

# .yml: Mở rộng tệp .yml đại diện cho "YAML" (YAML Ain't Markup Language). YAML là một ngôn 
# ngữ đánh dấu dựa trên văn bản sử dụng để biểu diễn dữ liệu dưới dạng văn bản, và nó thường
#  được sử dụng để lưu trữ cấu hình và dữ liệu cấu trúc. Trong ngữ cảnh của mô hình nhận diện
#  khuôn mặt, tệp .yml có thể được sử dụng để lưu trữ thông tin về mô hình đã được huấn luyện,
#  ví dụ: trọng số của mô hình và các thông số cấu hình.

# npy: Mở rộng tệp .npy liên quan đến NumPy, một thư viện Python phổ biến cho tính toán số và xử
#  lý dữ liệu đa chiều. Tệp .npy chứa dữ liệu đã được lưu trữ dưới dạng mảng NumPy. Điều này có
#  lợi ích làm cho việc lưu trữ và tải dữ liệu hiệu quả hơn, đồng thời bảo đảm tính nhất quán về 
# kiểu dữ liệu và kích thước dữ liệu.