import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('./opencv/Session3Face/haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

#  tạo ra một mô hình nhận diện khuôn mặt sử dụng phương pháp LBPH 
face_recognizer = cv.face.LBPHFaceRecognizer.create()

# face_recognizer: Đây là biến hoặc đối tượng mô hình nhận diện khuôn mặt sử dụng phương pháp LBPH.
# .read(): Đây là một phương thức của mô hình nhận diện khuôn mặt, được sử dụng để đọc mô 
# hình từ một tệp dữ liệu đã được lưu trữ. Tệp này chứa thông tin về mô hình, bao gồm trọng 
# số đã được học và các tham số cấu hình.
face_recognizer.read('./opencv/Session3Face/face_trained.yml')

# ây là một hàm của thư viện OpenCV để đọc hình ảnh từ một tệp ảnh. Hàm này chấp nhận
#  đường dẫn đến tệp ảnh và trả về một mảng NumPy chứa dữ liệu hình ảnh.
#  Trong trường hợp này, hình ảnh được đọc từ đường dẫn đã được chỉ định
img = cv.imread(r'F:\machineLearning\opencv\Faces\val\elton_john\1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    # : Dòng này sử dụng các giá trị x, y, w, và h của mỗi hộp chữ nhật 
    # để cắt ra một vùng con (Region of Interest - ROI) từ hình ảnh xám 
    # (grayscale image) được lưu trữ trong biến gray. Kết quả của dòng
    #  này là biến faces_roi, chứa vùng khuôn mặt đã được cắt ra từ hình ảnh gốc.
    faces_roi = gray[y:y+h,x:x+w]

    # Sử dụng mô hình nhận diện khuôn mặt, hàm predict được gọi
    #  để dự đoán người trong khuôn mặt. Hàm này trả về hai giá trị:

# label: Là nhãn của người mà khuôn mặt được dự đoán thuộc về. Mỗi người được 
# đại diện bằng một nhãn riêng biệt trong quá trình đào tạo.
# confidence: Độ tin cậy của việc dự đoán, thường là một số vô hướng, có 
# giá trị thấp nếu mô hình không tự tin về dự đoán của mình và cao nếu mô hình 
# tự tin hơn. Giá trị này có thể được sử dụng để kiểm tra mức độ chính xác của dự đoán
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # (20, 20): Đây là tọa độ (x, y) của gốc của đoạn văn bản trên hình ảnh. Trong 
    # trường hợp này, văn bản sẽ được vẽ tại điểm có tọa độ (20, 20) từ góc trái của hình ảnh.
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)