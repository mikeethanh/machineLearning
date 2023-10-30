import cv2 as cv 

# img = cv.imread('./Photos/cat_large.jpg')
# cv.im('Cat',img)

def rescaleFrame(frame ,scale = 0.75):
    # frame.shape trả về một tuple (chiều cao, chiều rộng, số kênh màu)
    # frame shape[1]: tra ve chieu rong cua khung hinh 
    width = int(frame.shape[1] * scale)
    # frame.shape[0] : tra ve chieu cao cua khung hinh
    height = int(frame.shape[0] * scale)
    #  Ở đây, một tuple dimensions được tạo để lưu trữ chiều rộng và chiều cao mới tính toán ở các bước trước.
    dimentions = (width,height)
    # Đoạn mã này sử dụng hàm cv.resize() của OpenCV để thay đổi kích thước hình ảnh frame thành
    #  dimensions, với việc sử dụng phương pháp nội suy (interpolation) là
    # cv.INTER_AREA. Phương pháp nội suy giúp làm mượt hình ảnh sau khi 
    # thay đổi kích thước để tránh hiện tượng pixelation (hiện tượng các 
    # pixel rõ rệt) trong hình ảnh sau khi thu nhỏ hoặc phóng to. 
    return cv.resize(frame , dimentions,interpolation = cv.INTER_AREA)

# thay doi kich thuoc o Live video
def changeRes(width,heigh):
    capture.set(3,width)
    capture.set(4,heigh)

capture = cv.VideoCapture('./Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
    # if cv.waitKey(20) & 0xFF==ord('d'):
    # This is the preferred way - if `isTrue` is false (the frame could 
    # not be read, or we're at the end of the video), we immediately
    # break from the loop. 
    if isTrue:    
        cv.imshow('Video', frame)
        cv.imshow('Video_resized',frame_resized)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
    else:
        break

capture.release()
cv.destroyAllWindows()