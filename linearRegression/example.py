import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#đoạn code sinh ra dữ liệu
#numOfPoint = 30
#noise = np.random.normal(0,1,numOfPoint).reshape(-1,1)
#x = np.linspace(30, 100, numOfPoint).reshape(-1,1)
#N = x.shape[0]

data = pd.read_csv('./linearRegression/data_linear.csv').values

# shape[0] đại diện cho số hàng (số dòng) của mảng.
# shape[1] đại diện cho số cột của mảng.
# N = data.shape[0]: Đây là dòng đầu tiên và nó tính toán số mẫu (số hàng)
#  trong dữ liệu của bạn và gán giá trị này cho biến N. Trong ngữ cảnh này, N biểu thị
#  tổng số điểm dữ liệu trong tập dữ liệu.
N = data.shape[0]
# x = data[:, 0].reshape(-1, 1): Dòng này tạo một biến x bằng cách lấy 
# cột đầu tiên (cột 0) của dữ liệu data. Cột đầu tiên thường chứa dữ 
# liệu về biến độc lập (trục x) trong bài toán hồi quy tuyến tính. 
# Sau đó, .reshape(-1, 1) được sử dụng để biến đổi x thành một mảng NumPy 2 chiều với một cột 
# và số dòng tương ứng với N.


# x = data[:, 0].reshape(-1, 1): Dòng này tạo một biến x bằng cách lấy cột đầu tiên (cột 0)
#  của dữ liệu data. Cột đầu tiên thường chứa dữ liệu về biến độc lập (trục x) trong bài toán 
# hồi quy tuyến tính. Sau đó, .reshape(-1, 1) được sử dụng để biến đổi x thành một mảng NumPy
#  2 chiều với một cột và số dòng tương ứng với N
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

# np.ones((N, 1)): Đoạn này tạo một ma trận 1 (ma trận toàn số 1) có kích thước (N, 1). Điều này đặt một cột giá trị 1 vào ma trận.
#  Ma trận này thường được sử dụng để biểu diễn giá trị bias (tham số w0) trong mô hình hồi quy tuyến tính.

# np.hstack((...)): Hàm này nối theo chiều ngang (horizontal) nhiều mảng hoặc ma trận lại với nhau. Trong trường hợp này, nó nối ma
#  trận 1 (bias) và ma trận x theo chiều ngang, tạo thành một ma trận kết quả mở rộng cho dữ liệu hồi quy tuyến tính. Kết quả sẽ có kích thước (N, 2) với N là số mẫu dữ liệu.
x = np.hstack((np.ones((N, 1)), x))
# Dòng code w = np.array([0., 1.]).reshape(-1, 1) được sử dụng để khởi tạo vector trọng số (weights) của mô hình hồi quy tuyến tính
w = np.array([0.,1.]).reshape(-1,1)

numOfIteration = 100
cost = np.zeros((numOfIteration,1))
learning_rate = 0.000001
for i in range(1, numOfIteration):
    # Hàm np.dot trong NumPy được sử dụng để thực hiện phép nhân ma trận hoặc phép nhân vector. Hàm này có thể được sử dụng để nhân ma trận với ma trận, 
    # vector với ma trận hoặc vector với vector.

    # r là một vector, và mỗi phần tử trong vector r tương ứng với hiệu giữa giá trị dự đoán và giá trị thực tế cho từng điểm dữ liệu. Dòng code r = np.dot(x, w) - y 
    # thực hiện tính toán sự sai lệch giữa giá trị dự đoán và giá trị thực tế cho tất cả các điểm dữ liệu trong tập dữ liệu.

    # Trong trường hợp này, x là ma trận dữ liệu đầu vào (bao gồm một cột ma trận có giá trị 1 và một cột ma trận x), w là vector trọng số của mô hình (bao gồm trọng số w0 và w1),
    #  và y là vector giá trị thực tế tương ứng cho từng điểm dữ liệu.
    r = np.dot(x, w) - y
    cost[i] = 0.5*np.sum(r*r)
    w[0] -= learning_rate*np.sum(r)
    # correct the shape dimension
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
    print(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.show()
x1 = 50
y1 = w[0] + w[1] * x1
print('Giá nhà cho 50m^2 là : ', y1)
# Lưu w với numpy.save(), định dạng '.npy'
np.save('weight.npy', w)
# Đọc file '.npy' chứa tham số weight
w = np.load('weight.npy')


# LinearRegression với thư viện sklearn
from sklearn.linear_model import LinearRegression
data = pd.read_csv('./linearRegression/data_linear.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')
# Tạo mô hình hồi quy tuyến tính
lrg = LinearRegression()
# Train mô hình với data giá đất
lrg.fit(x, y)
# Đoán giá nhà đất
y_pred = lrg.predict(x)
plt.plot((x[0], x[-1]),(y_pred[0], y_pred[-1]), 'r')
plt.show()
# Lưu nhiều tham số với numpy.savez(), định dạng '.npz'
np.savez('w2.npz', a=lrg.intercept_, b=lrg.coef_)
# Lấy lại các tham số trong file .npz
k = np.load('w2.npz')
lrg.intercept_ = k['a']
lrg.coef_ = k['b']


# Linear regression la mot thuat toan giu[ giai quyet cac bai toan co au ra la gia tri thuc te 

# dua vao cac thong tin ma de bai cho vi du , gia tien phu thuoc vao dien tich , phong ngu , bla blo
# y = ax1 + bx2 +cx3 + d-> hay goi cach khac y = w1x1 + w2x2 +w3x3 + w4
# x1 la dien tich 
# x2 la phong ngu 
# x3 ... goi chung la cac thuoc tinh 

# vay viec minh can lam la tinh cac w1 w2 3 w4 hay con goi la cac trong so sao cho 
# fit nhat voi du lieu dau vao 

# tinh bang cach nao : 
# gia su w1 w2 w3 w4 co cac gia tri ban dau la 1 , 2 , 3 
# neu cho cac gia tri dau vao nhuw the thi gia tri du doan su co muc chenh
# lech rat lon so voi gia tri thuc te hay con goi la loss function

# lossfunction co hai loai la MSE va MAE

# vay moi sinh ra thuat toan gradient de tim gia tri nho nhat cua ham lossfunction 
# tuc la tinh do chenh lech nho nhat 

# thuat toan gradient decent
# 1. Khởi tạo giá trị x = x0 tùy ý
# 2. Gán x = x - learning_rate * f’(x) ( learning_rate là hằng số dương ví dụ learning_rate = 0.001)
# 3. Tính lại f(x): Nếu f(x) đủ nhỏ thì dừng lại, ngược lại tiếp tục bước 2

# sau khi tim duoc do chenh lech nho nhat
# thi minh se tim ra duoc cac trong so them cong thuc cua gradient decent 
# 
# vay de bieu dien bai toa ntren duoi dang code ta bieu dien duoi dang ma tran 
# ma tran x voi 1 hang n cot (cac thuoc tinh , gia tri cua dien tich ) 
# ma tran y voi 1 hang n cot (gia tri thuc , gia nha thuc te ) 
# sau do them mot hang vao truc x bieu dien cho he so tu do 

# sau khi them mot hang va otruc x de bieu dien he so tu do r ta nhan ma tran x voi ma tran w
# ma tran w la cac trong so 

