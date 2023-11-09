# Hàm sigmoid
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# # Load data từ file csv
# data = pd.read_csv('./logitsticRegression/dataset.csv').values

# # Trong đoạn mã N, d = data.shape, biến N sẽ chứa số hàng của mảng data, và biến d sẽ chứa số cột của mảng data.
# N, d = data.shape
# x = data[:, 0:d-1].reshape(-1, d-1)
# y = data[:, 2].reshape(-1, 1)
# # Vẽ data bằng scatter
# plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
# plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
# plt.legend(loc=1)
# plt.xlabel('mức lương (triệu)')
# plt.ylabel('kinh nghiệm (năm)')
# # Thêm cột 1 vào dữ liệu x

# # Thêm cột 1 vào dữ liệu x bằng x = np.hstack((np.ones((N, 1)), x)) là để tạo một cột dữ liệu tự do (bias term) trong mô hình hồi quy tuyến tính. Cột này có giá trị bằng 1 và thường được sử dụng để biểu diễn hệ số tự do của phương trình hồi quy tuyến tính.

# # Trong phương trình hồi quy tuyến tính, bạn có một số hệ số (slope) cho từng biến đầu vào và một hệ số tự do (intercept), nó biểu diễn giá trị dự đoán khi tất cả các biến đầu vào đều bằng 0. Thêm cột 1 vào dữ liệu giúp bạn biểu diễn hệ số tự do này.

# # Ví dụ: Nếu bạn có một mô hình hồi quy tuyến tính dạng y = ax + b, thì hệ số tự do (intercept) là b. Thêm cột 1 vào dữ liệu x tương đương với việc mở rộng mô hình thành y = ax + bx', trong đó b là hệ số tự do và x' tương ứng với cột dữ liệu tự do có giá trị 1.
# x = np.hstack((np.ones((N, 1)), x))

# w = np.array([0.,0.1,0.1]).reshape(-1,1)
# # Số lần lặp bước 2
# numOfIteration = 1000
# cost = np.zeros((numOfIteration,1))
# learning_rate = 0.01
# for i in range(1, numOfIteration):
# # Tính giá trị dự đoán
#     y_predict = sigmoid(np.dot(x, w))
#     cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + \
#     np.multiply(1-y, np.log(1-y_predict)))
#     # Gradient descent
#     w = w - learning_rate * np.dot(x.T, y_predict-y)
#     print(cost[i])
# # Vẽ đường phân cách.
# t = 0.5
# plt.plot((4, 10),(-(w[0]+4*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 10*w[1]+ \
#         np.log(1/t-1))/w[2]), 'g')
# plt.show()
# # Lưu weight dùng numpy.save(), định dạng '.npy'
# np.save('weight logistic.npy', w)
# # Load weight từ file ''.npy'
# w = np.load('weight logistic.npy')