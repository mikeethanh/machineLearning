data = pd.read_csv('./linearRegression/data_linear.csv').values

# # shape[0] đại diện cho số hàng (số dòng) của mảng.
# # shape[1] đại diện cho số cột của mảng.
# # N = data.shape[0]: Đây là dòng đầu tiên và nó tính toán số mẫu (số hàng)
# #  trong dữ liệu của bạn và gán giá trị này cho biến N. Trong ngữ cảnh này, N biểu thị
# #  tổng số điểm dữ liệu trong tập dữ liệu.
# N = data.shape[0]
# # x = data[:, 0].reshape(-1, 1): Dòng này tạo một biến x bằng cách lấy 
# # cột đầu tiên (cột 0) của dữ liệu data. Cột đầu tiên thường chứa dữ 
# # liệu về biến độc lập (trục x) trong bài toán hồi quy tuyến tính. 
# # Sau đó, .reshape(-1, 1) được sử dụng để biến đổi x thành một mảng NumPy 2 chiều với một cột 
# # và số dòng tương ứng với N.


# # x = data[:, 0].reshape(-1, 1): Dòng này tạo một biến x bằng cách lấy cột đầu tiên (cột 0)
# #  của dữ liệu data. Cột đầu tiên thường chứa dữ liệu về biến độc lập (trục x) trong bài toán 
# # hồi quy tuyến tính. Sau đó, .reshape(-1, 1) được sử dụng để biến đổi x thành một mảng NumPy
# #  2 chiều với một cột và số dòng tương ứng với N
# x = data[:, 0].reshape(-1, 1)
# y = data[:, 1].reshape(-1, 1)

# plt.scatter(x, y)
# plt.xlabel('mét vuông')
# plt.ylabel('giá')

# # np.ones((N, 1)): Đoạn này tạo một ma trận 1 (ma trận toàn số 1) có kích thước (N, 1). Điều này đặt một cột giá trị 1 vào ma trận.
# #  Ma trận này thường được sử dụng để biểu diễn giá trị bias (tham số w0) trong mô hình hồi quy tuyến tính.

# # np.hstack((...)): Hàm này nối theo chiều ngang (horizontal) nhiều mảng hoặc ma trận lại với nhau. Trong trường hợp này, nó nối ma
# #  trận 1 (bias) và ma trận x theo chiều ngang, tạo thành một ma trận kết quả mở rộng cho dữ liệu hồi quy tuyến tính. Kết quả sẽ có kích thước (N, 2) với N là số mẫu dữ liệu.
# x = np.hstack((np.ones((N, 1)), x))
# # Dòng code w = np.array([0., 1.]).reshape(-1, 1) được sử dụng để khởi tạo vector trọng số (weights) của mô hình hồi quy tuyến tính
# w = np.array([0.,1.]).reshape(-1,1)

# numOfIteration = 100
# cost = np.zeros((numOfIteration,1))
# learning_rate = 0.000001
# for i in range(1, numOfIteration):
#     # Hàm np.dot trong NumPy được sử dụng để thực hiện phép nhân ma trận hoặc phép nhân vector. Hàm này có thể được sử dụng để nhân ma trận với ma trận, 
#     # vector với ma trận hoặc vector với vector.

#     # r là một vector, và mỗi phần tử trong vector r tương ứng với hiệu giữa giá trị dự đoán và giá trị thực tế cho từng điểm dữ liệu. Dòng code r = np.dot(x, w) - y 
#     # thực hiện tính toán sự sai lệch giữa giá trị dự đoán và giá trị thực tế cho tất cả các điểm dữ liệu trong tập dữ liệu.

#     # Trong trường hợp này, x là ma trận dữ liệu đầu vào (bao gồm một cột ma trận có giá trị 1 và một cột ma trận x), w là vector trọng số của mô hình (bao gồm trọng số w0 và w1),
#     #  và y là vector giá trị thực tế tương ứng cho từng điểm dữ liệu.
#     r = np.dot(x, w) - y
#     cost[i] = 0.5*np.sum(r*r)
#     w[0] -= learning_rate*np.sum(r)
#     # correct the shape dimension
#     w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
#     print(cost[i])
# predict = np.dot(x, w)
# plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
# plt.show()
# x1 = 50
# y1 = w[0] + w[1] * x1
# print('Giá nhà cho 50m^2 là : ', y1)
# # Lưu w với numpy.save(), định dạng '.npy'
# # np.save('weight.npy', w)
# # Đọc file '.npy' chứa tham số weight
# # w = np.load('weight.npy')