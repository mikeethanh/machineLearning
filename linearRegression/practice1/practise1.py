import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Tạo dữ liệu giả định
np.random.seed(0)
X = np.random.rand(100, 1) * 100
y = 3 * X.squeeze() + np.random.randn(100) * 10

# Hiển thị dữ liệu
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dữ liệu Linear Regression')

# Thêm cột bias vào dữ liệu X
N = X.shape[0]
y = y.reshape(-1, 1)
X_b = np.hstack((np.ones((N, 1)), X))

# Số lượng vòng lặp và learning rate
num_iterations = 1000
learning_rate = 0.0001

# Khởi tạo các hệ số
w = np.array([0., 1.]).reshape(-1, 1)

# Gradient Descent
for i in range(num_iterations):
    y_pred = np.dot(X_b, w)
    error = y_pred - y
    gradient = np.dot(X_b.T, error)
    w -= learning_rate * gradient / N

# Đường thẳng dự đoán
predict = np.dot(X_b, w)
plt.plot(X, predict, 'r')
plt.show()

# Áp dụng Linear Regression
# model = LinearRegression()
# model.fit(X, y)

# # Vẽ đường thẳng dự đoán
# plt.plot(X, model.predict(X), color='red', label='Predicted line')
# plt.legend()
# plt.show()
# # In ra các thông số của mô hình
# print("Hệ số hồi quy (slope):", model.coef_[0])
# print("Intercept:", model.intercept_)
