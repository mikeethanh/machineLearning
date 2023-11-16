import numpy as np

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dữ liệu XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Thêm cột 1 vào dữ liệu X để có bias
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# Số lần lặp và learning rate
num_iterations = 10000
learning_rate = 0.01

# Khởi tạo trọng số ngẫu nhiên
np.random.seed(0)
weights = np.random.rand(X_bias.shape[1])

# Huấn luyện mô hình Logistic Regression
for i in range(num_iterations):
    # Tính giá trị dự đoán
    predictions = sigmoid(np.dot(X_bias, weights))
    
    # Tính độ lỗi
    error = y - predictions
    
    # Cập nhật trọng số theo gradient descent
    weights += learning_rate * np.dot(X_bias.T, error)

# Dự đoán đầu ra
final_predictions = (predictions >= 0.5).astype(int)

# In kết quả dự đoán cuối cùng
print("Predictions:", final_predictions)
