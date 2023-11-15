import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Tạo dữ liệu giả định
np.random.seed(0)
X = np.random.rand(100, 1) * 100  # 100 mẫu với giá trị từ 0 đến 100
y = 3 * X.squeeze() + np.random.randn(100) * 10  # Biểu thức giả định y = 3X + noise

# Hiển thị dữ liệu
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dữ liệu Linear Regression')
plt.show()

# Áp dụng Linear Regression
model = LinearRegression()
model.fit(X, y)

# In ra các thông số của mô hình
print("Hệ số hồi quy (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
