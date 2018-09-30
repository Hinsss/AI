import numpy as np
from sklearn.linear_model import LinearRegression
# 线性回归

# 假装建立一个数据训练集
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 构建回归模型
lin_reg = LinearRegression()
# 模型训练
lin_reg.fit(X, y)
# 类似w0和w1的值打印出来，看是否相似
print(lin_reg.intercept_, lin_reg.coef_)

# 将测试值输进去，得到预测值
X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))