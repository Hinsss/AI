# 文件名: polynomial_regression.py

# 多项式回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 模拟数据集
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
# 做出数据集的散点图
plt.plot(X, y, 'b.')

d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    # 根据i的再弄几列x^i+x^(i-1)...（生维，为了让线性模型去拟合非线性模型），把非线性的数据变化成类似线性的变化，然后用线性模型去拟合，下面就是
    # include_bias = False即w0为False,因为后面的回归算法会添加上
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    # print(X[0])
    # print(X_poly[0])
    # print(X_poly)
    # print(X_poly[:, 0])

    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_poly, y)
    # 打印出w0,[w1,w2,w3...]
    print(lin_reg.intercept_, lin_reg.coef_)

    # 将转化的数据预测出来作图比较
    y_predict = lin_reg.predict(X_poly)
    # 用第一列数据来作图就行，因为本来都是从第一列衍生出来的数据
    plt.plot(X_poly[:, 0], y_predict, d[i])

plt.show()