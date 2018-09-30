import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

# 岭回归的l2

X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.rand(100,1)

# alpha就是正则化里的超参数
# ridge_reg = Ridge(alpha=1,solver='auto')
# ridge_reg.fit(X,y)
# print(ridge_reg.predict(1.5))
# print(ridge_reg.intercept_)
# print(ridge_reg.coef_)


# 这个类比较通用，只要传进penalty='l2'就等价于上面的方法
sgd_reg = SGDRegressor(penalty='l2', max_iter=1000)
# 跟以往不一样，这里要将y从列向量变成行向量
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(1.5))
print("W0=", sgd_reg.intercept_)
print("W1=", sgd_reg.coef_)