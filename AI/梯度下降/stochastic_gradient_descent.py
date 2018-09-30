import numpy as np

# 随机梯度下降法

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 500
t0,t1 = 5,50

m = 100

# 步长越来越小
def learning_schedule(t):
    return t0/(t + t1)

# 初始化
theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        learning_rate = learning_schedule(epoch*m+i)
        theta = theta - learning_rate*gradients

print(theta)