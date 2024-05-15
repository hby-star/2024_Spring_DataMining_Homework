import numpy as np

a = np.array([22, 1, 42, 10])
b = np.array([20, 0, 36, 8])

# 欧式距离
e_distance = np.linalg.norm(a - b, ord=2)
print(e_distance)

# 曼哈顿距离
m_distance = np.linalg.norm(a - b, ord=1)
print(m_distance)

# q=3的闵可夫斯基距离
e_distance = np.linalg.norm(a - b, ord=3)
print(e_distance)

# 上确界距离
c_distance = np.linalg.norm(a - b, ord=np.inf)
print(c_distance)
