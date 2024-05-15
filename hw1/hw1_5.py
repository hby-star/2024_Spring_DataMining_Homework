import numpy as np
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(vec1, vec2):
    dot_product = dot(vec1, vec2)
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


x1 = [1.5, 1.7]
x2 = [2, 1.9]
x3 = [1.6, 1.8]
x4 = [1.2, 1.5]
x5 = [1.5, 1.0]

x_all = [x1, x2, x3, x4, x5]

x = [1.4, 1.6]

# 欧式距离
print('欧式距离')
for i in x_all:
    dist = norm(np.array(x) - np.array(i), ord=2)
    print(dist)
print('###')

# 曼哈顿距离
print('曼哈顿距离')
for i in x_all:
    dist = norm(np.array(x) - np.array(i), ord=1)
    print(dist)
print('###')

# 上确界距离
print('上确界距离')
for i in x_all:
    dist = norm(np.array(x) - np.array(i), ord=np.inf)
    print(dist)
print('###')

# 余弦相似性
print('余弦相似性')
for i in x_all:
    dist = cosine_similarity(x, i)
    print(dist)
print('###')

# 规格化后的欧式距离
print('规格化后的欧式距离')
for i in x_all:
    dist = norm(np.array(x) / norm(np.array(x)) - np.array(i) / norm(np.array(i)), ord=2)
    print(dist)
print('###')
