import numpy as np

# 假设有两个相似度矩阵 A 和 B
A = np.array([[1, 1, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 1, 1]])

B = np.array([[1, 0.8, 0.65, 0.55],
              [0.8, 1, 0.7, 0.6],
              [0.65, 0.7, 1, 0.9],
              [0.55, 0.6, 0.9, 1]])

# 计算 A 和 B 的皮尔逊相关系数
pearson_correlation = np.corrcoef(A.flatten(), B.flatten())[0, 1]

print("Pearson correlation between matrix A and B:", pearson_correlation)
