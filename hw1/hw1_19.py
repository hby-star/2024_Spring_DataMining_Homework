import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


# 计算欧式距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# 计算最近邻,使用三角不等式的性质加速
def computeNNindex(x, distances, centers):
    d_c1_x = euclidean_distance(x, centers[0])
    c1_index = 0

    for i in range(1, len(centers)):
        d_c1_c2 = distances[c1_index, i]

        # 此处使用三角不等式的性质减少计算量
        if d_c1_c2 > 2 * d_c1_x:
            continue

        d_c2_x = euclidean_distance(x, centers[i])
        if d_c2_x < d_c1_x:
            d_c1_x = d_c2_x
            c1_index = i

    return c1_index


# 加载Iris数据集
iris = load_iris()
X = iris.data

# 初始化K-means模型
kmeans = KMeans(n_clusters=3, random_state=0)

# 使用K-means算法进行聚类
kmeans.fit(X)

# 获取聚类中心
centers = kmeans.cluster_centers_
n_centers = len(centers)

# 初始化存储距离的矩阵
distances = np.zeros((n_centers, n_centers))

# 计算并存储每对点i,j之间的距离
for i in range(n_centers):
    for j in range(i + 1, n_centers):
        distance = euclidean_distance(centers[i], centers[j])
        distances[i, j] = distance
        distances[j, i] = distance

# 计算并打印索引
index = []
for i in range(len(X)):
    index.append(computeNNindex(X[i], distances, centers))
print(index)

# 可视化聚类结果
plt.figure(figsize=(8, 6))

# 绘制聚类结果
for i in range(3):
    cluster_points = X[np.where(np.array(index) == i)]  # 获取属于第 i 类的数据点
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='Cluster {}'.format(i+1))

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1], s=100, c='red', label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering with Triangle Inequality Improvement')
plt.legend()
plt.show()