# 构造测试数据
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import multivariate_normal
import numpy as np
np.random.seed(0)
data, labels = make_blobs(n_samples=1000, n_features=20, centers=5)

# 实现算法
def kmeanspp(X, K):
    n = X.shape[0]
    # 初始化质心列表并随机选择一个质心
    centroids = [X[np.random.randint(0, n)]]

    for _ in range(1, K):
        # 计算每个数据点到最近质心的距离
        D = euclidean_distances(X, centroids).min(axis=1)
        # 选择下一个质心，概率与距离成正比
        probs = D / D.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(X[i])

    # 迭代更新质心
    assignments = assign_points(X, centroids)
    old_assignments = None
    while not np.array_equal(assignments, old_assignments):
        new_centroids = compute_centroids(X, assignments, K)
        old_assignments = assignments
        assignments = assign_points(X, new_centroids)

    return assignments

def assign_points(X, centroids):
    distances = euclidean_distances(X, centroids)
    return np.argmin(distances, axis=1)

def compute_centroids(X, assignments, K):
    centroids = np.array([X[assignments == k].mean(axis=0) for k in range(K)])
    return centroids


def em(X, K, max_iter=100):
    n, d = X.shape

    # 初始化参数
    weights = np.ones((K)) / K
    means = np.random.choice(X.flatten(), (K,d))
    covs = np.array([np.eye(d)] * K)

    for _ in range(max_iter):
        # E step
        likelihood = np.zeros((n, K))
        for i in range(K):
            # 将协方差矩阵加上一个小的正数，防止奇异矩阵
            covs[i] += np.eye(d) * 1e-6
            distribution = multivariate_normal(means[i], covs[i])
            likelihood[:,i] = distribution.pdf(X)

        b = likelihood * weights
        b = b / np.sum(b, axis=1, keepdims=True)

        # M step
        for i in range(K):
            weight = b[:, [i]]
            total_weight = weight.sum()
            means[i] = (X * weight).sum(axis=0) / total_weight
            covs[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)

        weights = np.mean(b, axis=0)

    # 分配标签
    labels = np.argmax(b, axis=1)

    return labels

from sklearn.metrics import adjusted_rand_score
kmeanspp_labels=kmeanspp(data, 5)
ari = adjusted_rand_score(labels, kmeanspp_labels)
print(ari)
assert ari>0.999
em_labels=em(data, 5)
ari = adjusted_rand_score(labels, em_labels)
print(ari)
assert ari>0.45
print("PASS")