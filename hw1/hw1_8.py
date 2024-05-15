import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import jaccard_score


def cosine_similarity(vec1, vec2):
    dot_product = dot(vec1, vec2)
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def compute(vec1, vec2):
    # 余弦相似性
    dist = cosine_similarity(vec2, vec1)
    print(f'余弦： {dist}')
    # Pearson相关系数
    corr_coef = np.corrcoef(np.array(vec2), np.array(vec1))[0, 1]
    print(f'相关： {corr_coef}')
    # 欧式距离
    dist = norm(np.array(vec2) - np.array(vec1), ord=2)
    print(f'欧式： {dist}')
    # jaccard系数
    vec1_bin = (np.array(vec1) > 0).astype(int)
    vec2_bin = (np.array(vec2) > 0).astype(int)
    jaccard_coefficient = jaccard_score(vec1_bin, vec2_bin)
    print(f'jaccard： {jaccard_coefficient}')


print('#######')
compute([1, 1, 1, 1], [2, 2, 2, 2])
print('#######')
compute([0, 1, 0, 1], [1, 0, 1, 0])
print('#######')
compute([0, -1, 0, 1], [1, 0, -1, 0])
print('#######')
compute([1, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 1])
print('#######')
compute([2, -1, 0, 2, 0, -3], [-1, 1, -1, 0, 0, -1])
print('#######')
