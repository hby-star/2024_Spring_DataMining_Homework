import numpy as np

# 给定数据组
data = np.array([200, 300, 400, 600, 1000])


# 最小-最大规范化
def min_max_normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# z分数规范化
def z_score_normalization(data):
    return (data - np.mean(data)) / np.std(data)


# z分数规范化，使用均值绝对偏差
def mad(data):
    median = np.median(data)
    return np.median(np.abs(data - median))


def z_score_mad_normalization(data):
    return (data - np.median(data)) / mad(data)


# 小数定标规范化
def decimal_scaling(data):
    return data / 1000


# 输出规范化后的数据
print("最小-最大规范化结果：", min_max_normalization(data))
print("z分数规范化结果：", z_score_normalization(data))
print("z分数规范化（使用均值绝对偏差）结果：", z_score_mad_normalization(data))
print("小数定标规范化结果：", decimal_scaling(data))
