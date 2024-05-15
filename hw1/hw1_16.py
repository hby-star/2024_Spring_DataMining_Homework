import numpy as np

age = np.array([23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61])
fat = np.array([9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7])


def z_score_normalization(data):
    return (data - np.mean(data)) / np.std(data)


print("z分数-年龄：", z_score_normalization(age).round(2))
print("z分数-脂肪：", z_score_normalization(fat).round(2))

corr_coef = np.corrcoef(age, fat)[0, 1]
print(f'相关系数： {corr_coef.round(2)}')
