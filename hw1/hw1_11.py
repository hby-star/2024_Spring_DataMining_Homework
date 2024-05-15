import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# 读取数据集
read_data = pd.read_csv("data/kddcup.data_10_percent_corrected_10000.csv", header=None)

# 选择分类属性列
categorical_data = read_data.iloc[:, [1, 2, 3]]
class_data = read_data.iloc[:, [1, 2, 3, 41]]

# 创建一个仅包含分类属性的数据集
data = np.array(categorical_data)
test_data = np.array(class_data)

# 使用LabelEncoder将分类数据转换为数字编码
label_encoders = [LabelEncoder() for _ in range(data.shape[1])]
data_encoded = np.array([label_encoders[i].fit_transform(data[:, i]) for i in range(data.shape[1])]).T

# 匹配度量
nn_match = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='hamming')
nn_match.fit(data_encoded)

# 逆发生频率度量
nn_inverse = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='matching')
nn_inverse.fit(data_encoded)

# 使用match measure计算最近邻
distances_match, indices_match = nn_match.kneighbors(data_encoded)

# 使用inverse occurrence frequency measure计算最近邻
distances_inv, indices_inv = nn_inverse.kneighbors(data_encoded)

# 计算具有类标签匹配的案例数量
num_matches = sum(test_data[i][-1] == test_data[indices_match[i, 1]][-1] for i in range(len(test_data)))

# 计算具有类标签匹配的案例数量
num_inverse = sum(test_data[i][-1] == test_data[indices_inv[i, 1]][-1] for i in range(len(test_data)))

print("Num match:", num_matches)
print("Num inverse:", num_inverse)
