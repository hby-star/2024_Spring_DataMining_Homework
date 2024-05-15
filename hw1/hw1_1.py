import matplotlib.pyplot as plt
import statistics
import numpy as np

data = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]

# 求均值
mean_value = statistics.mean(data)
print(f'Mean: {mean_value}')

# 求中位数
median_value = statistics.median(data)
print(f'Median: {median_value}')

# 求众数
mode_value = statistics.mode(data)
print(f'Mode: {mode_value}')

# 求中列数
min_value = np.min(data)
max_value = np.max(data)
midrange = (min_value+max_value)/2
print(f'MidRange: {midrange}')

# 求五数概括
minimum = np.min(data)
q1 = np.percentile(data, 25)
median = np.median(data)
q3 = np.percentile(data, 75)
maximum = np.max(data)
print(f'FiveNum: {minimum},{q1},{median},{q3},{maximum}')

# 画箱型图
plt.boxplot(data)
plt.title('Box Plot')
plt.ylabel('age')
plt.show()
