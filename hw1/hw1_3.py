import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statistics

age = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fat = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]

# 求均值
mean_value = statistics.mean(age)
print(f'Mean age: {mean_value}')
mean_value = statistics.mean(fat)
print(f'Mean fat: {mean_value}')

# 求中位数
median_value = statistics.median(age)
print(f'Median age: {median_value}')
median_value = statistics.median(fat)
print(f'Median fat: {median_value}')

# 求标准差
sd_value = statistics.stdev(age)
print(f'Stdev age: {sd_value}')
sd_value = statistics.stdev(fat)
print(f'Stdev fat: {sd_value}')

plt.figure(figsize=(10, 6))
age = np.array([23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61])
fat = np.array([9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7])
# 箱型图
plt.boxplot([age, fat], labels=['Age', 'Fat'])
plt.title('Boxplot of Age and Fat')
plt.show()

# 散点图
plt.scatter(age, fat)
plt.xlabel('Age')
plt.ylabel('Fat')
plt.title('Scatterplot of Fat against Age')

plt.tight_layout()
plt.show()

# Q-Q图
age_sorted = np.sort(age)
fat_sorted = np.sort(fat)

age_sorted = (age_sorted - np.mean(age_sorted)) / np.std(age_sorted)
fat_sorted = (fat_sorted - np.mean(fat_sorted)) / np.std(fat_sorted)

plt.figure(figsize=(6,6))
plt.scatter(age_sorted, fat_sorted)
plt.plot([-3, 3], [-3, 3], color='red')  # add a reference line
plt.xlabel('Age')
plt.ylabel('Fat')
plt.title('Q-Q plot comparing Age and Fat')
plt.show()
