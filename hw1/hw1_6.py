import numpy as np
import matplotlib.pyplot as plt

num_groups = 10
group_size = 100
num_samples = 1000
sample_sizes = range(10, 70)

data = np.random.rand(num_groups * group_size).reshape(num_groups, group_size)

success_rates = []

for sample_size in sample_sizes:
    successes = 0
    for _ in range(num_samples):
        sample_indices = np.random.choice(num_groups * group_size, sample_size, replace=False)
        sample = data.flatten()[sample_indices].reshape(-1, 1)
        similarities = np.abs(sample - sample.T)

        # 找到每组的最大值
        similar_points = []
        for i in range(num_groups):
            group_max_value = np.max(similarities[i])
            similar_points.append(group_max_value)

        representative_points = np.unique(similar_points)
        if len(np.unique(representative_points)) >= num_groups:
            successes += 1
    success_rate = successes / num_samples
    success_rates.append(success_rate)

plt.plot(sample_sizes, success_rates)
plt.xlabel('Sample Size')
plt.ylabel('Success Rate')
plt.title('Success Rate of Representative Points vs. Sample Size')
plt.show()
