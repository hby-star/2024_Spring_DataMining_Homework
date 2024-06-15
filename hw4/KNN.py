import csv
import random
import sys
import time
from collections import Counter

import numpy as np


class KNN:
    def __init__(self, k=5, distance='manhattan'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.Y_train = None

    def fit(self, X: list, Y: list) -> None:
        self.X_train = X
        self.Y_train = Y

    def predict(self, X: list) -> list:
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # 计算距离
        if self.distance == 'euclidean':
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'chebyshev':
            distances = [self._chebyshev_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError('Invalid distance')

        # 获取最近的k个样本
        k_indices = np.argsort(distances)[:self.k]

        # 获取k个样本的标签
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        # 返回出现次数最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    @staticmethod
    def _euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def _manhattan_distance(x1, x2):
        return np.sum(np.abs(x1 - x2))

    @staticmethod
    def _chebyshev_distance(x1, x2):
        return np.max(np.abs(x1 - x2))


def main(X: list, Y: list, test_x: list, k=5, distance='manhattan') -> list:
    knn = KNN(k, distance)
    knn.fit(np.array(X), np.array(Y))
    return knn.predict(np.array(test_x))


def load_csv(filename):
    """
    读取csv文件
    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        ret = list(reader)
    data = ret[1:]
    # 字符串转数字
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    return data


def spit_data(data, test_rate):
    """
    划分数据集
    :param data:
    :param test_rate:
    :return:
    """
    random.shuffle(data)

    train_X, train_Y, test_X, test_Y = [], [], [], []
    train_len = int(len(data) * (1 - test_rate))
    for i in range(len(data)):
        if i < train_len:
            train_X.append(data[i][1:-1])
            train_Y.append(data[i][-1])
        else:
            test_X.append(data[i][1:-1])
            test_Y.append(data[i][-1])
    return train_X, train_Y, test_X, test_Y


def grid_search(data, test_rate):
    """
    网格搜索
    :param data:
    :param test_rate:
    :return:
    """
    # 搜索参数
    search_space = {
        'k': [5, 10, 20, 30],
        'distance': ['manhattan', 'euclidean', 'chebyshev']
    }
    # 对于每一个参数组合，进行3次测试，取平均值
    search_num = 3
    # 搜索最佳参数
    with open('search_result.log', 'w') as f:

        # 重定向输出流到文件
        sys.stdout = f

        for k in search_space['k']:
            for distance in search_space['distance']:
                print('--------------------------------------------')
                print(f'k: {k}, distance: {distance}')
                accuracy_list = []
                for i in range(search_num):
                    start_time = time.time()
                    train_X, train_Y, test_X, test_Y = spit_data(data, test_rate)

                    predict_Y = main(train_X, train_Y, test_X, k, distance)
                    accuracy = compute_accuracy(predict_Y, test_Y)

                    end_time = time.time()
                    print(f'accuracy{i}: {accuracy}  time: {end_time - start_time}')
                    accuracy_list.append(accuracy)
                print(f'average_accuracy: {sum(accuracy_list) / len(accuracy_list)}')
                print('--------------------------------------------')
                sys.stdout.flush()

        # 恢复标准输出流
        sys.stdout = sys.__stdout__


def compute_accuracy(predict_Y, test_Y):
    """
    计算准确率
    :param predict_Y:
    :param test_Y:
    :return:
    """
    count = 0
    for i in range(len(predict_Y)):
        if predict_Y[i] == test_Y[i]:
            count += 1
    return count / len(predict_Y)


if __name__ == '__main__':
    data = load_csv('./data/span_pub.csv')

    test_rate = 0.3

    # grid_search(data, test_rate)

    train_X, train_Y, test_X, test_Y = spit_data(data, test_rate)
    print(f'train_data_len: {len(train_X)}')
    print(f'test_data_len: {len(test_X)}')

    predict_Y = main(train_X, train_Y, test_X)
    accuracy = compute_accuracy(predict_Y, test_Y)

    print(f'accuracy: {accuracy}')
