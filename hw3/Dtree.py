import csv
import sys
import time

import numpy as np
import random
from collections import Counter


# CART树实现。

class Node:
    def __init__(self, feature=None, threshold=None, data=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.data = data
        self.value = value
        self.left = left
        self.right = right


def _split(feature, threshold, X, y):
    """
    划分节点
    :param feature:
    :param threshold:
    :param X:
    :param y:
    :return:
    """
    left_idx = np.where(X[:, feature] <= threshold)
    right_idx = np.where(X[:, feature] > threshold)
    left = (X[left_idx], y[left_idx])
    right = (X[right_idx], y[right_idx])
    return left, right


def _most_common_label(y):
    """
    返回y中出现次数最多的元素
    :param y:
    :return:
    """
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class Dtree:
    def __init__(self, min_samples_split=5, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def _calculate_gini(y):
        """
        计算基尼指数
        :param y:
        :return:
        """
        classes = list(set(y))
        gini = 1.0
        for c in classes:
            p = list(y).count(c) / len(y)
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        """
        寻找最佳划分特征
        :param X:
        :param y:
        :return:
        """
        best_gini = 1
        best_feature, best_threshold = None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left, right = _split(feature, threshold, X, y)
                gini_left = self._calculate_gini(left[1])
                gini_right = self._calculate_gini(right[1])
                gini = len(left[1]) / len(y) * gini_left + len(right[1]) / len(y) * gini_right
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        构建决策树
        :param X:
        :param y:
        :param depth:
        :return:
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = _most_common_label(y)
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)
        left, right = _split(feature, threshold, X, y)
        left_node = self._build_tree(*left, depth + 1)
        right_node = self._build_tree(*right, depth + 1)
        return Node(feature, threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        """
        训练模型
        :param X:
        :param y:
        :return:
        """
        self.root = self._build_tree(X, y)

    def _predict(self, x, tree):
        """
        预测
        :param x:
        :param tree:
        :return:
        """
        if tree.value is not None:
            return tree.value
        if x[tree.feature] <= tree.threshold:
            return self._predict(x, tree.left)
        return self._predict(x, tree.right)

    def predict(self, X):
        """
        预测
        :param X:
        :return:
        """
        return [self._predict(x, self.root) for x in X]


def main(X: list, Y: list, test_x: list, min_samples_split=5, max_depth=10) -> list:
    cart = Dtree(min_samples_split, max_depth)
    cart.fit(np.array(X), np.array(Y))
    return cart.predict(np.array(test_x))


def load_csv(filename):
    """
    读取csv文件
    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        ret = list(reader)
    return ret[1:]


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
        'min_samples_split': [5, 10, 20],
        'max_depth': [5, 10, 20]
    }
    # 对于每一个参数组合，进行3次测试，取平均值
    search_num = 3
    # 搜索最佳参数
    with open('search_result.log', 'w') as f:

        # 重定向输出流到文件
        sys.stdout = f

        for min_samples_split in search_space['min_samples_split']:
            for max_depth in search_space['max_depth']:
                print('--------------------------------------------')
                print(f'min_samples_split: {min_samples_split}, max_depth: {max_depth}')
                accuracy = []
                for i in range(search_num):
                    start_time = time.time()
                    train_X, train_Y, test_X, test_Y = spit_data(data, test_rate)
                    predict_Y = main(train_X, train_Y, test_X, min_samples_split, max_depth)

                    count = 0
                    for j in range(len(predict_Y)):
                        if predict_Y[j] == test_Y[j]:
                            count += 1
                    end_time = time.time()
                    print(f'accuracy{i}: {count / len(predict_Y)}  time: {end_time - start_time}')
                    accuracy.append(count / len(predict_Y))
                print(f'average_accuracy: {sum(accuracy) / len(accuracy)}')
                print('--------------------------------------------')
                sys.stdout.flush()

        # 恢复标准输出流
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    data = load_csv('./data/span_pub.csv')

    test_rate = 0.3

    # grid_search(data, test_rate)

    train_X, train_Y, test_X, test_Y = spit_data(data, test_rate)
    print(f'train_data_len: {len(train_X)}')
    print(f'test_data_len: {len(test_X)}')

    predict_Y = main(train_X, train_Y, test_X)

    count = 0
    for i in range(len(predict_Y)):
        if predict_Y[i] == test_Y[i]:
            count += 1
    print(f'accuracy: {count / len(predict_Y)}')
