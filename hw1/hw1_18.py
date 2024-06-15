def read(file):
    # 读取文件并将数据存储在Instances中
    Instances = []
    fp = open(file, 'r')
    for line in fp:
        line = line.strip('\n')
        if line != '':
            Instances.append(line.split(','))
    fp.close()
    return Instances


def split(Instances, i):
    # 将Instances中的数据按照给定的索引i进行分割，构建log列表
    log = []
    for r in Instances:
        log.append([r[i], r[4]])
    return log


def count(log):
    # 对log列表中的数据进行计数
    log_cnt = []
    log.sort(key=lambda log: log[0])
    i = 0
    while i < len(log):
        cnt = log.count(log[i])
        record = log[i][:]
        record.append(cnt)
        log_cnt.append(record)
        i += cnt
    return log_cnt


def build(log_cnt):
    # 构建字典log_dic，记录不同类别的计数
    log_dic = {}
    for record in log_cnt:
        if record[0] not in log_dic.keys():
            log_dic[record[0]] = [0, 0, 0]
        if record[1] == 'Iris-setosa':
            log_dic[record[0]][0] = record[2]
        elif record[1] == 'Iris-versicolor':
            log_dic[record[0]][1] = record[2]
        elif record[1] == 'Iris-virginica':
            log_dic[record[0]][2] = record[2]
        else:
            raise TypeError("Data Exception")
    log_triple = sorted(log_dic.items())
    return log_triple


def collect(Instances, i):
    # 收集数据并构建log_tuple
    log = split(Instances, i)
    log_cnt = count(log)
    log_tuple = build(log_cnt)
    return log_tuple


def combine(a, b):
    # 将两个记录合并
    c = a[:]
    for i in range(len(a[1])):
        c[1][i] += b[1][i]
    return c


def chi2(A):
    # 计算卡方值
    m = len(A)
    k = len(A[0])
    R = []
    for i in range(m):
        sum_row = 0
        for j in range(k):
            sum_row += A[i][j]
        R.append(sum_row)

    C = []
    for j in range(k):
        sum_col = 0
        for i in range(m):
            sum_col += A[i][j]
        C.append(sum_col)

    N = sum(C)
    res = 0
    for i in range(m):
        for j in range(k):
            Eij = R[i] * C[j] / N
            if Eij != 0:
                res += (A[i][j] - Eij) ** 2 / Eij
    return res


def ChiMerge(log_tuple, max_interval):
    # 使用ChiMerge算法进行离散化处理
    num_interval = len(log_tuple)
    while num_interval > max_interval:
        num_pair = num_interval - 1
        chi_values = []
        for i in range(num_pair):
            arr = [log_tuple[i][1], log_tuple[i + 1][1]]
            chi_values.append(chi2(arr))
        min_chi = min(chi_values)
        for i in range(num_pair - 1, -1, -1):
            if chi_values[i] == min_chi:
                log_tuple[i] = combine(log_tuple[i], log_tuple[i + 1])
                log_tuple[i + 1] = 'Merged'
        while 'Merged' in log_tuple:
            log_tuple.remove('Merged')
        num_interval = len(log_tuple)
    split_points = [record[0] for record in log_tuple]
    return split_points


def discrete(path):
    # 读取数据文件并进行离散化处理
    Instances = read(path)
    max_interval = 6
    num_log = 4
    for i in range(num_log):
        # 对数据进行收集、合并和离散化处理
        log_tuple = collect(Instances, i)
        split_points = ChiMerge(log_tuple, max_interval)
        print(split_points)


import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 主程序入口，调用discrete函数处理iris.data文件
    discrete('data/iris.data')
    data = Instances = read('data/iris.data')

    data = list(zip(*data))

    for i in range(4):
        sorted_data = sorted(data[i], key=lambda x: float(x))

        # 绘制直方图
        plt.hist(sorted_data, bins=10, color='skyblue', edgecolor='black')

        # 添加标题和标签
        plt.title(f'Histogram of Attribute { i }')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.xticks(rotation=45)

        # 显示图形
        plt.show()
