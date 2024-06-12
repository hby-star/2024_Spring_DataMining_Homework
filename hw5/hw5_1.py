import math
import matplotlib.pyplot as plt


def compute_p_a(k):
    p = math.factorial(k) / math.pow(k, k)
    return p


def hw5_1_a():
    # 求 k = 2,3,4,...,100 时的 p 值，并绘制折线图
    p_values = []
    for k in range(2, 101):
        p = compute_p_a(k)
        p_values.append(p)

    plt.plot(range(2, 101), p_values)
    plt.xlabel('k')
    plt.ylabel('p')
    plt.title('p value with respect to k')
    plt.show()


if __name__ == '__main__':
    hw5_1_a()
