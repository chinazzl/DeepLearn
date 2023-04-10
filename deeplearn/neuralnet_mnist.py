# coding = utf-8
# 神经网格的推理处理
"""
    神经网络的输入层有 784 个神经元，输出层有 10 个神经元。
    输入层的 784 这个数字来源于图像大小的 28 × 28 = 784，
    输出层的 10 这个数字来源于 10 类别分类（数字 0 到 9，共 10 类别）。
    此外，这个神经网络有 2 个隐藏层，第 1 个隐藏层有 50 个神经元，第 2 个隐藏层有 100 个神经元。
    这个 50 和 100 可以设置为任何值。
"""
import pickle

import numpy as np

from dataset.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("./dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
        return network


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 计算神经元
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    exp_sum_a = np.sum(exp_a)
    y = exp_a / exp_sum_a
    return y


if __name__ == '__main__':
    # x,t = get_data()
    # network = init_network()
    # accuracy_cnt = 0
    # for i in range(len(x)):
    #     y = predict(network, x[i])
    #     # 将获取被赋给参数 x 的数组中的最大值元素的索引
    #     p = np.argmax(y)
    #     if p == t[i]:
    #         accuracy_cnt += 1
    # print("Accuracy: ", str(float(accuracy_cnt / len(x))))
    '''
        批处理
        x,t = get_data()
        batch_size = 100
        for i in range(0, len(x), batch_size):
            x_batch = x[i: batch_size]
            y_batch = predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy += np.sum(p == t[i: i+batch_size])
    '''
    # =========================输出神经网络的各层权重情况========================================
    x, _ = get_data()
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    print("x shape: " + str(x.shape))
    print("x[0] shape: " + str(x.shape))
    print("W1 shape: " + str(W1.shape))
    print("W2 shape: " + str(W2.shape))
    print("W3 shape: " + str(W3.shape))

