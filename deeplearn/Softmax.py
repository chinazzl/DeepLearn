"""
    恒等函数
    输出层设计，回归问题用恒等函数，分类问题使用softmax函数
    回归问题：根据某个输入预测一个（连续）数值问题
    分类问题：属于哪一个类别的问题，例如区分图像中的人时男性还是女性的问题

    yk = exp(ak)/sum(exp(a))
    exp(x)表示e^x指数函数，表示假设输出层共有n个神经元，计算第k个神经元的输出yk
"""
import numpy as np


class Softmax:

    def softmax(self, a):
        '''
        需要增加一个常数c 来避免计算时因为数据太大导致计算错误的问题
        '''
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y


if __name__ == '__main__':
    softm = Softmax()
    a = np.array([0.2, 2.9, 4.0])
    y = softm.softmax(a)
    '''
    比如，上面的例子可以解释成 y[0] 的概率是 0.018（1.8 %），y[1] 的概率是 0.245（24.5 %），y[2] 的概率是 0.737（73.7 %）。
    从概率的结果来看，可以说“因为第 2 个元素的概率最高，所以答案是第 2 个类别”。
    而且，还可以回答“有 74 % 的概率是第 2 个类别，有 25 % 的概率是第 1 个类别，有 1 % 的概率是第 0 个类别”。
    也就是说，通过使用 softmax 函数，我们可以用概率的（统计的）方法处理问题。
    '''
    print(y)
    print(np.sum(y))
