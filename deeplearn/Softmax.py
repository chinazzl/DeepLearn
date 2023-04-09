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
    a = np.array([1010, 1000, 990])
    print(softm.softmax(a))