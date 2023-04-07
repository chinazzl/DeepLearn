# 神经元网路
import numpy as np


class NeuronNetwork:

    def sigmoidFunction(self, x):
        return 1 / (1 + np.exp(-x))

    def init_network(self):
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network['b1'] = np.array([0.1, 0.2, 0.3])
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network['b2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network['b3'] = np.array([0.1, 0.2])
        return network

    def forward(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoidFunction(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoidFunction(a2)
        a3 = np.dot(z2, W3) + b3
        y = a3
        return y


if __name__ == '__main__':
    """
    从入口计算下一个神经元
        neu = NeuronNetwork()
        x = np.array([1.0, 0.5])
        W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        print("w1 shaper:", W1.shape)
        print("x shaper:", x.shape)
        B1 = np.array([0.1, 0.2, 0.3])
        # 下一个神经元
        A1 = np.dot(x, W1) + B1
        print("A1 value: ", A1)
        # 调用激活函数
        Z1 = neu.sigmoidFunction(A1)
        print("Z1 sigmoid：", Z1)
    """
    '''
        这里定义了 init_network() 和 forward() 函数。init_network() 函数会进行权重和偏置的初始化，
        并将它们保存在字典变量 network 中。这个字典变量 network 中保存了每一层所需的参数（权重和偏置）。
        forward() 函数中则封装了将输入信号转换为输出信号的处理过程。
        另外，这里出现了 forward（前向）一词，它表示的是从输入到输出方向的传递处理。后面在进行神经网络的训练时，我们将介绍后向（backward，从输出到输入方向）的处理。
        至此，神经网络的前向处理的实现就完成了。通过巧妙地使用 NumPy 多维数组，我们高效地实现了神经网络。    
    '''
    neu = NeuronNetwork()
    network = neu.init_network()
    x = np.array([1.0, 0.5])
    y = neu.forward(network, x)
    print("neuron y = ", y)
