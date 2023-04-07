"""
    激活函数：h(x)函数会将输入信号的总和转换为输出信号
    h(x) = {
        0 (x <= 0)
        1 (x > 1)
    }
"""
import numpy as np
import matplotlib.pylab as plt


class ActiveSigmoidClass:

    def pilotSimpleStepFunction(self):
        # 将x转换为Boolean类型，然后通过Boolean类型转为int类型，就变为0和1的结果。
        x = np.arange(-5.0, 5.0, 0.1)
        yd = np.array(x > 0, dtype=np.int64)
        plt.plot(x, yd)
        plt.ylim(-0.1, 1.1)
        plt.show()


print(__name__)
if __name__ == '__main__':

    me = ActiveSigmoidClass()
    me.pilotSimpleStepFunction()
