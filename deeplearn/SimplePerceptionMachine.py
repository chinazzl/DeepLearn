# 感知机

import numpy as np


class Perception:
    """
        简单构造
    """

    def simpleEx(x1, x2):
        # 设置权重和阈值
        w1, w2, theta = 0.5, 0.5, 0.7
        tmp = x1 * w1 + x2 * w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
            return 1

    '''
    使用np编写简单的感知机
    '''

    def simpleUseNp(self):
        x = np.array([0, 1])
        w = np.array([0.5, 0.5])
        b = -0.7
        xmulw = np.sum(x * w)
        print(xmulw)
        out = b + xmulw
        return 0 if out < 0 else 1
