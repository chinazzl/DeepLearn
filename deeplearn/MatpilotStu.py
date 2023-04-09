import numpy as np;
import matplotlib.pyplot as plt


class MatpilotStu:
    """
        阶跃函数
    """

    def stepFunction(self, x):
        return np.array(x > 0, dtype=np.int32)

    """
        sigmoid函数，非线性函数，y = 1/(1 + exp(-x))
    """

    def sigmoid(self, x):
        return 1 + np.exp(-x)

    """
        ReLU函数：在输入大于0时，直接输出该值；在输入小于等于0时，输出0
        h(x) {
            x > 0  x
            x <= 0 0
        }
    """

    def reluFunction(self, x):
        return np.maximum(0, x)


if __name__ == "__main__":
    mp = MatpilotStu()
    x = np.arange(-5., 5., .1)
    y = mp.stepFunction(x)
    plt.plot(x, y, linestyle='--', label="step")

    sy = 1 / mp.sigmoid(x)
    plt.plot(x, sy, label="sigmoid")
    plt.show()
