# numpy learn
import numpy as np


class NumpyStu:

    """
        矩阵的乘积
        [[1,2], [3,4]] * [[1,2,3],[4,5,6]] => {
            [
            [(1*1+2*4),(1*2+2*5),(1*3+2*6)],
            [(3*1+4*4),(3*2+4*5),(3*3+4*6)]
            ]
        }

        """
    def matrixMultiplication(self, x, y):
        return np.dot(x, y)


if __name__ == '__main__':
    ns = NumpyStu()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x shape: ", x.shape)
    y = np.array([[1, 2], [3, 4], [5, 6]])
    print("y shape: ", y.shape)
    mm = ns.matrixMultiplication(x, y)
    print(mm)
