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

    def ha(self):
        arr = np.array([[5, 2], [4, 5], [1, 2], [3, 4]])
        '''
            flattern 将二维数组转换成一维数组
        '''
        print(repr(arr.shape) + repr(arr.flatten()))

    def multiArr(self, x):
        print("获取维度：" + repr(np.ndim(x)))
        # 返回一个元组 （元素个数，每一个元素内部的数据个数）
        print("数组的形状为：" + repr(np.shape(x)))


if __name__ == '__main__':
    ns = NumpyStu()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x shape: ", x.shape)
    y = np.array([[1, 2], [3, 4], [5, 6]])
    print("y shape: ", y.shape)
    mm = ns.matrixMultiplication(x, y)
    print(mm)
