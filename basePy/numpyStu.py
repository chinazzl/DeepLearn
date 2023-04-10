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
        '''
            维度例如：
            一维数组 1
            二维数组 2
        :param x:
        :return:
        '''
        print("获取维度：" + repr(np.ndim(x)))
        # 返回一个元组 （元素个数，每一个元素内部的数据个数）
        print("数组的形状为：" + repr(np.shape(x)))

    def test_argmax(self, x, axis):
        # 按照指定维度axis也就是n维数组， 进行获取最大值的索引。
        return np.argmax(x, axis=axis)


if __name__ == '__main__':
    ns = NumpyStu()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x shape: ", x.shape)
    y = np.array([[1, 2], [3, 4], [5, 6]])
    print("y shape: ", y.shape)
    mm = ns.matrixMultiplication(x, y)
    print(mm)
    ns.multiArr(x)
    # [[0,0], [1,1]]
    arr = np.array([[[3, 2, 1], [3, 2, 1]], [[4, 6, 5], [4, 6, 5]]])
    argmaxIndex = ns.test_argmax(arr, 3)
    print('arr argmax index is ', str(argmaxIndex))
