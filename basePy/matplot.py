# matplotlib version 3.5.3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 初始化matplot
def init_matplotlib():
    xpoints = np.array([0, 6])
    ypoints = np.array([0, 100])
    xpoints1 = np.array([3, 9])
    ypoints2 = np.array([20, 100])
    plt.plot(xpoints, ypoints, xpoints1, ypoints2)
    plt.show()


# pandas的 plot
def pandas_plot_init():
    df = pd.DataFrame(np.random.randn(9, 4), index=pd.date_range('2/1/2023', periods=9), columns=list('abcd'))
    print(df)
    df.plot()
    plt.show()


pandas_plot_init()
