# coding=utf-8
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math
import calendar
import pyflux as pf

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Image
import matplotlib

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
from matplotlib.pyplot import subplot
from IPython.core.pylabtools import figsize
from sklearn.model_selection._split import train_test_split

plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

# 导入数据
energy = pd.read_csv('data.csv')
# 划分训练数据还是测试数据
# energy[(energy.index < )]
energy.head(3)
print("==========")
dates = pd.date_range(start='2023-02-07', freq='MS', periods=len(energy))
energy['Month'] = dates.month
energy['Month'] = energy['Month'].apply(lambda x: calendar.month_abbr[x])
energy['Year'] = dates.year
# 删除旧的字段
energy.drop(['time'], axis=1, inplace=True)
energy = energy[['Month','Year','usage']]
cpu_usage = energy['usage']
year = energy['Year']

energy.set_index(dates, inplace=True)
print(energy.head(5))
# 进行绘图
plt.figure(figsize=(10,5))
plt.plot(year,cpu_usage)
plt.xlabel('Months', fontsize=12)
plt.ylabel('usage', fontsize=12)
train_test_split = energy[:1500];
model_fit = pf.ARIMA(data=train_test_split,ar=11,ma=11,integ=0,target='usage')
x = model_fit.fit("M-H")
model_fit.plot_fit(figsize=(10,5))
# plt.show()
