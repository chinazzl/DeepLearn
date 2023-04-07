# coding=utf-8
import itertools

import pandas as pd
import matplotlib

matplotlib.use("TkAGG")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# 指定图形的风格
plt.style.use('fivethirtyeight')

# 内置测试数据
# data = sm.datasets.co2.load_pandas().data
# data = data['co2'].resample('MS').mean()
# data = data.fillna(data.bfill())
# print(data.head())
# data.plot(figsize=(12, 6))
# plt.show()
# print(data.shape)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取本地数据
data1 = pd.read_csv("resources/202.245.csv")
i = 500
newdata = data1["cpu"][:4000]
# 创建一个列表
# newdata.index = range(len(newdata))
# time = pd.date_range(start='2023-02-10', freq='MS', periods=len(newdata))
# 获取时间序列
time = data1["time"][:4000]
newdata.index = time

newdata.plot(figsize=(12, 6))

# data1.plot(x='time', y='cpu', figsize=(12, 6))
plt.title("测试过程数据趋势变化曲线")
plt.xlabel("时间")
plt.ylabel("指标大小")
plt.show()

# 2. 下面对非平稳时间序列进行时间序列的差分，找出适合的差分次数d的值
fig = plt.figure(figsize=(12, 8))
# 创建一个1 * 1 的网格，并在第一位创建一个axes，
ax1 = fig.add_subplot(111)
# 将数据向下移动一位（第一位NAN，第二位变为源数据的第一位）并与原数据的差异值
diff1 = newdata.diff(1)
print('第一阶差分变化曲线>> ' + repr(diff1))
diff1.plot(ax=ax1)
plt.title("测试过程数据第一阶差分变化曲线")
plt.xlabel("时间")
plt.ylabel("指标大小")
# plt.show()
# 这里是做了1阶差分，可以看出时间序列的均值和方差基本平稳，不过还是可以比较一下二阶差分的效果：

# 这里进行二阶差分
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
diff2 = newdata.diff(2)
print("第二阶差分变化曲线>>> " + repr(diff2))
diff2.plot(ax=ax2)
plt.title("测试过程数据第二阶差分变化曲线")
plt.xlabel("时间")
plt.ylabel("指标大小")
# plt.show()

# 这里我们使用一阶差分的时间序列
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
""" 
    使用组合迭代器，包括排列、笛卡尔积、离散元素的选择
"""
# Generate all different combinations of p, q and q triplets
# 笛卡尔积组合，并将组合后的元素转换为列表
pdq = list(itertools.product(p, d, q))
print('pdq >>> ' + repr(pdq))
print("================================================")
# Generate all different combinations of seasonal p, q and q triplets
'''
有三个不同的整数（p,d,q）用于参数化ARIMA模型。可以用符号ARIMA(p,d,q)。这三个参数共计数据集中的季节性、趋势和噪音
    * p是模型的自回归部分。它允许我们将过去价值观的影响纳入我们的模型。直观地说，这将是类似的，表示如果过去三天已经变暖，明天可能会变暖
    * d是模型的集成部分。这包括模型中包含差异量（即从当前值减去的过去时间点的数量）以适用于时间序列的术语。直观地说，这将类似于说如果过去三天的温度差异非常小，
       明天可能会有相同的温度。
    * q是模型的移动平均部分。这允许我们将模型的误差设置为过去以前时间点观察到的误差的线性组合。

'''
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('seasonal_pdq' + repr(seasonal_pdq))
'''
order ( iterable or iterable of iterables , optional ) -- 模型的 (p,d,q) 顺序，用于 AR 参数、差异和 MA 参数的数量。 d 必须是一个整数，表示过程的积分顺序，
    而 p 和 q 可以是一个整数，表示 AR 和 MA 顺序（以便包括所有滞后于这些顺序）或给出特定 AR 和/或 MA 的迭代滞后包括在内。默认为 AR(1) 模型：(1,0,0)。
seasonal_order ( iterable , optional ) – 模型季节性分量的 (P,D,Q,s) 顺序，用于 AR 参数、差异、MA 参数和周期性。 d 必须是一个整数，表示过程的积分顺序，
    而 p 和 q 可以是一个整数，表示 AR 和 MA 顺序（以便包括所有滞后于这些顺序）或给出特定 AR 和/或 MA 的迭代滞后包括在内。 s 是一个整数，给出周期性（季节中的周期数），
    对于季度数据通常为 4，对于月度数据通常为 12。默认为无季节性影响。
enforce_stationarity ( boolean , optional ) -- 是否转换 AR 参数以在模型的自回归组件中强制执行平稳性。默认为真。
enforce_invertibility ( boolean , optional ) -- 是否转换 MA 参数以在模型的移动平均组件中强制执行可逆性。默认为真。

我们现在可以使用上面定义的参数三元组来自动化不同组合对ARIMA模型进行培训和评估的过程。 在统计和机器学习中，这个过程被称为模型选择的网格搜索（或超参数优化）。

在评估和比较配备不同参数的统计模型时，可以根据数据的适合性或准确预测未来数据点的能力，对每个参数进行排序。 我们将使用AIC （Akaike信息标准）值，
该值通过使用statsmodels安装的ARIMA型号方便地返回。 AIC衡量模型如何适应数据，同时考虑到模型的整体复杂性。 在使用大量功能的情况下，
适合数据的模型将被赋予比使用较少特征以获得相同的适合度的模型更大的AIC得分。 因此，我们有兴趣找到产生最低AIC值的模型。
确定order seasonal_order的数据：
e.g
 下面的代码块通过参数的组合来迭代，并使用SARIMAX函数来适应相应的季节性ARIMA模型。 这里， order参数指定(p, d, q)参数，而seasonal_order参数指定季节性ARIMA模型的(P, D, Q, S)季节分量。 
 在安装每个SARIMAX()模型后，代码打印出其各自的AIC得分

<code>
warnings.filterwarnings("ignore") # specify to ignore warning messages
 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
 
            results = mod.fit()
 
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
</code>
SARIMAX(0, 0, 0)x(0, 0, 1, 12) - AIC:6787.3436240402125
SARIMAX(0, 0, 0)x(1, 0, 1, 12) - AIC:1056.2878315690562
SARIMAX(0, 0, 0)x(1, 1, 0, 12) - AIC:1361.6578978064144
SARIMAX(0, 0, 0)x(1, 1, 1, 12) - AIC:1044.7647912940095
...
...
...
SARIMAX(1, 1, 1)x(1, 1, 0, 12) - AIC:444.12436865161305
SARIMAX(1, 1, 1)x(1, 1, 1, 12) - AIC:277.7801413828764

我们的代码的输出表明， SARIMAX(1, 1, 1)x(1, 1, 1, 12)产生最低的AIC值为277.78。 因此，我们认为这是我们考虑过的所有模型中的最佳选择
'''
model = sm.tsa.statespace.SARIMAX(newdata, order=(1, 2, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False,
                                  enforce_invertibility=False)
# 通过卡尔曼滤波器的最大似然来拟合模型。
results = model.fit()
print('通过卡尔曼滤波器的最大似然来拟合模型。' + repr(results))
print(results.summary().tables[1])
# 一个内生变量的标准化残差的诊断图
'''
在适合季节性ARIMA模型（以及任何其他模型）的情况下，运行模型诊断是非常重要的，以确保没有违反模型的假设。 
plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为。

    左上：残差图 残差（residual）是因变量的观测值y_{i}与根据估计的回归方程求出的预测 \hat{y}_{i} 之差，用e表示。
         反映了用估计的回归方程去预测y_{i}而引起的误差。第i个观察值的残差为： e_{i}=y_{i}-\hat{y}_{i}
    常用残差图：有关x残差图，有关\hat{y}的残差图，标准化残差图
               有关x残差图：用横轴表示自变量x的值，纵轴表示对应残差 e_{i}=y_{i}-\hat{y}_{i}，每个x的值与对应的残差用图上的一个点来表示。
               分析残差图，首先考察残差图的形态及其反映的信息。
    右上：核密度直方图
    
    （1）右上角的红色KDE线和黄色N(0,1)线非常的接近，它表明残差是服从正态分布的。

    （2）左下角的QQ分位图表明，从N(0,1)的标准正态分布抽取的样本，残差(蓝点)的有序分布服从线性趋势。 同样，这是一个强烈的迹象表明，残差是正态分布。

    （3）随着时间的推移（左上图）残差不会显示任何明显的季节性，似乎是白噪声。这通过右下角的自相关（即相关图）证实了这一点，
         它表明时间序列的残差与其自身的滞后版本有很低的相关性。
'''
results.plot_diagnostics(figsize=(12, 12))
plt.show()

'''
    样本内预测和样本外预测。从2501开始进行校验
    
    我们已经获得了我们时间序列的模型，现在可以用来产生预测。 我们首先将预测值与时间序列的实际值进行比较，这将有助于我们了解我们的预测的准确性。 
    get_prediction()和conf_int()属性允许我们获得时间序列预测的值和相关的置信区间。
    
    e.g
    上述规定需要从1998年1月开始进行预测。dynamic=False参数确保我们产生一步前进的预测，这意味着每个点的预测都将使用到此为止的完整历史生成。
    我们可以绘制二氧化碳时间序列的实际值和预测值，以评估我们做得如何。 注意我们如何在时间序列的末尾放大日期索引。
'''
pred = results.get_prediction(start=newdata.index[3500], dynamic=False)
'''
alpha ( float , optional ) -- 置信区间的显着性水平。即，默认 alpha = .05 返回 95% 的置信区间。
cols ( array-like , optional ) -- cols 指定要返回的置信区间
方法(字符串) -- 尚未实现 估计 confidence_interval 的方法。“默认”：使用基于逆 Hessian 的 self.bse，用于 MLE “hjjh”：“jac”：“boot-bse”“boot_quant”“profile”

'''
pred_ci = pred.conf_int()
print('pred_ci 置信度：')
print(pred_ci.head(3))
print("--------------------------------")
# 置信区间的数据
print(pred_ci.iloc[:, 0])
print('================================')
print(pred_ci.iloc[:, 1])
# 返回拟合参数的置信区间。
ax = newdata.plot(label='Observed', figsize=(12, 6))
# predicted_mean 是与索引(例如日期索引)相关联的pandas Series对象。
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
'''
matplotlib.pyplot.fill_between( x , y1 , y2=0 , where=None , interpolate=False , step=None , * , data=None , **kwargs )
    fill_between 填充两条水平曲线之间的区域。
    x数组（长度 N）
        定义曲线的节点的X坐标，覆盖的区域，我直接复制为x，表示整个x都覆盖。
    
    y1数组（长度 N）或标量
        定义第一条曲线的节点的Y坐标。
    
    y2数组（长度 N）或标量，默认值：0
        定义第二条曲线的节点的Y坐标。
        
    alpha: 覆盖区域的透明度[0,1],其值越大，表示越不透明
    iloc: [a,b,c] a：第一行索引起始位置（起始位置可以查到），b：需要查询的数据行结尾位置（只能查到前一位），c：取得是哪一个位置的数据
'''
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('时间')
ax.set_ylabel('指标大小')
plt.title("过程测试ARIMA时间序列预测模型验证")
plt.legend()
plt.show()

# 直接输出相应的MSE,比较对应函数的趋势分析
'''
    量化我们的预测的准确性也是有用的。 我们将使用MSE（均方误差），它总结了我们的预测的平均误差。 对于每个预测值，我们计算其到真实值的距离并对结果求平方。 
    结果需要平方，以便当我们计算总体平均值时，正/负差异不会相互抵消。
'''
data_forecasted = pred.predicted_mean
data_truth = newdata[3000:]
print(data_forecasted, len(data_forecasted))
print(data_truth, len(data_truth))
plt.plot(range(len(data_forecasted) - 1), data_forecasted[1:], "r")
plt.plot(range(len(data_truth)), data_truth, "b")
plt.show()

# Compute the mean square error
mse = ((data_forecasted - data_truth) ** 2).mean()
# mse 的值越小越好。
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Get forecast 500 steps ahead in future
# model = sm.tsa.statespace.SARIMAX(newdata, order=(1,2,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
# results = model.fit()
pred_uc = results.get_forecast(steps=25)
print(pred_uc.predicted_mean)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

plt.figure()
plt.plot(newdata)
plt.plot(pred_uc.predicted_mean)
plt.show()

ax = newdata.plot(label='Observed', figsize=(12, 6))

# plt.show()
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('时间')
ax.set_ylabel('指标大小')
plt.title("过程测试ARIMA模型预测未来三天指标变化趋势")
plt.legend()
plt.show()
