import pandas as pd

data = pd.read_csv("../resources/cpu_data_fake.csv")


def pd_iloc():
    data.index = data['time']
    # iloc: [a,b,c] a：第一行索引起始位置（起始位置可以查到），b：需要查询的数据行结尾位置（只能查到前一位），c：取得是哪一个位置的数据
    print(data.index)
    print(" tmp: {}".format(data['2022-07-17 15:44:27':]))
    return data.iloc[:2, 2]


def pd_date():
    time = data['time'].head()
    dTime = pd.to_datetime(time)
    print(dTime)
    pt = dTime.to_period('W')
    print(pt)


pd_date()