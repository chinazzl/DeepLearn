import unittest
import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def testArr(self):
        data = pd.read_csv("resources/data.csv");
        # print(data)
        # d = data['cpu'][:3];
        # print(d)
        # s = d.shift()
        # print(s)
        # s1 = d - s
        # print(s1)
        # df = d.diff()
        # print(df)
        v = data.values
        print(v)
        print("================================================")
        y = 365
        d = list()
        for i in range(1,3):
            vm = v[i+1] - v[i]
            d.append(vm)
        print(d)

    def testInnerMethod(self):
        r = range(0, 2)
        for i in r:
            print(i)

    def testCpu(self):
        data = pd.read_csv("resources/cpu_data_fake.csv")
        newdata = data["cpu"][:4000]
        time = data['time'][:4000]
        # print(newdata.index)
        print("================================================")
        t = time.DatetimeIndex(time.index).to_period('M')
        print(t)

    def test_str(self):
        seasonal_pdq = "123"
        print('seasonal_pdq' + repr(seasonal_pdq))


if __name__ == '__main__':
    unittest.main()
