from bitmex_backtest import Backtest
from numpy import cov
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy.ma as ma
import talib as tb

bt = Backtest(test=True)
filepath = "anyOHLCV.csv"

bt.read_csv(filepath)

hlc3 = (bt.H + bt.L + bt.C) / 3
df = pd.DataFrame(hlc3)
for i in range(1, hlc3.size):
    if hlc3.iloc[i] == hlc3.iloc[i-1]:
        hlc3.iloc[i] += 0.1

weights = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
sum_weights = np.sum(weights)

swma = bt.df['weighted_ma'] = (hlc3
                               .rolling(window=4, center=False)
                               .apply(lambda x: np.sum(weights * x) / sum_weights, raw=False)
                               )

# NEW VARS
src = hlc3

len_reg1 = 5
len_reg2 = 4

maLength1 = 3
maLength2 = 4

trendParam = -1.3
trendParam2 = -1.5
maxLength = 100

per = 38
alphaPclma = 1


def sma(src, m):
    coef = np.ones(m) / m
    return np.convolve(src, coef, mode="valid")


def stdev(src, m):
    a = sma(src * src, m)
    b = np.power(sma(src, m), 2)
    return np.sqrt(a - b)


def correlation(x, y, m):
    cov = sma(x * y, m) - sma(x, m) * sma(y, m)
    den = stdev(x, m) * stdev(y, m)
    return cov / den


def linreg(x, y, m):
    x_ = tb.EMA(x, timeperiod=m)
    y_ = tb.EMA(y, timeperiod=m)
    mx = stdev(x, m)
    my = stdev(y, m)
    c = correlation(x, y, m)
    slope = c * (my / mx)
    for i in range(m - 1):
        slope = np.insert(slope, 0, np.nan)

    inter = y_ - slope * x_
    reg = x * slope + inter
    return reg


def Bama(_src, length, trendParam0, maxLength):
    mom = abs(_src.diff(periods=length))
    volatility = abs(_src.diff(periods=1))
    volatility = np.convolve(volatility, np.ones(length, dtype=int), 'valid')

    for i in range(2):
        volatility = np.insert(volatility, 0, np.nan)
    er = mom / volatility

    for i in range(3):
        er[i] = 0

    ver = pow(er - ((2 * er) - 1) / 2 * (1 - trendParam0) + 0.5, 2)
    vlength = (length - ver + 1) / ver
    for i in range(0, vlength.size):
        if vlength.iloc[i] > maxLength:
            vlength.iloc[i] = maxLength
        else:
            vlength.iloc[i] = vlength.iloc[i]
    valpha = 2 / (vlength + 1)
    bama = _src
    src1 = _src
    for i in range(1, bama.size):
        if src1.iloc[i-1] == np.nan:
            bama.iloc[i] = valpha.iloc[i] * _src.iloc[i] + (1 - valpha.iloc[i]) * src1.iloc[i]
        else:
            bama.iloc[i] = valpha.iloc[i] * _src.iloc[i] + (1 - valpha.iloc[i]) * src1.iloc[i-1]
    return bama


def Pclma(src, length, per):
    sum = 0
    sumw = 0
    l = 0
    for i in range(4):
        l += 1
        w = l - per / 100 * length
        sumw += w
        sum += src.shift(length - i) * w

    return sum / sumw


bar_index = []
for i in range(0, 849802):
    bar_index.append(i)
bar_index = np.array(bar_index, dtype='double')

ma1 = Bama(src, maLength1, trendParam, maxLength)
ma1.to_csv("~/PycharmProjects/zip/venv/ma.csv")
#ma1 = pd.read_csv('ma.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
ma2 = Pclma(swma, maLength2, per).shift(-1)
ma3 = Bama(src, maLength1, trendParam2, maxLength)
ma4 = Pclma(swma, maLength2, per).shift(-1)

maAvg1 = (ma1 + ma2) / 2
maAvg2 = (ma3 + ma4) / 2

trend_reg = linreg(bar_index, maAvg1, 5)
trend_reg2 = linreg(bar_index, trend_reg, 4)
trend_reg3 = linreg(bar_index, maAvg2, 5)
trend_reg4 = linreg(bar_index, trend_reg, 4)


bt.buy_entry = bt.sell_exit = trend_reg3 > trend_reg4.shift()
bt.sell_entry = bt.buy_exit = trend_reg < trend_reg2.shift()

# RUN THE BACKTEST
print(bt.run())
bt.plot("backtest.png")
