from bitmex_backtest import Backtest
import numpy as np
import pandas as pd
import talib as tb

# Required libs + check comments at line 158
#pip3 install bitmex-backtest
#pip3 install ta-lib
#pip3 install pandas
#pip3 install numpy

bt = Backtest(test=True)
filepath = "XBTMINUTES.csv"
bt.read_csv(filepath)

def setSrc():
    hlc3 = (bt.H + bt.L + bt.C) / 3
    for i in range(1, hlc3.size):
        if hlc3.iloc[i] == hlc3.iloc[i-1]:
            hlc3.iloc[i] += 0.01
    return hlc3

def setBarIndex():
    bar_index = []
    for i in range(0, src.size):
        bar_index.append(i)
    return np.array(bar_index, dtype='double')


def symmetricallyWeightedMovingAverage():
    weights = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6])
    sum_weights = np.sum(weights)
    swmaLength = 4
    swma = bt.df['weighted_ma'] = (src
                                   .rolling(window=swmaLength, center=False)
                                   .apply(lambda x: np.sum(weights * x) / sum_weights, raw=False)
                                   )
    return swma


def sma(src, length):
    coef = np.ones(length) / length
    return np.convolve(src, coef, mode="valid")


def stdev(src, length):
    a = sma(src * src, length)
    b = np.power(sma(src, length), 2)
    return np.sqrt(a - b)


def correlation(x, y, length):
    cov = sma(x * y, length) - sma(x, length) * sma(y, length)
    den = stdev(x, length) * stdev(y, length)
    return cov / den


def linreg(x, y, length):
    emaX = tb.EMA(x, timeperiod=length)
    emaY = tb.EMA(y, timeperiod=length)
    deviationX = stdev(x, length)
    deviationY = stdev(y, length)
    correlationXY = correlation(x, y, length)
    slope = correlationXY * (deviationY / deviationX)
    for i in range(length - 1):
        slope = np.insert(slope, 0, np.nan)
    inter = emaY - slope * emaX
    reg = x * slope + inter
    return reg


def setMom(src, length):
    return abs(src.diff(periods=length))


def setVolatility(src, length):
    volatility = abs(src.diff(periods=1))
    volatility = np.convolve(volatility, np.ones(length, dtype=int), 'valid')
    for i in range(2):
        volatility = np.insert(volatility, 0, np.nan)
    return volatility


def setEr(mom, volatility):
    er = mom/volatility
    for i in range(3):
        er[i] = 0
    return er

def setVer(er, trendParam):
    return pow(er - ((2 * er) - 1) / 2 * (1 - trendParam) + 0.5, 2)


def setVlength(ver, length, maxLength):
    vlength = (length - ver + 1) / ver
    for i in range(0, vlength.size):
        if vlength.iloc[i] > maxLength:
            vlength.iloc[i] = maxLength
        else:
            vlength.iloc[i] = vlength.iloc[i]
    return vlength


def setValpha(vlength):
    return 2 / (vlength + 1)


def Bama(src, length, trendParam, maxLength):
    mom = setMom(src, length)
    volatility = setVolatility(src, length)
    er = setEr(mom, volatility)
    ver = setVer(er, trendParam)
    vlength = setVlength(ver, length, maxLength)
    valpha = setValpha(vlength)
    bama = src
    src1 = src
    for i in range(1, src1.size):
        if src1.iloc[i-1] == np.nan:
            bama.iloc[i] = valpha.iloc[i] * src.iloc[i] + (1 - valpha.iloc[i]) * src1.iloc[i]
        else:
            bama.iloc[i] = valpha.iloc[i] * src.iloc[i] + (1 - valpha.iloc[i]) * src1.iloc[i-1]
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


def setTrendReg(barLength, x, y, length):
    return linreg(barLength, (x+y)/2, length)


# VARS
src = setSrc()
bar_index = setBarIndex()
swma = symmetricallyWeightedMovingAverage()

len_reg1 = 5
len_reg2 = 4

maLength1 = 3
maLength2 = 4

trendParam1 = -1.3
trendParam2 = -1.5

maxLength = 100
per = 38

# Run the script once, then comment lines 159, 160 and uncomment line 161, then run it again.
ma1 = Bama(src, maLength1, trendParam1, maxLength)
ma1.to_csv("~/PycharmProjects/CIGL/venv/ma.csv")  # Change the path where you want to save the ma.csv
# ma1 = pd.read_csv('ma.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # Change the path too.
ma2 = Pclma(swma, maLength2, per).shift(-1)
ma3 = Bama(src, maLength1, trendParam2, maxLength)

trend_reg = setTrendReg(bar_index, ma1, ma2, len_reg1)
trend_reg2 = linreg(bar_index, trend_reg, len_reg2)
trend_reg3 = setTrendReg(bar_index, ma3, ma2, len_reg1)

# Backtest logic

bt.buy_entry = bt.sell_exit = trend_reg3 > trend_reg2.shift()
bt.sell_entry = bt.buy_exit = trend_reg < trend_reg2.shift()

# RUN THE BACKTEST
print(bt.run())
bt.plot("backtest.png")
