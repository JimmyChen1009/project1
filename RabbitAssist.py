#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# calculate ten technical indicators
def smaCal(price, period):
    # Simple Moving Average
    sma = pd.Series(np.nan, price.index)
    for i in range(period-1, len(price)):
        sma[i] = np.mean(price[i-period+1:i+1])
    return sma
    
def wmaCal(price, period):
    # Weighted Moving Average
    w = np.arange(1, period+1)/np.sum(np.arange(1, period+1))
    wma = pd.Series(np.nan, index=price.index)
    for i in range(period-1, len(price)):
        wma[i] = np.sum(w*price[i-period+1: i+1])
    return wma
    
def emaCal(price, period):
    # Exponential Moving Average
    alpha = 2/(1+period)
    ema = pd.Series(np.nan, price.index)
    ema[period-1] = np.mean(price[:period])
    for i in range(period, len(price)):
        ema[i] = alpha*price[i]+(1-alpha)*ema[i-1]
    return ema
    
def momCal(price, period):
    # Momentum
    mom = (price-price.shift(period))
    return mom
def rsiCal(price, period):
    # Relative Strength Index (RSI)
    closeDif = price - price.shift(1)
    upDif = pd.Series(0, index=closeDif.index)
    upDif[closeDif>0] = closeDif[closeDif>0]
    downDif = pd.Series(0, index=closeDif.index)
    downDif[closeDif<0] = closeDif[closeDif<0]
    rsi_df = pd.concat([price, closeDif, upDif, downDif], axis=1)
    rsi_df.columns = ['Close', 'closeChange', 'upPrc', 'downPrc']
    rsi = pd.Series(np.nan, index=price.index)
    for i in range(period-1, len(rsi_df)):
        avgUp = np.mean(rsi_df['upPrc'][i-period+1:i+1])
        avgDown = -np.mean(rsi_df['downPrc'][i-period+1:i+1])
        rsi[i] = 100-(100/(1+avgUp/avgDown))
    return rsi
    
def lwrCal(close, high, low, period):
    # Larry William's R%
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    lwr = (close-hh) / (hh-ll)*100
    return lwr
    
def adCal(close, high, low):
    # A/D Oscillators
    ad = pd.Series(np.nan, index=close.index)
    for i in range(1, len(close)):
        ad[i] = (high[i]-close[i-1])/(high[i]-low[i])
    return ad
    
def kdCal(close, high, low, period):
    # Stochastic KD
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    stoK = (close-ll)/(hh-ll)*100
    stoD = smaCal(stoK, period)
    return stoK, stoD
    
def macdCal(close, period):
    # Moving Averaging Convergence Divergence
    dif = emaCal(close, 12) - emaCal(close, 26)
    macd = emaCal(dif.dropna(), period)
    return macd

def cciCal(close, high, low, period):
    # Commodity Channel Index
    m = (high+low+close)/3
    sm = smaCal(m, period)
    d = pd.Series(np.nan, index=close.index)
    for i in range(period-1, len(close)):
        d[i] = np.mean((m[i-9:i+1]-sm[i]).abs())
    cci = (m-sm) / (0.015*d)
    return cci


# In[3]:


def toIndicators(df, period):
    # convert dataframe to ten indicators
    close, high, low = df['Close'], df['High'], df['Low']
    df['sma'] = smaCal(close, period)
    df['wma'] = wmaCal(close, period)
    df['mom'] = momCal(close, period)
    df['stoK'], df['stoD'] = kdCal(close, high, low, period)
    df['rsi'] = rsiCal(close, period)
    df['macd'] = macdCal(close, period)
    df['lwR'] = lwrCal(close, high, low, period)
    df['ad'] = adCal(close, high, low)
    df['cci'] = cciCal(close, high, low, period)
    return df.dropna()


# In[4]:


def annualSummary(df):
    # count upward and downward movements per year
    total_arr = df['Return'].resample('Y').count()
    up_arr = df[df['Return']>0]['Return'].resample('Y').count()
    down_arr = df[df['Return']<=0]['Return'].resample('Y').count()
    result = pd.DataFrame(
        {'Increase': up_arr.values,
        'Increase (%)': (up_arr/total_arr).values,
        'Decrease': down_arr.values,
        'Decrease (%)': (down_arr/total_arr).values,
        'Total': total_arr.values},
        index=total_arr.index.year
    )
    return result


# In[5]:


def indicToTrend(df):
    # convert numerical indicators into trend deterministic data (0 or 1)
    indicators_arr = ['sma', 'wma', 'mom', 'stoK', 'stoD', 'rsi', 'macd', 'lwR', 'ad', 'cci']
    new_df = pd.DataFrame(np.nan, index=df.index, columns=indicators_arr)
    for i in range(1, len(df)):
        p_time = df.index[i-1]
        c_time = df.index[i]
        new_df.loc[c_time, 'sma'] = 1 if df.loc[c_time, 'Close'] > df.loc[c_time, 'sma'] else 0
        new_df.loc[c_time, 'wma'] = 1 if df.loc[c_time, 'Close'] > df.loc[c_time, 'wma'] else 0
        new_df.loc[c_time, 'mom'] = 1 if df.loc[c_time, 'mom'] > df.loc[p_time, 'mom'] else 0
        new_df.loc[c_time, 'stoK'] = 1 if df.loc[c_time, 'stoK'] > df.loc[p_time, 'stoK'] else 0
        new_df.loc[c_time, 'stoD'] = 1 if df.loc[c_time, 'stoD'] > df.loc[p_time, 'stoD'] else 0
        new_df.loc[c_time, 'rsi'] = 1 if df.loc[c_time, 'rsi']<30 else 1 if (df.loc[c_time, 'rsi'] > df.loc[p_time, 'rsi']) & (df.loc[c_time, 'rsi']>=30) & (df.loc[c_time, 'rsi']<=70) else 0 
        new_df.loc[c_time, 'macd'] = 1 if df.loc[c_time, 'macd'] > df.loc[p_time, 'macd'] else 0
        new_df.loc[c_time, 'lwR'] = 1 if df.loc[c_time, 'lwR'] > df.loc[p_time, 'lwR'] else 0
        new_df.loc[c_time, 'ad'] = 1 if df.loc[c_time, 'ad'] > df.loc[p_time, 'ad'] else 0
        new_df.loc[c_time, 'cci'] = 1 if df.loc[c_time, 'cci']<-200 else 1 if (df.loc[c_time, 'cci'] > df.loc[p_time, 'cci']) & (df.loc[c_time, 'rsi']>=-200) & (df.loc[c_time, 'rsi']<=200) else 0 
    new_df[['Close', 'Return', 'Movement', 'Future_Movement']] = df[['Close', 'Return', 'Movement', 'Future_Movement']]
    return new_df.dropna()


# In[ ]:




