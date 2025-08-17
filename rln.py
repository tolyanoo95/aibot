import json
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from stocktrends import indicators
from renko import Renko


# Set Pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Adjust the width as needed

def calculate_profit_percent(entry_price, exit_price, position_type):
    if entry_price == 0:
        return 0

    if position_type == 'BUY':
        net_profit = exit_price - entry_price
    elif position_type == 'SELL':
        net_profit = entry_price - exit_price
    else:
        raise ValueError(f"Unknown position type: {position_type}")


    profit_percent = (net_profit / entry_price) * 100
    return profit_percent - 0.2


def get_binance_data(symbol, time = '5m', start_date_str=None, end_date_str=None):
    if start_date_str is None:
        start_date_str = "2024-03-10 00:00"

    start_time = int(datetime.strptime(start_date_str, "%Y-%m-%d %H:%M").timestamp() * 1000)

    if end_date_str is None:
        # Уменьшаем end_time на три часа
        end_time = int((datetime.utcnow().timestamp() + 3 * 60 * 60) * 1000)
    else:
        end_time = int(datetime.strptime(end_date_str, "%Y-%m-%d %H:%M").timestamp() * 1000)

    if start_time >= end_time:
        print(
            "Warning: start_time is greater than or equal to end_time. Adjusting the start_time to match the end_time.")
        start_time = end_time - (5 * 60 * 1000)

    closing_prices = []
    while start_time < end_time:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={time}&limit=1500&startTime={start_time}&endTime={end_time}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data:
                closing_prices.extend([(int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), int(candle[6]), int(candle[6])) for candle in data])
                start_time = int(data[-1][0]) + (5 * 60 * 1000)
            else:
                break
        else:
            print(f"Error: Unable to get data. {response.content}")
            break

    return closing_prices if closing_prices else None


symbol = 'BTCUSDT'
data_15 = get_binance_data(symbol=symbol, time='15m', start_date_str="2024-03-01 00:00")
data_30 = get_binance_data(symbol=symbol, time='1h', start_date_str="2024-03-01 00:00")


#del data_30[-1]

df_15 = pd.DataFrame(data_15, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
df_30 = pd.DataFrame(data_30, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])

df_15['date_time'] = pd.to_datetime(df_15['close_time'], unit='ms') + pd.Timedelta(hours=3)
df_30['date_time'] = pd.to_datetime(df_30['close_time'], unit='ms') + pd.Timedelta(hours=3)

df_15['date'] = df_15['close_time'].astype(int)
df_30['date'] = df_30['close_time'].astype(int)

#print(ta.supertrend(close=data_15['close'], high=data_15['high'], low=data_15['low'], multiplier=3))
renko = indicators.Renko(df_15)
renko.brick_size = 200
renko.chart_type = indicators.Renko.PERIOD_CLOSE
data_15 = renko.get_ohlc_data()
data_15['date_time'] = pd.to_datetime(data_15['date'], unit='ms') + pd.Timedelta(hours=3)
data_15 = data_15.drop_duplicates(subset='date_time', keep='last')
data_15 = data_15.merge(df_15[['date', 'close']], on='date', how='left', suffixes=('', '_30'))


renko = indicators.Renko(df_30)
renko.brick_size = 200
renko.chart_type = indicators.Renko.PERIOD_CLOSE
data_30 = renko.get_ohlc_data()
data_30['date_time'] = pd.to_datetime(data_30['date'], unit='ms') + pd.Timedelta(hours=3)
data_30 = data_30.drop_duplicates(subset='date_time', keep='last')
data_30 = data_30.merge(df_30[['date', 'close']], on='date', how='left', suffixes=('', '_30'))

data_15['SP'] = ta.ema(close=data_15['close'], length=50)
data_30['SP'] = ta.ema(close=data_30['close'], length=50)

data_15 = data_15.merge(data_30[['date', 'SP']], on='date', how='left', suffixes=('', '_30'))
#data_15['SP_30'] = data_15['SP_30'].fillna(method='ffill')


print(data_15)

#############

sum = 0
depo = 2000
date = ''
buy_long = 0
check_buy_long = False
n = False


buy_short = 0
check_buy_short = False

for i in range(0, len(data_15['close'])):
    if i > 0:
        if data_15['SP'][i] > data_15['SP_30'][i] and data_15['close'][i] < data_15['SP'][i] and check_buy_long == False:
            buy_long = data_15['close_30'][i]
            date = data_15['date_time'][i]
            check_buy_long = True
            n = False

        profit = calculate_profit_percent(buy_long, data_15['close_30'][i], 'BUY')

        if (profit > 2 or profit < -8) and check_buy_long == True and n == False:
            print('BUY', date, buy_long, data_15['close'][i], profit)
            sum = sum + (depo * (profit / 100))
            n = True


        if data_15['SP'][i] < data_15['SP_30'][i] and check_buy_long == True:
            check_buy_long = False
            if n == False:
                print('BUY', date, buy_long, data_15['close_30'][i], profit)
                sum = sum + (depo * (profit / 100))


for i in range(0, len(data_15['close'])):
    if i > 0:
        if data_15['SP'][i] < data_15['SP_30'][i] and data_15['close'][i] > data_15['SP'][i] and check_buy_long == False:
            buy_long = data_15['close_30'][i]
            date = data_15['date_time'][i]
            check_buy_long = True
            n = False

        profit = calculate_profit_percent(buy_long, data_15['close_30'][i], 'SELL')

        if (profit > 2 or profit < -8) and check_buy_long == True and n == False:
            print('SELL', date, buy_long, data_15['close'][i], profit)
            sum = sum + (depo * (profit / 100))
            n = True


        if data_15['SP'][i] > data_15['SP_30'][i] and check_buy_long == True:
            check_buy_long = False
            if n == False:
                print('SELL', date, buy_long, data_15['close_30'][i], profit)
                sum = sum + (depo * (profit / 100))


date = ''
buy_long = 0
check_buy_long = False


print(sum)