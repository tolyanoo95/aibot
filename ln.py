import json
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from stocktrends import indicators

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

sum = 0

for d in range(1, 31):
    print(f'2024-01-{d:02} 00:00')

    if d > 2:

        symbol = 'SOLUSDT'
        data_15 = get_binance_data(symbol=symbol, time='5m', start_date_str=f'2024-01-{d:02} 00:00', end_date_str=f'2024-01-{d:02} 23:59')
        data_30 = get_binance_data(symbol=symbol, time='5m', start_date_str=f'2024-01-{d-1:02} 00:00', end_date_str=f'2024-01-{d-1:02} 23:59')


        #del data_30[-1]

        df_15 = pd.DataFrame(data_15, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
        df_30 = pd.DataFrame(data_30, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])

        df_15['date_time'] = pd.to_datetime(df_15['close_time'], unit='ms') + pd.Timedelta(hours=3)
        df_30['date_time'] = pd.to_datetime(df_30['close_time'], unit='ms') + pd.Timedelta(hours=3)

        df_15['date'] = df_15['close_time'].astype(int)
        df_30['date'] = df_30['close_time'].astype(int)

        lb_15 = indicators.LineBreak(df_15)
        lb_15.line_number = 1
        data_15 = lb_15.get_ohlc_data()

        lb_30 = indicators.LineBreak(df_30)
        lb_30.line_number = 3
        data_30 = lb_30.get_ohlc_data()

        data_15['date_time'] = pd.to_datetime(data_15['date'], unit='ms') + pd.Timedelta(hours=3)

        data_15['SP'] = ta.ema(close=data_15['close'], high=data_15['high'], low=data_15['low'], length=50)
        data_30['SP'] = ta.supertrend(close=data_15['close'], high=data_15['high'], low=data_15['low'], multiplier=3)['SUPERT_7_3.0']

        data_15 = data_15.merge(data_30[['date', 'SP']], on='date', how='left', suffixes=('', '_30'))
        data_15 = data_15.merge(data_30[['date', 'uptrend']], on='date', how='left', suffixes=('', '_30'))


        #data_15['SP_30'] = data_15['SP_30'].fillna(method='ffill')

        print(data_15)



        #############
