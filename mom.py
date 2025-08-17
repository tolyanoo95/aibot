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
        return  0

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



#url = "https://fapi.binance.com/fapi/v1/klines?symbol=SOLUSDT&interval=15m&limit=1500"

#response = requests.get(url) #get_binance_data(symbol='C98USDT', start_date_str="2024-03-10 00:00")


sum = 0

for d in range(1, 31):
    print(f'2024-01-{d:02} 00:00')

    if d > 2:

        symbol = 'SOLUSDT'
        data_15 = get_binance_data(symbol=symbol, time='5m', start_date_str=f'2024-04-02 {d:02}:00', end_date_str=f'2024-04-02 {d:02}:59')
        data_30 = get_binance_data(symbol=symbol, time='5m', start_date_str=f'2024-05-09 03:00', end_date_str=f'2024-05-10 02:59')
        data_d = get_binance_data(symbol=symbol, time='5m', start_date_str=f'2023-04-01 00:00',end_date_str=f'2023-04-01 23:59')

        del data_15[-1]
        #del data_30[-1]


        df_15 = pd.DataFrame(data_15, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
        df_30 = pd.DataFrame(data_30, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
        df_d = pd.DataFrame(data_d, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time'])

        #df_15['date'] = df_15['close_time'].astype(int)
        #df_30['date'] = df_30['close_time'].astype(int)

        df_15['date_time'] = pd.to_datetime(df_15['close_time'], unit='ms') + pd.Timedelta(hours=3)


        df_15['SP'] = df_15.ta.supertrend()['SUPERT_7_3.0']
        df_15['SPt'] = df_15.ta.supertrend()['SUPERTd_7_3.0']
       # df_15['RSI'] = df_15.ta.rsi()
       # df_15['EMA'] = df_15.ta.ema(length=10)
       # df_15['ADX'] = df_15.ta.adx()['ADX_14']
       # df_15['PVT'] = df_15.ta.pvt()
       # df_15['EMAP'] = df_15.ta.ema(close=df_15['PVT'])
        #df_30['date'] = df_30['date'] + (3600000*1)

        lb_15 = indicators.LineBreak(df_15)
        lb_15.line_number = 1
        data_15 = lb_15.get_ohlc_data()

       # data_15['EMA'] = ta.ema(close=data_15['close'], high=data_15['high'], low=data_15['low'], length=10)

        vp = df_30.ta.vp()
        #print(vp)
        vpd = df_d.ta.vp()

        print(vp)

        #print(pvt)
        #print(vp)
        #print(len(df_30))


        #print(df_15)




        #with open('/Users/anatolii/Desktop/dd.json', 'r') as f:
            #dd = json.load(f)


        depo = 2000
        date = ''
        buy_long = 0
        check_buy_long = False
        n = False
        operation = False
        h = False
        l = False
        stop_loss = 0
        for i in range(len(df_15['close'])):

            if i > 0:
                for j in range(len(vp['low_close'])):
                    if df_15['close'][i] < vp['mean_close'][0] and df_15['close'][i] > vp['mean_close'][4] and df_15['close'][i] > vpd['mean_close'][4] and check_buy_long == False: #and df_15['RSI'][i] < 30
                        buy_long = df_15['close'][i]
                        date = df_15['date_time'][i]
                        check_buy_long = True
                        n = False
                        operation = 'BUY'
                        h = vp['high_close'][j]

                    if df_15['close'][i] > vp['mean_close'][9] and df_15['close'][i] < vp['mean_close'][4] and df_15['close'][i] < vpd['mean_close'][4] and check_buy_long == False: #and df_15['RSI'][i] > 70
                        buy_long = df_15['close'][i]
                        date = df_15['date_time'][i]
                        check_buy_long = True
                        n = False
                        operation = 'SELL'
                        l = vp['low_close'][j]


                profit = calculate_profit_percent(buy_long, df_15['close'][i], operation)



                if (profit > 0.4 or profit < -2 or len(df_15['close']) - 1 == i) and check_buy_long == True and n == False:  #or str(df_15['date_time'][i]).find("23:54") >= 0
                    check_buy_long = False
                    print(operation, date, buy_long, df_15['close'][i], profit)
                    sum = sum + (depo * (profit / 100))
                    n = True


                '''
                for j in range(len(vp['low_close'])):
                    if df_15['close'][i] < vp['mean_close'][4] and n == True and check_buy_long == True and operation == 'BUY':
                        check_buy_long = False
                        if n == False:
                            print(operation, date, buy_long, df_15['close'][i], profit)
                            sum = sum + (depo * (profit / 100))

                    if df_15['close'][i] > vp['mean_close'][4] and n == True and check_buy_long == True and operation == 'SELL':
                        check_buy_long = False
                        if n == False:
                            print(operation, date, buy_long, df_15['close'][i], profit)
                          sum = sum + (depo * (profit / 100))
                '''
        print(sum)