import requests
import json
import pandas as pd
import pandas_ta as ta


pd.set_option('display.max_columns', None)

def calculate_profit_percent(entry_price, exit_price, position_type):
    if position_type == 'BUY':
        net_profit = exit_price - entry_price
    elif position_type == 'SELL':
        net_profit = entry_price - exit_price
    else:
        raise ValueError(f"Unknown position type: {position_type}")

    profit_percent = (net_profit / entry_price) * 100
    return profit_percent

trend = False
check_trend = False
entry_price = 0
deposit = 100.1
all_profit = 0
all_loss = 0
close_check_price = 0

url = "https://fapi.binance.com/fapi/v1/klines?symbol=SOLUSDT&interval=15m&limit=100"
url1 = "https://fapi.binance.com/fapi/v1/klines?symbol=SOLUSDT&interval=1h&limit=100"


while True:
    try:
        response = requests.get(url)

        data = json.loads(response.text)

        df = pd.DataFrame(data,
                          columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                   'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

        df.drop(['quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'], axis=1, inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms') + pd.Timedelta(hours=2)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms') + pd.Timedelta(hours=2)

        df['supertrend'] = ta.supertrend(df['high'], df['low'], df['close'], 10, 3)['SUPERTd_10_3.0']


        if (trend == False and df.iloc[-2]['supertrend'] == 1 and df.iloc[-2]['close'] != close_check_price):
            trend = 'BUY'
            check_trend = False
            print(f'Operation: {trend}')
            print(f'Price Open: {df.iloc[-1]['close']}$')
            print(f'Deposit: {deposit}$')
            print(f'Time open: {df.iloc[-1]['close_time']}')

            entry_price = df.iloc[-1]['close']
            close_check_price = df.iloc[-2]['close']

        if (trend == False and df.iloc[-2]['supertrend'] == -1 and df.iloc[-2]['close'] != close_check_price):
            trend = 'SELL'
            check_trend = False
            print(f'Operation: {trend}')
            print(f'Price Open: {df.iloc[-1]['close']}$')
            print(f'Deposit: {deposit}$')
            print(f'Time open: {df.iloc[-1]['close_time']}')

            entry_price = df.iloc[-1]['close']
            close_check_price = df.iloc[-2]['close']


        ''''''''''''
        if(trend == 'BUY' and df.iloc[-2]['supertrend'] == -1):
            trend = False
            if check_trend == False:
                r = calculate_profit_percent(float(entry_price), float(df.iloc[-1]['close']), 'BUY')
                print(f'Price Close: {df.iloc[-1]['close']}$')
                print(f'Profit: {r}%')
                print(f'Deposit close: {deposit + ((deposit * r)/100)}$')
                print(f'Time close: {df.iloc[-1]['close_time']}')
                all_profit = all_profit + ((deposit * r) / 100)
                all_loss = all_loss + ((deposit * r) / 100)
                deposit = deposit * 2
                print(f'All profit: {all_profit}')
                print('=======================================')


        if (trend == 'SELL' and df.iloc[-2]['supertrend'] == 1):
            trend = False
            if check_trend == False:
                r = calculate_profit_percent(float(entry_price), float(df.iloc[-1]['close']), 'SELL')
                print(f'Price Close: {df.iloc[-1]['close']}$')
                print(f'Profit: {r}%')
                print(f'Deposit close: {deposit + ((deposit * r)/100)}$')
                print(f'Time close: {df.iloc[-1]['close_time']}')
                all_profit = all_profit + ((deposit * r) / 100)
                all_loss = all_loss + ((deposit * r) / 100)
                deposit = deposit * 2
                print(f'All profit: {all_profit}')
                print('=======================================')

        ''''''''''''
        if(trend == 'BUY' and check_trend == False and calculate_profit_percent(float(entry_price), float(df.iloc[-1]['high']), 'BUY') <= -1):
            trend = False
            r = calculate_profit_percent(float(entry_price), float(df.iloc[-1]['close']), 'BUY')
            print(f'Price Close: {df.iloc[-1]['close']}$')
            print(f'Profit: {r}%')
            print(f'Deposit close: {deposit + ((deposit * r) / 100)}$')
            print(f'Time close: {df.iloc[-1]['close_time']}')
            all_profit = all_profit + ((deposit * r) / 100)
            all_loss = all_loss + ((deposit * r) / 100)
            deposit = deposit * 2
            print(f'All profit: {all_profit}')
            print('=======================================')

        if (trend == 'SELL' and check_trend == False and calculate_profit_percent(float(entry_price), float(df.iloc[-1]['high']), 'SELL') <= -1):
            trend = False
            r = calculate_profit_percent(float(entry_price), float(df.iloc[-1]['close']), 'SELL')
            print(f'Price Close: {df.iloc[-1]['close']}$')
            print(f'Profit: {r}%')
            print(f'Deposit close: {deposit + ((deposit * r) / 100)}$')
            print(f'Time close: {df.iloc[-1]['close_time']}')
            all_profit = all_profit + ((deposit * r) / 100)
            all_loss = all_loss + ((deposit * r) / 100)
            deposit = deposit * 2
            print(f'All profit: {all_profit}')
            print('=======================================')


        ''''''''''''
        if (trend == 'BUY' and check_trend == False and calculate_profit_percent(float(entry_price), float(df.iloc[-1]['high']), 'BUY') >= 1):
            print(f'Price Close: {df.iloc[-1]['close']}$')
            print(f'Profit: {calculate_profit_percent(float(entry_price), float(df.iloc[-1]['close']), 'BUY')}%')
            print(f'Deposit close: {deposit + (deposit * 0.01)}$') #0.01
            print(f'Time close: {df.iloc[-1]['close_time']}')
            check_trend = True
            all_profit = all_profit + (deposit * 0.01)
            all_loss = 0
            deposit = 100
            print(f'All profit: {all_profit}')
            print('=======================================')


        if (trend == 'SELL' and check_trend == False and calculate_profit_percent(float(entry_price), float(df.iloc[-1]['low']), 'SELL') >= 1):
            print(f'Price Close: {df.iloc[-1]['close']}$')
            print(f'Profit: {calculate_profit_percent(float(entry_price), float(df.iloc[-1]['close']), 'SELL')}%')
            print(f'Deposit close: {deposit + (deposit * 0.01)}$') #0.01
            print(f'Time close: {df.iloc[-1]['close_time']}')
            check_trend = True
            all_profit = all_profit + (deposit * 0.01)
            all_loss = 0
            deposit = 100
            print(f'All profit: {all_profit}')
            print('=======================================')

    except Exception as e:
        print("Ошибка при отправке запроса")
        print(e)