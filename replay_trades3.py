#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import ccxt
import joblib
import warnings

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------- defaults -----------------
DEFAULT_MODEL_PATH = "final_model_lgbm.pkl"
DEFAULT_THRESHOLDS_PATH = "best_thresholds.json"
DEFAULT_MAX_BARS = 10000 # должно совпадать с aibot.py для консистентности признаков
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_SMOOTH_SPAN = None # None -> берем из файла

# =================================================================================
# КОД, СКОПИРОВАННЫЙ ИЗ AIBOT.PY ДЛЯ ПОЛНОЙ АВТОНОМНОСТИ
# =================================================================================

def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith('m'): return int(tf[:-1])
    if tf.endswith('h'): return int(tf[:-1]) * 60
    if tf.endswith('d'): return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Неизвестный таймфрейм: {tf}")

def fetch_ohlcv_all(exchange, symbol, timeframe='15m',
                    since_ms: int | None = None, limit=1000, max_bars=60_000):
    tf2min = timeframe_to_minutes(timeframe)
    ms_step = tf2min * 60_000
    if since_ms is None:
        since_ms = int((pd.Timestamp.utcnow()
                        - pd.Timedelta(minutes=(max_bars + 5) * tf2min)).timestamp() * 1000)
    out, last_ts = [], None
    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not chunk: break
            if last_ts is not None and chunk[-1][0] <= last_ts: break
            out += chunk
            last_ts = chunk[-1][0]
            since_ms = last_ts + ms_step
            if len(out) >= max_bars:
                out = out[-max_bars:]
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}. Повтор через 5 сек...")
            time.sleep(5)
    df = pd.DataFrame(out, columns=["ts","o","h","l","c","v"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def calculate_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Вычисляет Supertrend и возвращает (линия Supertrend, направление тренда True/False)."""
    df = df.copy()
    df['h'] = pd.to_numeric(df['h'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['c'] = pd.to_numeric(df['c'], errors='coerce')
    
    atr = AverageTrueRange(high=df['h'], low=df['l'], close=df['c'], window=atr_period).average_true_range()
    
    hl2 = (df['h'] + df['l']) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    in_uptrend = pd.Series(True, index=df.index)
    supertrend_line = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        current = i
        previous = i - 1
        
        if df['c'].iloc[current] > upper_band.iloc[previous]:
            in_uptrend.iloc[current] = True
        elif df['c'].iloc[current] < lower_band.iloc[previous]:
            in_uptrend.iloc[current] = False
        else:
            in_uptrend.iloc[current] = in_uptrend.iloc[previous]

        if in_uptrend.iloc[current] and lower_band.iloc[current] < lower_band.iloc[previous]:
            lower_band.iloc[current] = lower_band.iloc[previous]
        
        if not in_uptrend.iloc[current] and upper_band.iloc[current] > upper_band.iloc[previous]:
            upper_band.iloc[current] = upper_band.iloc[previous]

        if in_uptrend.iloc[current]:
            supertrend_line.iloc[current] = lower_band.iloc[current]
        else:
            supertrend_line.iloc[current] = upper_band.iloc[current]
            
    return supertrend_line, in_uptrend

# ----------------------------- Features & Labels -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # base
    d['ret1']  = d['c'].pct_change(1)
    d['ret3']  = d['c'].pct_change(3)
    d['ret5']  = d['c'].pct_change(5)
    d['ret20'] = d['c'].pct_change(20)

    # --- Убираем RSI, добавляем ADX ---
    adx_ind = ADXIndicator(high=d['h'], low=d['l'], close=d['c'], window=14)
    d['adx'] = adx_ind.adx()
    d['adx_pos'] = adx_ind.adx_pos()
    d['adx_neg'] = adx_ind.adx_neg()

    d['sma20'] = SMAIndicator(d['c'], 20).sma_indicator()
    d['sma50'] = SMAIndicator(d['c'], 50).sma_indicator()
    
    # Новый признак на основе Supertrend
    st_line, _ = calculate_supertrend(d, atr_period=14, atr_multiplier=2.5)
    d['st_dist'] = (d['c'] / st_line - 1).replace([np.inf, -np.inf], 0)

    # Новый, более робастный признак долгосрочного тренда
    d['sma200'] = SMAIndicator(d['c'], 200).sma_indicator()
    d['sma_ratio_long'] = d['sma50'] / d['sma200'] - 1
    
    macd = MACD(d['c'])
    # Нормализуем MACD
    d['macd_norm']      = macd.macd() / d['sma50']
    d['macd_sig_norm']  = macd.macd_signal() / d['sma50']
    d['macd_diff'] = macd.macd_diff()

    bb = BollingerBands(d['c'], window=20, window_dev=2.0)
    rng = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    d['bb_pos'] = (d['c'] - bb.bollinger_mavg()) / rng

    d['vol20'] = d['ret1'].rolling(20).std()

    # --- Признаки на основе объемов ---
    v_sma50 = d['v'].rolling(50).mean()
    d['v_sma_ratio'] = (d['v'] / v_sma50 - 1).replace([np.inf, -np.inf], 0)
    obv = OnBalanceVolumeIndicator(d['c'], d['v']).on_balance_volume()
    d['obv_momentum'] = obv.pct_change(50).replace([np.inf, -np.inf], 0)
    
    # ATR for Stop-Loss
    atr = AverageTrueRange(high=d['h'], low=d['l'], close=d['c'], window=14)
    d['atr'] = atr.average_true_range()
    
    # Нормализованный ATR для оценки волатильности относительно цены
    d['atr_norm'] = d['atr'] / d['sma20']

    d = d.dropna()
    return d

def _run_backtest_engine(df: pd.DataFrame, proba: pd.Series, tf: str, thresholds: dict,
                         min_hold: int, cooldown: int, fee_per_side: float, slippage_per_side: float,
                         trend_filter: pd.Series | None, use_stop_loss: bool, sl_atr_multiplier: float) -> dict:
    idx = df.index.intersection(proba.index)
    if len(idx) < 10:
        return {'pos': pd.Series(dtype=int), 'ret': pd.Series(dtype=float)}
        
    df = df.loc[idx]
    proba = proba.loc[idx]
    
    if trend_filter is not None:
        trend_filter = trend_filter.reindex(idx, method='ffill').fillna(False)
    if use_stop_loss and 'atr' not in df.columns:
        raise ValueError("ATR column not found for Stop-Loss.")
        
    cost_per_side = fee_per_side + slippage_per_side
    pos, hold_bars, cd_bars = 0, 0, 0
    entry_price, sl_price = np.nan, np.nan
    # Новые переменные для Trailing SL
    peak_since_entry, trough_since_entry = np.nan, np.nan
    positions, returns = [], []

    for i in range(len(df) - 1):
        ts, p = df.index[i], proba.iloc[i]
        is_exit, exit_price = False, np.nan
        current_pos = pos
        
        # 1. Проверка на Stop-Loss (приоритет)
        if current_pos != 0 and use_stop_loss and not np.isnan(sl_price):
            # Сначала обновляем Trailing Stop
            if current_pos == 1:
                peak_since_entry = max(peak_since_entry, df['h'].iloc[i])
                new_sl = peak_since_entry - sl_atr_multiplier * df['atr'].iloc[i]
                sl_price = max(sl_price, new_sl) # Стоп может только двигаться вверх
            elif current_pos == -1:
                trough_since_entry = min(trough_since_entry, df['l'].iloc[i])
                new_sl = trough_since_entry + sl_atr_multiplier * df['atr'].iloc[i]
                sl_price = min(sl_price, new_sl) # Стоп может только двигаться вниз

            # Теперь проверяем срабатывание
            if (current_pos == 1 and df['l'].iloc[i] <= sl_price) or \
               (current_pos == -1 and df['h'].iloc[i] >= sl_price):
                is_exit, exit_price = True, sl_price
        
        # 2. Проверка на выход по сигналу
        if not is_exit and current_pos != 0 and hold_bars >= min_hold:
            if (current_pos == 1 and p < thresholds['exit_long']) or \
               (current_pos == -1 and p > thresholds['exit_short']):
                is_exit = True
        
        # 3. Проверка на вход
        is_entry = False
        if current_pos == 0 and cd_bars == 0:
            can_long = trend_filter is None or trend_filter.iloc[i]
            can_short = trend_filter is None or not trend_filter.iloc[i]
            if p > thresholds['enter_long'] and can_long:
                is_entry, pos = True, 1
            elif p < thresholds['enter_short'] and can_short:
                is_entry, pos = True, -1
        
        # 4. Расчет доходности и обновление состояния
        ret_i = 0.0
        if is_exit:
            exit_price = df['o'].iloc[i+1] if np.isnan(exit_price) else exit_price
            gross_ret = (exit_price / entry_price - 1) if current_pos == 1 else (entry_price / exit_price - 1)
            ret_i = (1 + gross_ret) * (1 - cost_per_side) - 1 # учитываем только комиссию на выход
            pos, hold_bars, cd_bars = 0, 0, cooldown
            entry_price, sl_price = np.nan, np.nan
            peak_since_entry, trough_since_entry = np.nan, np.nan # Сброс
        
        if is_entry:
            entry_price = df['o'].iloc[i+1]
            ret_i -= cost_per_side # комиссия на вход
            hold_bars = 1
            if use_stop_loss:
                atr_val = df['atr'].iloc[i]
                if pos == 1:
                    sl_price = entry_price - sl_atr_multiplier * atr_val
                    peak_since_entry = entry_price
                else: # pos == -1
                    sl_price = entry_price + sl_atr_multiplier * atr_val
                    trough_since_entry = entry_price
        
        # Если позиция удерживается
        if pos != 0 and not is_exit and not is_entry:
            ret_i = (df['o'].iloc[i+1] / df['o'].iloc[i] - 1) * pos
            hold_bars += 1
        
        returns.append(ret_i)
        positions.append(pos)
        
        if cd_bars > 0 and pos == 0:
            cd_bars -= 1
            
    pos_s = pd.Series(positions, index=df.index[:-1])
    ret_s = pd.Series(returns, index=df.index[:-1])
    return {'pos': pos_s, 'ret': ret_s}

def extract_trades_from_pos_open(pos_s: pd.Series, df: pd.DataFrame, fee_per_side, slippage_per_side) -> pd.DataFrame:
    idx = pos_s.index.intersection(df.index)
    pos, o = pos_s.loc[idx], df.loc[idx, 'o']
    cost_side = fee_per_side + slippage_per_side
    trades, cur = [], None
    
    for i in range(len(pos)):
        p, ts, price = pos.iloc[i], pos.index[i], o.iloc[i]
        if cur is None and p != 0:
            cur = {'side': p, 'entry_ts': ts, 'entry_price': price, 'entry_i': i}
        elif cur is not None and p != cur['side']:
            side, entry_price = cur['side'], cur['entry_price']
            gross_ret = (price / entry_price - 1) if side == 1 else (entry_price / price - 1)
            net_ret = (1 + gross_ret) * (1 - cost_side) - 1 - cost_side # 2 комиссии
            
            trades.append({'side': "long" if side == 1 else "short", 'entry_ts': cur['entry_ts'], 'exit_ts': ts,
                           'entry_price': entry_price, 'exit_price': price, 'net_ret': net_ret,
                           'net_pct': net_ret * 100, 'bars': i - cur['entry_i']})
            
            cur = {'side': p, 'entry_ts': ts, 'entry_price': price, 'entry_i': i} if p != 0 else None

    if trades:
        df_tr = pd.DataFrame(trades)
        df_tr['win'] = df_tr['net_ret'] > 0
        return df_tr
    return pd.DataFrame()

def summarize_trades(trades: pd.DataFrame, timeframe: str) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'winrate': 0.0, 'profit_factor': 0.0, 'expectancy_pct': 0.0,
                'avg_win_pct': 0.0, 'avg_loss_pct': 0.0, 'avg_hold_hours': 0.0}
    
    wins = trades[trades['win']]
    losses = trades[~trades['win']]
    
    pf = np.inf
    if not losses.empty and losses['net_ret'].sum() != 0:
        pf = wins['net_ret'].sum() / abs(losses['net_ret'].sum())

    tf_min = timeframe_to_minutes(timeframe)

    res = {
        'n_trades': len(trades),
        'winrate': trades['win'].mean(),
        'profit_factor': pf,
        'expectancy_pct': trades['net_ret'].mean() * 100,
        'avg_win_pct': wins['net_pct'].mean() if not wins.empty else 0.0,
        'avg_loss_pct': losses['net_pct'].mean() if not losses.empty else 0.0,
        'avg_hold_hours': (trades['bars'].mean() * tf_min) / 60.0
    }
    return res

def save_trades_json(trades, json_path, csv_path):
    if not trades.empty:
        trades_csv = trades.copy()
        trades_csv['entry_ts'] = pd.to_datetime(trades_csv['entry_ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_csv['exit_ts'] = pd.to_datetime(trades_csv['exit_ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_csv.to_csv(csv_path, index=False)
        
        trades_json = trades.copy()
        trades_json['entry_ts'] = pd.to_datetime(trades_json['entry_ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_json['exit_ts'] = pd.to_datetime(trades_json['exit_ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(json_path, 'w') as f:
            json.dump(trades_json.to_dict('records'), f, indent=2)
    print(f"Saved: {csv_path}, {json_path}")

def load_cashflows(path):
    if not os.path.exists(path): return pd.Series(dtype=float)
    df = pd.read_json(path)
    if 'ts' not in df.columns or 'amount' not in df.columns: return pd.Series(dtype=float)
    return pd.Series(df['amount'].values, index=pd.to_datetime(df['ts'])).sort_index()

def align_cashflows_to_index(cflows, bar_index):
    if cflows.empty: return pd.Series(dtype=float)
    # каждый кэшфлоу на ближайший СЛЕДУЮЩИЙ бар
    return cflows.reindex(bar_index, method='bfill').dropna()

def equity_with_cashflows(strat_ret: pd.Series, initial_capital: float, cashflows_ts: pd.Series):
    cflows_aligned = pd.Series(dtype=float)
    if not cashflows_ts.empty:
        cflows_aligned = cashflows_ts.reindex(strat_ret.index).fillna(0)

    eq = pd.Series(index=strat_ret.index, dtype=float)
    capital = initial_capital
    
    # Итерируемся и считаем эквити бар за баром
    for ts, r in strat_ret.items():
        if ts in cflows_aligned.index:
            capital += cflows_aligned[ts]
        capital *= (1 + r)
        eq.loc[ts] = capital
    
    final_val = eq.iloc[-1] if not eq.empty else initial_capital + cashflows_ts.sum()
    total_deposits = cashflows_ts[cashflows_ts > 0].sum()
    total_withdrawals = cashflows_ts[cashflows_ts < 0].sum()
    
    net_contrib = initial_capital + total_deposits
    profit = final_val - net_contrib + abs(total_withdrawals)
    roi = profit / net_contrib if net_contrib > 0 else 0.0
    
    return eq, final_val, profit, roi

# =================================================================================
# КОНЕЦ СКОПИРОВАННОГО КОДА
# =================================================================================

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Replay trades from saved model & thresholds (+cashflows)")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH, help="файл модели (из aibot.py)")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS_PATH, help="файл порогов (из aibot.py)")
    ap.add_argument("--symbol", default=None, help="переписать символ (иначе из thresholds)")
    ap.add_argument("--timeframe", default=None, help="переписать TF (иначе из thresholds)")
    ap.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS)
    ap.add_argument("--smooth-span", type=int, default=DEFAULT_SMOOTH_SPAN, help="переписать сглаживание (иначе из модели/порогов)")
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="стартовый депозит")
    ap.add_argument("--cashflows", type=str, default="cashflows.json", help="JSON с депозитами/выводами")
    ap.add_argument("--out-prefix", default="trades_replay", help="префикс для файлов вывода")

    # --- Новые флаги для переопределения ---
    ap.add_argument("--disable-htf-filter", dest="use_htf_filter_override", action="store_false", help="Принудительно отключить HTF-фильтр")
    ap.add_argument("--enable-htf-filter", dest="use_htf_filter_override", action="store_true", help="Принудительно включить HTF-фильтр")
    ap.set_defaults(use_htf_filter_override=None)

    ap.add_argument("--disable-stop-loss", dest="use_stop_loss_override", action="store_false", help="Принудительно отключить Stop-Loss")
    ap.add_argument("--enable-stop-loss", dest="use_stop_loss_override", action="store_true", help="Принудительно включить Stop-Loss")
    ap.set_defaults(use_stop_loss_override=None)
    
    ap.add_argument("--sl-atr-multiplier", type=float, default=None, help="Переопределить множитель ATR для стоп-лосса")

    args = ap.parse_args()

    for path in [args.model, args.thresholds]:
        if not os.path.exists(path):
            print(f"Ошибка: Не найден файл '{path}'. Сначала запустите aibot.py."); sys.exit(1)

    # --- Читаем конфиг из файла порогов ---
    with open(args.thresholds, "r") as f:
        cfg = json.load(f)
    
    th = cfg["thresholds"]
    min_hold = int(cfg["min_hold"])
    cooldown = int(cfg["cooldown"])
    fee_side = float(cfg["fees"]["fee_per_side"])
    slip_side = float(cfg["fees"]["slippage_per_side"])
    symbol = args.symbol or cfg["symbol"]
    timeframe = args.timeframe or cfg["timeframe"]
    
    # HTF filter
    htf_filter_settings = cfg.get("htf_filter", {})
    use_htf = htf_filter_settings.get("use", False)
    if args.use_htf_filter_override is not None:
        use_htf = args.use_htf_filter_override # Переопределение из CLI
    htf_tf = htf_filter_settings.get("timeframe")
    htf_supertrend_period = htf_filter_settings.get("supertrend_period")
    htf_supertrend_multiplier = htf_filter_settings.get("supertrend_multiplier")

    # Stop-Loss
    sl_settings = cfg.get("stop_loss", {})
    use_sl = sl_settings.get("use", False)
    if args.use_stop_loss_override is not None:
        use_sl = args.use_stop_loss_override # Переопределение из CLI
        
    sl_mult_from_file = cfg.get("stop_loss", {}).get("atr_multiplier")
    sl_mult = args.sl_atr_multiplier if args.sl_atr_multiplier is not None else float(sl_mult_from_file) if sl_mult_from_file is not None else 3.0

    # --- Читаем модель ---
    pack = joblib.load(args.model)
    model = pack["model"]
    # Используем финальный набор признаков
    feat_cols = [
        # Сила и направление тренда
        'adx', 'adx_pos', 'adx_neg',
        # Подтверждение объемом
        'v_sma_ratio', 'obv_momentum',
        # Долгосрочное направление и моментум
        'sma_ratio_long', 'macd_norm', 'macd_sig_norm',
        # Положение относительно тренда
        'st_dist',
        # Волатильность
        'vol20', 'atr_norm'
    ]
    smooth_span = args.smooth_span if args.smooth_span is not None else cfg.get("smooth_span", 1)

    # --- Загрузка данных ---
    ex = ccxt.binance(); ex.enableRateLimit = True
    print(f"Загрузка: {symbol} {timeframe} (max {args.max_bars} баров)")
    df = fetch_ohlcv_all(ex, symbol, timeframe=timeframe, max_bars=args.max_bars)
    if len(df) < 500:
        print(f"Слишком мало баров: {len(df)}"); sys.exit(1)

    # --- HTF Фильтр (если включен) ---
    trend_filter = None
    if use_htf:
        print(f"Загрузка данных для HTF: {symbol} {htf_tf}")
        main_tf_mins = timeframe_to_minutes(timeframe)
        htf_tf_mins = timeframe_to_minutes(htf_tf)
        htf_bars_needed = int((len(df) * main_tf_mins) / htf_tf_mins) + (htf_supertrend_period or 14) + 5
        
        df_htf = fetch_ohlcv_all(ex, symbol, timeframe=htf_tf, max_bars=htf_bars_needed)
        _, trend_direction = calculate_supertrend(
            df_htf, 
            atr_period=htf_supertrend_period or 14, 
            atr_multiplier=htf_supertrend_multiplier or 2.5
        )
        trend_filter = trend_direction.rename("trend_up")
        print("HTF фильтр рассчитан.")

    # --- Признаки и предсказания ---
    df_feat = build_features(df)
    X = df_feat[feat_cols].shift(1).dropna() # shift(1) для защиты от заглядывания в будущее
    
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    if smooth_span and smooth_span > 1:
        proba = proba.ewm(span=int(smooth_span), min_periods=int(smooth_span)).mean()

    # --- Бэктест ---
    print("Запуск бэктеста на финальной модели...")
    backtest_results = _run_backtest_engine(
        df=df_feat, proba=proba, tf=timeframe, thresholds=th,
        min_hold=min_hold, cooldown=cooldown,
        fee_per_side=fee_side, slippage_per_side=slip_side,
        trend_filter=trend_filter,
        use_stop_loss=use_sl, sl_atr_multiplier=sl_mult
    )
    pos_s = backtest_results.get("pos")
    strat_ret = backtest_results.get("ret")

    if pos_s is None or pos_s.empty:
        print("Бэктест не вернул позиций."); return

    # --- Анализ и сохранение результатов ---
    trades = extract_trades_from_pos_open(pos_s, df_feat, fee_side, slip_side)
    summary = summarize_trades(trades, timeframe)
    
    print("\n--- Результаты Replay ---")
    print(f"Сделок: {summary['n_trades']} | Винрейт: {summary['winrate']:.1%} | Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Средняя прибыль/убыток: {summary['avg_win_pct']:.2f}% / {summary['avg_loss_pct']:.2f}%")
    print(f"Мат. ожидание: {summary['expectancy_pct']:.3f}% | Среднее удержание: {summary['avg_hold_hours']:.1f} ч")

    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"
    save_trades_json(trades, json_path, csv_path)

    # --- Эквити с учетом кэшфлоу ---
    cfs = load_cashflows(args.cashflows)
    cfs_aligned = align_cashflows_to_index(cfs, strat_ret.index)
    
    equity, final_val, profit, roi = equity_with_cashflows(
        strat_ret, args.initial_capital, cfs_aligned
    )

    eq_path = f"{args.out_prefix}_equity.csv"
    summary_path = f"{args.out_prefix}_summary.json"
    pd.DataFrame({"value": equity}).to_csv(eq_path)
    
    summary_data = {
        "initial_capital": args.initial_capital,
        "total_deposits": cfs[cfs > 0].sum(),
        "total_withdrawals": cfs[cfs < 0].sum(),
        "final_value": final_val,
        "net_profit": profit,
        "roi_pct": roi * 100,
        "start_ts": str(equity.index[0]) if not equity.empty else None,
        "end_ts": str(equity.index[-1]) if not equity.empty else None,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved: {eq_path}, {summary_path}")
    print(f"Финансовый результат: Начальный капитал={args.initial_capital:.2f} -> Конечный={final_val:.2f}, "
          f"Профит={profit:.2f} (ROI {roi*100:.2f}%)")

if __name__ == "__main__":
    main()
