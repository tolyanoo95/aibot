#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands

from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 50)

# ----------------------------- Defaults -----------------------------
DEFAULT_SYMBOL = "SOL/USDT"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_MAX_BARS = 100000             # ~2-3 года 15m
DEFAULT_HORIZON_BARS = 24            # горизонт цели (≈ 6ч для 15m)
DEFAULT_SMOOTH_SPAN = 12             # EMA сглаживание вероятностей
FEE_PER_SIDE = 0.0003                # комиссия на сторону (0.03%)
SLIPPAGE_PER_SIDE = 0.0001           # проскальзывание на сторону (0.01%)
RANDOM_STATE = 42
DEFAULT_TURNOVER_CAP = 0.05          # макс. поворотов на бар
DEFAULT_LAST_DAYS = 0               # окно "последний месяц"

# ----------------------------- Utils -----------------------------
def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith('m'):
        return int(tf[:-1])
    if tf.endswith('h'):
        return int(tf[:-1]) * 60
    if tf.endswith('d'):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Неизвестный таймфрейм: {tf}")

def bars_per_year(tf: str) -> float:
    minutes = timeframe_to_minutes(tf)
    return (365.0 * 24.0 * 60.0) / minutes

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0

# ----------------------------- Data -----------------------------
def fetch_ohlcv_all(exchange, symbol, timeframe='15m',
                    since_ms: int | None = None, limit=1000, max_bars=60000):
    tf2min = timeframe_to_minutes(timeframe)
    ms_step = tf2min * 60_000

    # якорим к "сейчас": берем последние max_bars (+ небольшой запас)
    if since_ms is None:
        since_ms = int((pd.Timestamp.utcnow()
                        - pd.Timedelta(minutes=(max_bars + 5) * tf2min)).timestamp() * 1000)

    out, last_ts = [], None
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not chunk:
            break
        # защита от зацикливания
        if last_ts is not None and chunk[-1][0] <= last_ts:
            break
        out += chunk
        last_ts = chunk[-1][0]
        since_ms = last_ts + ms_step
        if len(out) >= max_bars:
            # оставляем последние max_bars чтобы точно «дотянуть» до конца
            out = out[-max_bars:]
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(out, columns=["ts","o","h","l","c","v"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

# ----------------------------- Features & Labels -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d['ret1']  = d['c'].pct_change()
    d['ret3']  = d['c'].pct_change(3)
    d['ret5']  = d['c'].pct_change(5)
    d['ret20'] = d['c'].pct_change(20)

    d['rsi'] = RSIIndicator(d['c'], window=14).rsi()

    d['sma20'] = SMAIndicator(d['c'], 20).sma_indicator()
    d['sma50'] = SMAIndicator(d['c'], 50).sma_indicator()
    d['sma_ratio'] = d['sma20'] / d['sma50'] - 1

    macd = MACD(d['c'])
    d['macd']      = macd.macd()
    d['macd_sig']  = macd.macd_signal()
    d['macd_diff'] = macd.macd_diff()

    bb = BollingerBands(d['c'], window=20, window_dev=2.0)
    rng = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    d['bb_pos'] = (d['c'] - bb.bollinger_mavg()) / rng

    d['vol20'] = d['ret1'].rolling(20).std()

    d = d.dropna()
    return d

def make_labels(df: pd.DataFrame, horizon_bars: int,
                fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> pd.Series:
    """Бинарная цель: 1 — если будущая доходность за H баров > полных издержек (вход+выход)."""
    fee_round_trip = 2.0 * (fee_per_side + slippage_per_side)
    fwd = df['c'].shift(-horizon_bars) / df['c'] - 1.0
    y = (fwd > fee_round_trip).astype(int)
    return y

# ----------------------------- Modeling -----------------------------
def oof_predict_lgbm(X: pd.DataFrame, y: pd.Series, n_splits=5, random_state=RANDOM_STATE) -> pd.Series:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba = pd.Series(index=X.index, dtype=float)
    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=10,
        min_data_in_bin=1,
        min_gain_to_split=1e-8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary',
        class_weight='balanced',
        force_col_wise=True,
        verbosity=-1,
        random_state=random_state
    )
    for tr, te in tscv.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xvl, yvl = X.iloc[te], y.iloc[te]
        clf = LGBMClassifier(**params)
        clf.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric='binary_logloss')
        proba.iloc[te] = clf.predict_proba(Xvl)[:, 1]
    return proba

def baseline_auc(X: pd.DataFrame, y: pd.Series) -> float | None:
    """Быстрый sanity-check: логистическая регрессия на последних 20% данных."""
    if len(X) < 2000:
        return None
    n = len(X)
    cut = int(n * 0.8)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X.iloc[:cut])
    Xte = scaler.transform(X.iloc[cut:])
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)

# ----------------------------- Position Logic (Hysteresis) -----------------------------
def positions_hysteresis(proba: pd.Series,
                         enter_long: float, exit_long: float,
                         enter_short: float, exit_short: float,
                         min_hold: int = 24, cooldown: int = 12) -> pd.Series:
    """Состояния -1/0/1 с гистерезисом, min_hold и cooldown после выхода."""
    state = 0
    hold = 0
    cd = 0
    pos = []
    for p in proba.values:
        if cd > 0:
            if state != 0:
                hold += 1
                if state == 1 and hold >= min_hold and p < exit_long:
                    state, hold, cd = 0, 0, cooldown
                elif state == -1 and hold >= min_hold and p > exit_short:
                    state, hold, cd = 0, 0, cooldown
            else:
                cd -= 1
        else:
            if state == 0:
                if p > enter_long:
                    state, hold = 1, 0
                elif p < enter_short:
                    state, hold = -1, 0
            elif state == 1:
                hold += 1
                if hold >= min_hold and p < exit_long:
                    state, hold, cd = 0, 0, cooldown
            elif state == -1:
                hold += 1
                if hold >= min_hold and p > exit_short:
                    state, hold, cd = 0, 0, cooldown
        pos.append(state)
    return pd.Series(pos, index=proba.index, dtype=int)

def _align_returns_and_proba(prices: pd.Series, proba: pd.Series):
    """Пересекаем индексы, считаем ретёрны, затем ещё раз выравниваем proba по ретёрнам."""
    proba = proba.dropna()
    idx = proba.index.intersection(prices.index)
    if len(idx) < 3:
        return None, None
    prices = prices.loc[idx]
    proba = proba.loc[idx]
    ret = prices.pct_change().dropna()
    proba = proba.loc[ret.index]
    return ret, proba

def backtest_hysteresis(prices: pd.Series, proba: pd.Series, tf: str,
                        enter_long_q=0.90, exit_long_q=0.70,
                        enter_short_q=0.10, exit_short_q=0.30,
                        min_hold=24, cooldown=12,
                        fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> dict:
    """Бэктест с гистерезисом и min_hold. Пороговые значения берутся по квантилям proba."""
    ret, proba = _align_returns_and_proba(prices, proba)
    if ret is None or len(ret) < 10:
        return dict(sharpe=0.0, cagr=0.0, max_dd=0.0, n_bars=0, turns=0,
                    long_share=0.0, short_share=0.0, neutral_share=1.0,
                    thresholds=dict(enter_long=np.nan, exit_long=np.nan,
                                    enter_short=np.nan, exit_short=np.nan),
                    params=dict(enter_long_q=enter_long_q, exit_long_q=exit_long_q,
                                enter_short_q=enter_short_q, exit_short_q=exit_short_q,
                                min_hold=min_hold, cooldown=cooldown),
                    equity=pd.Series(dtype=float), ret=pd.Series(dtype=float), pos=pd.Series(dtype=int))

    q = proba.quantile
    enter_long  = float(q(enter_long_q))
    exit_long   = float(q(exit_long_q))
    enter_short = float(q(enter_short_q))
    exit_short  = float(q(exit_short_q))

    pos_raw = positions_hysteresis(
        proba, enter_long, exit_long, enter_short, exit_short,
        min_hold=min_hold, cooldown=cooldown
    )
    # без утечки: позиции действуют со следующего бара (и под ret.index)
    pos = pos_raw.shift(1).reindex(ret.index).fillna(0)

    turns = pos.diff().abs().fillna(0)
    cost_per_side = fee_per_side + slippage_per_side
    costs = turns * cost_per_side

    strat_ret = pos * ret - costs
    strat_ret = strat_ret.dropna()
    eq = (1.0 + strat_ret).cumprod()

    bpy = bars_per_year(tf)
    sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * np.sqrt(bpy) if len(strat_ret) else 0.0
    if len(eq) >= 2:
        years = len(strat_ret) / bpy
        cagr = eq.iloc[-1] ** (1/years) - 1.0 if years > 0 else 0.0
    else:
        cagr = 0.0
    mdd = max_drawdown(eq)

    return dict(
        sharpe=float(sharpe), cagr=float(cagr), max_dd=float(mdd),
        n_bars=int(len(strat_ret)), turns=int(turns.sum()),
        long_share=float((pos_raw == 1).mean()) if len(pos_raw) else 0.0,
        short_share=float((pos_raw == -1).mean()) if len(pos_raw) else 0.0,
        neutral_share=float((pos_raw == 0).mean()) if len(pos_raw) else 1.0,
        thresholds=dict(enter_long=enter_long, exit_long=exit_long,
                        enter_short=enter_short, exit_short=exit_short),
        params=dict(enter_long_q=enter_long_q, exit_long_q=exit_long_q,
                    enter_short_q=enter_short_q, exit_short_q=exit_short_q,
                    min_hold=min_hold, cooldown=cooldown),
        equity=eq, ret=strat_ret, pos=pos_raw
    )

def smooth_proba(proba: pd.Series, span: int) -> pd.Series:
    if span is None or span <= 1:
        return proba
    return proba.ewm(span=span, adjust=False).mean()

def search_thresholds(proba: pd.Series, prices: pd.Series, tf: str,
                      turnover_cap=DEFAULT_TURNOVER_CAP,
                      fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE):
    """Перебор порогов + min_hold. Возвращает лучший по Sharpe (с cap) и лучший без cap."""
    grid_enter = [0.80, 0.85, 0.90, 0.93]
    grid_exit  = [0.60, 0.65, 0.70]
    grid_enter_s = [0.20, 0.15, 0.10, 0.07]
    grid_exit_s  = [0.40, 0.35, 0.30]
    grid_hold = [16, 24, 36, 48]
    cooldowns = [4, 12, 24]

    best, best_stats = None, None
    best_loose, best_loose_stats = None, None

    for el in grid_enter:
        for xl in grid_exit:
            if xl >= el:
                continue
            for es in grid_enter_s:
                for xs in grid_exit_s:
                    if xs <= es:
                        continue
                    for mh in grid_hold:
                        for cd in cooldowns:
                            stats = backtest_hysteresis(
                                prices, proba, tf,
                                enter_long_q=el, exit_long_q=xl,
                                enter_short_q=es, exit_short_q=xs,
                                min_hold=mh, cooldown=cd,
                                fee_per_side=fee_per_side, slippage_per_side=slippage_per_side
                            )
                            if stats['n_bars'] == 0:
                                continue
                            tp_bar = stats['turns'] / max(1, stats['n_bars'])
                            # лучший без ограничения
                            if best_loose is None or stats['sharpe'] > best_loose_stats['sharpe']:
                                best_loose, best_loose_stats = (el, xl, es, xs, mh, cd), stats
                            # с ограничением на оборот
                            if tp_bar <= turnover_cap:
                                if best is None or stats['sharpe'] > best_stats['sharpe']:
                                    best, best_stats = (el, xl, es, xs, mh, cd), stats
    return best, best_stats, best_loose, best_loose_stats

# ----------------------------- Trades extraction & JSON -----------------------------
def extract_trades_from_pos(prices: pd.Series, pos_exec: pd.Series,
                            fee_per_side=0.0003, slippage_per_side=0.0001) -> pd.DataFrame:
    """
    Превращает серию позиций (-1/0/1), уже сдвинутую на 1 бар (execution), в список сделок.
    Считаем вход/выход по цене закрытия бара; издержки — multiplicative: (1 - fee-slippage) на вход и на выход.
    """
    prices = prices.sort_index()
    pos_exec = pos_exec.sort_index().astype(int)

    idx = prices.index.intersection(pos_exec.index)
    px = prices.loc[idx]
    ps = pos_exec.loc[idx]

    cost_side = float(fee_per_side + slippage_per_side)

    trades = []
    cur = None  # {'side': 1/-1, 'entry_ts', 'entry_price', 'entry_i'}

    for i, ts in enumerate(idx):
        p = int(ps.iloc[i])
        price = float(px.iloc[i])

        if cur is None:
            if p != 0:
                cur = dict(side=p, entry_ts=ts, entry_price=price, entry_i=i)
            continue

        # в позиции
        if p == cur["side"]:
            continue

        # закрытие текущей позиции на этом баре
        exit_ts = ts
        exit_price = price
        side = cur["side"]
        entry_price = float(cur["entry_price"])

        if side == 1:
            gross = exit_price / entry_price - 1.0
        else:
            gross = entry_price / exit_price - 1.0

        # net через мультипликативные издержки (вход + выход)
        net = (1.0 + gross) * (1.0 - cost_side) * (1.0 - cost_side) - 1.0
        bars_held = i - int(cur["entry_i"])

        trades.append(dict(
            side="long" if side == 1 else "short",
            entry_ts=cur["entry_ts"], exit_ts=exit_ts,
            entry_price=entry_price, exit_price=exit_price,
            gross_ret=gross, net_ret=net,
            bars=bars_held
        ))

        # flip в ту же свечу, если p != 0
        cur = None
        if p != 0:
            cur = dict(side=p, entry_ts=ts, entry_price=price, entry_i=i)

    # если осталась открытая позиция — закроем на последней доступной цене (чтобы видеть все сделки)
    if cur is not None:
        ts = idx[-1]
        price = float(px.iloc[-1])
        side = cur["side"]
        entry_price = float(cur["entry_price"])
        if side == 1:
            gross = price / entry_price - 1.0
        else:
            gross = entry_price / price - 1.0
        net = (1.0 + gross) * (1.0 - cost_side) * (1.0 - cost_side) - 1.0
        bars_held = len(idx) - 1 - int(cur["entry_i"])
        trades.append(dict(
            side="long" if side == 1 else "short",
            entry_ts=cur["entry_ts"], exit_ts=ts,
            entry_price=entry_price, exit_price=price,
            gross_ret=gross, net_ret=net,
            bars=bars_held
        ))

    df_tr = pd.DataFrame(trades)
    if not df_tr.empty:
        df_tr["win"] = df_tr["net_ret"] > 0
        df_tr["gross_pct"] = df_tr["gross_ret"] * 100
        df_tr["net_pct"] = df_tr["net_ret"] * 100
    return df_tr

def save_trades_json(trades: pd.DataFrame, path="trades_all.json", fields=None) -> str:
    """Сохраняет сделки в JSON. По умолчанию: entry_ts, entry_price, exit_ts, exit_price, side, net_pct."""
    if trades is None or trades.empty:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return path

    t = trades.copy()
    # нормализуем формат времени
    t["entry_ts"] = pd.to_datetime(t["entry_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    t["exit_ts"]  = pd.to_datetime(t["exit_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    if fields is None:
        fields = ["entry_ts", "entry_price", "exit_ts", "exit_price", "side", "net_pct"]
    records = t[fields].to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return path

# ----------------------------- Cashflows (депозиты/выводы) -----------------------------
def load_cashflows(path: str) -> pd.Series:
    """Читает cashflows.json -> Series(amount) по времени. Пустая серия, если файла нет."""
    if not path or not os.path.exists(path):
        return pd.Series(dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        return pd.Series(dtype=float)
    df = pd.DataFrame(data)
    if "ts" not in df or "amount" not in df:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(df["ts"], errors="coerce")
    amt = pd.to_numeric(df["amount"], errors="coerce")
    ok = (~ts.isna()) & (~amt.isna())
    s = pd.Series(amt[ok].values, index=ts[ok].values)
    s = s.groupby(s.index).sum().sort_index()
    return s

def align_cashflows_to_index(cflows: pd.Series, bar_index: pd.Index) -> pd.Series:
    """Проецируем каждый кэшфлоу на ближайший СЛЕДУЮЩИЙ бар (если позже последнего — игнор)."""
    if cflows is None or cflows.empty or len(bar_index) == 0:
        return pd.Series(dtype=float)
    mapped = {}
    for ts, amt in cflows.items():
        if ts in bar_index:
            mapped[ts] = mapped.get(ts, 0.0) + float(amt)
        else:
            pos = bar_index.searchsorted(ts)  # первый bar >= ts
            if pos < len(bar_index):
                key = bar_index[pos]
                mapped[key] = mapped.get(key, 0.0) + float(amt)
    if not mapped:
        return pd.Series(dtype=float)
    s = pd.Series(mapped).sort_index()
    return s

def equity_with_cashflows(strat_ret: pd.Series, initial_capital: float, cflows_on_bars: pd.Series) -> tuple:
    """
    Итеративно считаем стоимость портфеля с учётом депозитов/выводов.
    Кэшфлоу применяется ПЕРЕД доходностью бара.
    """
    value = float(initial_capital)
    vals = []
    for ts in strat_ret.index:
        if (cflows_on_bars is not None) and (ts in cflows_on_bars.index):
            value += float(cflows_on_bars.loc[ts])
        value *= (1.0 + float(strat_ret.loc[ts]))
        vals.append((ts, value))
    if not vals:
        series = pd.Series(dtype=float)
    else:
        series = pd.Series([v for _, v in vals], index=[t for t, _ in vals])
    dep = float(cflows_on_bars[cflows_on_bars > 0].sum()) if cflows_on_bars is not None and len(cflows_on_bars) else 0.0
    wd  = float(-cflows_on_bars[cflows_on_bars < 0].sum()) if cflows_on_bars is not None and len(cflows_on_bars) else 0.0
    final_value = float(series.iloc[-1]) if len(series) else initial_capital + dep - wd
    net_contrib = float(initial_capital + dep - wd)
    profit = final_value - net_contrib
    roi = (profit / net_contrib) if net_contrib > 0 else np.nan
    return series, dep, wd, final_value, profit, roi

# (опционально) краткая сводка по сделкам
def summarize_trades(trades: pd.DataFrame, timeframe: str) -> str:
    if trades.empty:
        return "Сделок не найдено."

    total = len(trades)
    longs = trades[trades.side == "long"]
    shorts = trades[trades.side == "short"]

    wins = trades[trades.win]
    losses = trades[~trades.win]

    def winrate(df):
        return (df["win"].mean() * 100) if len(df) else 0.0

    pf = (wins["net_ret"].sum() / abs(losses["net_ret"].sum())) if len(losses) and abs(losses["net_ret"].sum()) > 1e-12 else np.inf
    exp = trades["net_ret"].mean()

    mins = timeframe_to_minutes(timeframe)
    avg_bars = trades["bars"].mean()
    avg_hours = avg_bars * mins / 60.0

    wl = trades["win"].astype(int).tolist()
    max_w, max_l, cur_w, cur_l = 0, 0, 0, 0
    for x in wl:
        if x == 1:
            cur_w += 1; max_w = max(max_w, cur_w); cur_l = 0
        else:
            cur_l += 1; max_l = max(max_l, cur_l); cur_w = 0

    lines = []
    lines.append(f"Всего сделок: {total}  (Long: {len(longs)}, Short: {len(shorts)})")
    lines.append(f"Winrate общий: {winrate(trades):.1f}%  |  Long: {winrate(longs):.1f}%  |  Short: {winrate(shorts):.1f}%")
    lines.append(f"Profit factor: {pf:.2f}  |  Expectancy/сделка: {exp*100:.2f}%")
    lines.append(f"Средняя длительность: {avg_bars:.1f} баров (~{avg_hours:.1f} ч)")
    lines.append(f"Макс серия побед: {max_w}  |  Макс серия убыточных: {max_l}")
    lines.append(f"Средняя прибыльная: {wins['net_ret'].mean()*100:.2f}%  |  Средняя убыточная: {losses['net_ret'].mean()*100:.2f}%")
    return "\n".join(lines)

# ----------------------------- Fixed-threshold backtest -----------------------------
def backtest_hysteresis_fixed_thresholds(prices: pd.Series, proba: pd.Series, tf: str,
                                         thresholds: dict, min_hold=24, cooldown=12,
                                         fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> dict:
    """Та же логика, но пороги заданы числами (не квантилями). Удобно для теста последнего окна."""
    ret, proba = _align_returns_and_proba(prices, proba)
    if ret is None or len(ret) < 10:
        return dict(sharpe=0.0, cagr=0.0, max_dd=0.0, n_bars=0, turns=0,
                    long_share=0.0, short_share=0.0, neutral_share=1.0,
                    thresholds=thresholds, params=dict(min_hold=min_hold, cooldown=cooldown),
                    equity=pd.Series(dtype=float), ret=pd.Series(dtype=float), pos=pd.Series(dtype=int))

    enter_long  = float(thresholds["enter_long"])
    exit_long   = float(thresholds["exit_long"])
    enter_short = float(thresholds["enter_short"])
    exit_short  = float(thresholds["exit_short"])

    pos_raw = positions_hysteresis(
        proba, enter_long, exit_long, enter_short, exit_short,
        min_hold=min_hold, cooldown=cooldown
    )
    pos = pos_raw.shift(1).reindex(ret.index).fillna(0)

    turns = pos.diff().abs().fillna(0)
    cost_per_side = fee_per_side + slippage_per_side
    costs = turns * cost_per_side

    strat_ret = pos * ret - costs
    strat_ret = strat_ret.dropna()
    eq = (1.0 + strat_ret).cumprod()

    bpy = bars_per_year(tf)
    sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * np.sqrt(bpy) if len(strat_ret) else 0.0
    if len(eq) >= 2:
        years = len(strat_ret) / bpy
        cagr = eq.iloc[-1] ** (1/years) - 1.0 if years > 0 else 0.0
    else:
        cagr = 0.0
    mdd = max_drawdown(eq)

    return dict(
        sharpe=float(sharpe), cagr=float(cagr), max_dd=float(mdd),
        n_bars=int(len(strat_ret)), turns=int(turns.sum()),
        long_share=float((pos_raw == 1).mean()) if len(pos_raw) else 0.0,
        short_share=float((pos_raw == -1).mean()) if len(pos_raw) else 0.0,
        neutral_share=float((pos_raw == 0).mean()) if len(pos_raw) else 1.0,
        thresholds=dict(enter_long=enter_long, exit_long=exit_long,
                        enter_short=enter_short, exit_short=exit_short),
        params=dict(min_hold=min_hold, cooldown=cooldown),
        equity=eq, ret=strat_ret, pos=pos_raw
    )

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="AI crypto bot (LightGBM, hysteresis, smoothing, OOT, trades JSON, cashflows, last-window)")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME)
    parser.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_BARS, help="target horizon in bars")
    parser.add_argument("--smooth-span", type=int, default=DEFAULT_SMOOTH_SPAN, help="EMA span for proba smoothing (bars)")
    parser.add_argument("--turnover-cap", type=float, default=DEFAULT_TURNOVER_CAP, help="max turns per bar")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="стартовый капитал для денежного эквити")
    parser.add_argument("--cashflows", type=str, default="cashflows.json", help="путь к JSON с кэшфлоу [{'ts': 'YYYY-MM-DD HH:MM:SS', 'amount': 1000}, ...]")
    parser.add_argument("--last-days", type=int, default=DEFAULT_LAST_DAYS, help="оценить результат только за последние N дней")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print(f"Downloading: {args.symbol} {args.timeframe} (max {args.max_bars} bars)")
    ex = ccxt.binance()
    ex.enableRateLimit = True

    df = fetch_ohlcv_all(ex, args.symbol, timeframe=args.timeframe, max_bars=args.max_bars)
    if len(df) < 5000:
        print(f"WARNING: bars={len(df)} — мало данных. Рекомендуется >= 10–20k.")
    df = df.rename(columns=str.lower)
    print("Raw bars:", len(df))

    df_feat = build_features(df)
    y = make_labels(df_feat, args.horizon, FEE_PER_SIDE, SLIPPAGE_PER_SIDE)

    feat_cols = ['ret1','ret3','ret5','ret20','rsi','sma_ratio','macd','macd_sig','macd_diff','bb_pos','vol20']
    X = df_feat[feat_cols].shift(1).dropna()  # shift(1) — защита от утечки
    y = y.loc[X.index]

    # Самодиагностика
    print("X shape:", X.shape, "  y balance (mean of class=1):", round(y.mean(), 3))
    nunique = X.nunique().to_dict()
    stds = X.std().to_dict()
    print("Unique per feature:", {k:int(v) for k,v in nunique.items()})
    print("Std per feature    :", {k:round(float(v),6) for k,v in stds.items()})

    # Базовый sanity-check
    auc = baseline_auc(X, y)
    if auc is not None:
        print(f"Baseline LogisticRegression AUC (last 20%): {auc:.3f}")

    # OOF-прогноз LightGBM
    print("Training LightGBM (walk-forward)...")
    proba = oof_predict_lgbm(X, y, n_splits=5, random_state=RANDOM_STATE)
    print("Proba ready. Describe:\n", proba.describe())

    # Сглаживание вероятностей
    proba_s = smooth_proba(proba, args.smooth_span)
    if args.smooth_span and args.smooth_span > 1:
        print(f"Applied EMA smoothing span={args.smooth_span}. Proba_smoothed describe:\n", proba_s.describe())

    # ===== Подбор порогов на всей выборке
    best, best_stats, best_loose, best_loose_stats = search_thresholds(
        proba_s, df_feat['c'], args.timeframe,
        turnover_cap=args.turnover_cap,
        fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
    )

    if best is None and best_loose is None:
        print("\nНе удалось подобрать пороги (в т.ч. без ограничения по обороту).")
        print("Попробуй: --horizon 48 --smooth-span 24 --turnover-cap 0.02")
        return

    def save_common_outputs(stats, el, xl, es, xs, mh, cd, loose_flag=False):
        # Печать
        tag = "LOOSE, cap violated" if loose_flag else "OOF, turnover≤cap"
        print(f"\n=== Best thresholds ({tag}) ===")
        print(f"enter_long_q={el}, exit_long_q={xl}, enter_short_q={es}, exit_short_q={xs}, min_hold={mh}, cooldown={cd}")
        print(f"Thresholds:", stats['thresholds'])
        tp_bar = stats['turns']/max(1,stats['n_bars'])
        print("\n=== Backtest (OOF · hysteresis) ===")
        print(f"Bars: {stats['n_bars']}, Turns: {stats['turns']}  (avg {tp_bar:.3f} per bar{'' if loose_flag else f', cap={args.turnover_cap}'})")
        print(f"Shares (L/S/F): {stats['long_share']:.2%} / {stats['short_share']:.2%} / {stats['neutral_share']:.2%}")
        print(f"Sharpe: {stats['sharpe']:.2f}, CAGR: {stats['cagr']:.2%}, MaxDD: {stats['max_dd']:.2%}")

        # График и бенчмарк
        if args.plot and len(stats['equity']):
            plt.figure(figsize=(10,5))
            stats['equity'].plot(label='Strategy')
            eq_bh = (df_feat['c'].loc[stats['equity'].index] / df_feat['c'].loc[stats['equity'].index][0])
            eq_bh.plot(label='Buy & Hold')
            plt.legend(); plt.title(f"Equity (OOF · hysteresis){' · LOOSE' if loose_flag else ''} {args.symbol} {args.timeframe}")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

        # Сохранить equity/ret/pos
        out = pd.DataFrame({'equity': stats['equity'], 'ret': stats['ret'], 'pos': stats['pos']})
        out.to_csv("backtest_oof_results.csv"); print("Saved: backtest_oof_results.csv")

        # Сохранить thresholds для live/paper
        import datetime as dt
        to_save = {
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "horizon": args.horizon,
            "smooth_span": args.smooth_span,
            "turnover_cap": args.turnover_cap,
            "quantiles": {"enter_long_q": el, "exit_long_q": xl, "enter_short_q": es, "exit_short_q": xs},
            "thresholds": stats["thresholds"],
            "min_hold": mh,
            "cooldown": cd,
            "fees": {"fee_per_side": FEE_PER_SIDE, "slippage_per_side": SLIPPAGE_PER_SIDE},
            "performance": {"sharpe": stats['sharpe'], "cagr": stats['cagr'], "max_dd": stats['max_dd']},
            "turnover_cap_violated": bool(loose_flag)
        }
        with open("best_thresholds.json", "w") as f:
            json.dump(to_save, f, indent=2)
        print("Saved: best_thresholds.json")

        # --- Все сделки (CSV + JSON с net_pct) ---
        pos_exec = stats['pos'].shift(1).reindex(df_feat['c'].index).fillna(0).astype(int)
        trades = extract_trades_from_pos(df_feat['c'], pos_exec, FEE_PER_SIDE, SLIPPAGE_PER_SIDE)
        trades.to_csv("trades_all.csv", index=False); print("Saved: trades_all.csv")
        json_path = save_trades_json(trades, "trades_all.json",
                                     fields=["entry_ts","entry_price","exit_ts","exit_price","side","net_pct"])
        print("Saved:", json_path)
        print("\n=== Trade summary ===")
        print(summarize_trades(trades, args.timeframe))

        # --- Денежное эквити с депозитами/выводами (по всей истории stats) ---
        cflows = load_cashflows(args.cashflows)
        cflows_on_bars = align_cashflows_to_index(cflows, stats['ret'].index)
        cash_eq, dep, wd, final_value, profit, roi = equity_with_cashflows(stats['ret'], args.initial_capital, cflows_on_bars)
        pd.DataFrame({"value": cash_eq}).to_csv("equity_cashflow.csv")
        summary = {
            "initial_capital": args.initial_capital,
            "total_deposits": dep,
            "total_withdrawals": wd,
            "net_contributions": args.initial_capital + dep - wd,
            "final_value": final_value,
            "net_profit": profit,
            "roi": roi,
            "start_ts": str(cash_eq.index[0]) if len(cash_eq) else None,
            "end_ts": str(cash_eq.index[-1]) if len(cash_eq) else None
        }
        with open("cashflow_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved: equity_cashflow.csv, cashflow_summary.json")
        print(f"Cashflow-adjusted: initial={args.initial_capital:.2f}, deposits={dep:.2f}, withdrawals={wd:.2f} -> final={final_value:.2f}, profit={profit:.2f} (ROI {roi*100:.2f}%)")

    # Печать/сохранение для лучшего варианта
    if best is not None:
        (el, xl, es, xs, mh, cd) = best
        save_common_outputs(best_stats, el, xl, es, xs, mh, cd, loose_flag=False)
    else:
        (el, xl, es, xs, mh, cd) = best_loose
        save_common_outputs(best_loose_stats, el, xl, es, xs, mh, cd, loose_flag=True)

    # ===== Оценка за последние N дней (default 30), калибровка до окна, фикс-пороги в окне
    if args.last_days and args.last_days > 0:
        cutoff = df_feat.index.max() - pd.Timedelta(days=args.last_days)

        # 1) КАЛИБРОВКА только на истории ДО окна
        proba_hist  = proba_s.loc[proba_s.index <= cutoff]
        price_hist  = df_feat['c'].loc[df_feat.index <= cutoff]
        best_hist, stats_hist, best_loose_hist, stats_loose_hist = search_thresholds(
            proba_hist, price_hist, args.timeframe,
            turnover_cap=args.turnover_cap,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
        use_best = (best_hist is not None, best_loose_hist is not None)
        if not any(use_best):
            print(f"\n[Last {args.last_days}d] Не удалось откалибровать пороги на истории до окна.")
        else:
            if best_hist is not None:
                el, xl, es, xs, mh, cd = best_hist
                th = stats_hist["thresholds"]
            else:
                el, xl, es, xs, mh, cd = best_loose_hist
                th = stats_loose_hist["thresholds"]

            # 2) ТЕСТ только в последнем окне (фиксированные thresholds!)
            proba_last = proba_s.loc[proba_s.index > cutoff]
            price_last = df_feat['c'].loc[df_feat.index > cutoff]
            stats_last = backtest_hysteresis_fixed_thresholds(
                price_last, proba_last, args.timeframe,
                thresholds=th, min_hold=mh, cooldown=cd,
                fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
            )

            print(f"\n=== Last {args.last_days} days (fixed thresholds from prior history) ===")
            print(f"Bars: {stats_last['n_bars']}, Turns: {stats_last['turns']}, Shares L/S/F: "
                  f"{stats_last['long_share']:.1%}/{stats_last['short_share']:.1%}/{stats_last['neutral_share']:.1%}")
            print(f"Sharpe: {stats_last['sharpe']:.2f}, CAGR: {stats_last['cagr']:.2%}, MaxDD: {stats_last['max_dd']:.2%}")

            # --- Все сделки за окно
            pos_exec_last = stats_last['pos'].shift(1).reindex(price_last.index).fillna(0).astype(int)
            trades_last = extract_trades_from_pos(price_last, pos_exec_last, FEE_PER_SIDE, SLIPPAGE_PER_SIDE)
            trades_last.to_csv("trades_last_window.csv", index=False)
            _ = save_trades_json(trades_last, "trades_last_window.json",
                                 fields=["entry_ts","entry_price","exit_ts","exit_price","side","net_pct"])
            print("Saved: trades_last_window.csv, trades_last_window.json")
            print("\n--- Trade summary (last window) ---")
            print(summarize_trades(trades_last, args.timeframe))

            # --- Денежное эквити за окно
            cflows = load_cashflows(args.cashflows)
            cflows_on_bars = align_cashflows_to_index(cflows, stats_last['ret'].index)
            cash_eq, dep, wd, final_value, profit, roi = equity_with_cashflows(
                stats_last['ret'], args.initial_capital, cflows_on_bars
            )
            pd.DataFrame({"value": cash_eq}).to_csv("equity_cashflow_last_window.csv")
            with open("cashflow_summary_last_window.json","w") as f:
                json.dump({
                    "initial_capital": args.initial_capital,
                    "total_deposits": dep,
                    "total_withdrawals": wd,
                    "net_contributions": args.initial_capital + dep - wd,
                    "final_value": final_value,
                    "net_profit": profit,
                    "roi": roi,
                    "window_days": args.last_days,
                    "start_ts": str(cash_eq.index[0]) if len(cash_eq) else None,
                    "end_ts": str(cash_eq.index[-1]) if len(cash_eq) else None
                }, f, indent=2)
            print("Saved: equity_cashflow_last_window.csv, cashflow_summary_last_window.json")
            print(f"Last-window PnL: profit={profit:.2f} (ROI {roi*100:.2f}%), "
                  f"final_value={final_value:.2f}, deposits={dep:.2f}, withdrawals={wd:.2f}")

    # ===== OOT: калибруем на 80%, тестим на 20% (фиксируем пороги из калибровки)
    cut = int(len(proba_s) * 0.8)
    proba_cal, proba_test = proba_s.iloc[:cut], proba_s.iloc[cut:]
    price_cal, price_test = df_feat['c'].iloc[:cut], df_feat['c'].iloc[cut:]

    best_cal, stats_cal, *_ = search_thresholds(
        proba_cal, price_cal, args.timeframe,
        turnover_cap=args.turnover_cap,
        fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
    )
    if best_cal is not None:
        el, xl, es, xs, mh, cd = best_cal
        stats_oot = backtest_hysteresis(
            price_test, proba_test, args.timeframe,
            enter_long_q=el, exit_long_q=xl,
            enter_short_q=es, exit_short_q=xs,
            min_hold=mh, cooldown=cd,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
        print("\n=== OOT (20%) with fixed thresholds from first 80% ===")
        print(f"Sharpe: {stats_oot['sharpe']:.2f}, CAGR: {stats_oot['cagr']:.2%}, MaxDD: {stats_oot['max_dd']:.2%}, Turns/bar: {stats_oot['turns']/max(1,stats_oot['n_bars']):.3f}")
        if args.plot and len(stats_oot['equity']):
            plt.figure(figsize=(10,5))
            stats_oot['equity'].plot(label='Strategy (OOT)')
            eq_bh = (df_feat['c'].loc[stats_oot['equity'].index] / df_feat['c'].loc[stats_oot['equity'].index][0])
            eq_bh.plot(label='Buy & Hold')
            plt.legend(); plt.title("Equity OOT (fixed thresholds)")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    else:
        print("\nOOT: не удалось найти пороги на калибровочном участке в заданном лимите оборота.")
    
    # === Save final model for live inference ===
    from joblib import dump
    final_params = dict(
        n_estimators=2000, learning_rate=0.03, num_leaves=63, max_depth=-1,
        min_data_in_leaf=10, min_data_in_bin=1, min_gain_to_split=1e-8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='binary', class_weight='balanced', force_col_wise=True,
        verbosity=-1, random_state=RANDOM_STATE
    )
    final_model = LGBMClassifier(**final_params)
    final_model.fit(X, y)  # тренируем на всей истории
    dump({"model": final_model,
        "feat_cols": feat_cols,
        "smooth_span": args.smooth_span}, "final_model_lgbm.pkl")
    print("Saved: final_model_lgbm.pkl")

if __name__ == "__main__":
    # macOS/libomp helper (если lightgbm ругается на libomp, установи: brew install libomp)
    try:
        import platform, subprocess
        if platform.system() == "Darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
            prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
            os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}/lib"
    except Exception:
        pass
    main()