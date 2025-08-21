#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt

from datetime import datetime, timezone

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from joblib import dump

# skopt (optional, для Bayesian thresholds)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    _HAS_SKOPT = True
except Exception:
    _HAS_SKOPT = False

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)

# ----------------------------- Defaults -----------------------------
DEFAULT_SYMBOL = "SOL/USDT"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_MAX_BARS = 60000           # ~2-3 года на 15m
DEFAULT_HORIZON_BARS = 24           # горизонт цели (≈ 6ч для 15m)
DEFAULT_SMOOTH_SPAN = 12            # EMA сглаживание вероятностей
FEE_PER_SIDE = 0.0003               # комиссия на сторону (0.03%)
SLIPPAGE_PER_SIDE = 0.0001          # проскальзывание на сторону (0.01%)
RANDOM_STATE = 42
DEFAULT_TURNOVER_CAP = 0.05         # макс. поворотов на бар
DEFAULT_MAX_DD_CAP = 0.30           # макс. просадка для отбора порогов (улучшено)
DEFAULT_LAST_DAYS = 0               # окно "последние N дней", 0=выключено

# ----------------------------- Utils -----------------------------
def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith('m'): return int(tf[:-1])
    if tf.endswith('h'): return int(tf[:-1]) * 60
    if tf.endswith('d'): return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Неизвестный таймфрейм: {tf}")

def bars_per_year(tf: str) -> float:
    minutes = timeframe_to_minutes(tf)
    return (365.0 * 24.0 * 60.0) / minutes

def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0

def calculate_kelly_fraction(returns: pd.Series, win_rate: float = None, avg_win: float = None, avg_loss: float = None) -> float:
    """Вычисляет долю Kelly для оптимального размера позиции"""
    if len(returns) < 20:
        return 0.25  # консервативное значение по умолчанию
    
    if win_rate is None or avg_win is None or avg_loss is None:
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.25
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
    
    if avg_loss == 0:
        return 0.25
    
    b = avg_win / avg_loss  # odds
    p = win_rate
    kelly = (b * p - (1 - p)) / b
    
    # Ограничиваем Kelly для безопасности
    return max(0.1, min(0.5, kelly))

def calculate_volatility_adjustment(returns: pd.Series, lookback: int = 60) -> float:
    """Корректировка размера позиции на основе волатильности"""
    if len(returns) < lookback:
        return 1.0
    
    recent_vol = returns.tail(lookback).std()
    long_vol = returns.std()
    
    if long_vol == 0:
        return 1.0
    
    vol_ratio = recent_vol / long_vol
    # Если текущая волатильность выше - уменьшаем позицию
    return min(1.5, max(0.5, 1.0 / vol_ratio))

def calculate_dynamic_position_size(proba: float, returns_history: pd.Series, 
                                  base_size: float = 1.0, max_size: float = 1.0) -> float:
    """Динамический расчет размера позиции"""
    # Kelly fraction
    kelly = calculate_kelly_fraction(returns_history)
    
    # Volatility adjustment
    vol_adj = calculate_volatility_adjustment(returns_history)
    
    # Confidence adjustment (на основе вероятности)
    conf_adj = 1.0
    if proba > 0.7:
        conf_adj = 1.2  # увеличиваем при высокой уверенности
    elif proba < 0.3:
        conf_adj = 1.2  # для шортов
    elif 0.45 < proba < 0.55:
        conf_adj = 0.6  # уменьшаем при неопределенности
    
    dynamic_size = base_size * kelly * vol_adj * conf_adj
    return max(0.1, min(max_size, dynamic_size))

def drawdown_protection_filter(equity: pd.Series, threshold: float = -0.15) -> bool:
    """Фильтр защиты от просадок - возвращает False если нужно приостановить торговлю"""
    if len(equity) < 10:
        return True
    
    current_dd = max_drawdown(equity)
    return current_dd > threshold

def volatility_filter(prices: pd.Series, max_daily_vol: float = 0.08, lookback_days: int = 7) -> bool:
    """Фильтр волатильности - возвращает False если волатильность слишком высокая"""
    if len(prices) < lookback_days * 24:  # предполагаем часовые данные
        return True
    
    returns = prices.pct_change().dropna()
    recent_returns = returns.tail(lookback_days * 24)
    daily_vol = recent_returns.std() * np.sqrt(24)  # аннуализация для часовых данных
    
    return daily_vol <= max_daily_vol

def improved_min_hold_adaptive(volatility: float, base_hold: int = 24) -> int:
    """Адаптивное время удержания позиции на основе волатильности"""
    if volatility > 0.06:  # высокая волатильность
        return max(12, base_hold // 2)
    elif volatility < 0.02:  # низкая волатильность
        return min(72, base_hold * 2)
    else:
        return base_hold

def psi(a: pd.Series, b: pd.Series, bins: int = 20) -> float:
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    if len(a) < 100 or len(b) < 100:
        return np.nan
    qs = np.quantile(a, np.linspace(0, 1, bins + 1))
    qs[-1] += 1e-12
    c_a, _ = np.histogram(a, qs); c_b, _ = np.histogram(b, qs)
    denom_a = max(1, int(c_a.sum()))
    denom_b = max(1, int(c_b.sum()))
    c_a = np.clip(c_a / denom_a, 1e-9, None)
    c_b = np.clip(c_b / denom_b, 1e-9, None)
    return float(np.sum((c_a - c_b) * np.log(c_a / c_b)))

# ----------------------------- Data -----------------------------
def fetch_ohlcv_all(exchange, symbol, timeframe='15m',
                    since_ms: int | None = None, limit=1000, max_bars=60_000):
    tf2min = timeframe_to_minutes(timeframe)
    ms_step = tf2min * 60_000

    if since_ms is None:
        since_ms = int((pd.Timestamp.utcnow()
                        - pd.Timedelta(minutes=(max_bars + 5) * tf2min)).timestamp() * 1000)

    out, last_ts = [], None
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not chunk:
            break
        if last_ts is not None and chunk[-1][0] <= last_ts:
            break
        out += chunk
        last_ts = chunk[-1][0]
        since_ms = last_ts + ms_step
        if len(out) >= max_bars:
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
    d = d.rename(columns=str.lower)

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

def make_labels_binary(df_feat: pd.DataFrame, horizon_bars: int,
                       fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> pd.Series:
    """1 — если будущая доходность за H баров > полных издержек (вход+выход)"""
    fee_round_trip = 2.0 * (fee_per_side + slippage_per_side)
    fwd = df_feat['c'].shift(-horizon_bars) / df_feat['c'] - 1.0
    y = (fwd > fee_round_trip).astype(int)
    return y

# ----------------------------- CV: purged / embargo + fold-wise calibration -----------------------------
def purged_splits(n, n_splits=5, purge=24, embargo=12):
    kf = KFold(n_splits=n_splits, shuffle=False)
    for tr_idx, te_idx in kf.split(np.arange(n)):
        tr_idx = np.asarray(tr_idx, dtype=int)
        te_idx = np.asarray(te_idx, dtype=int)
        tr_mask = []
        te_start, te_end = te_idx[0], te_idx[-1]
        for i in tr_idx:
            if i <= te_start - purge or i > te_end:
                tr_mask.append(i)
        tr_mask = np.asarray(tr_mask, dtype=int)
        emb_end = min(n - 1, te_end + embargo)
        tr_mask = tr_mask[(tr_mask < te_start) | (tr_mask > emb_end)]
        yield tr_mask, te_idx

def _fit_lgbm(Xtr, ytr, Xvl, yvl, params):
    clf = LGBMClassifier(**params)
    clf.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
    p_val = clf.predict_proba(Xvl)[:, 1]
    return clf, p_val

def _calibrate_fold(p_val, y_val, method: str):
    if method == "none":
        return p_val
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_val, y_val.astype(float))
        return ir.transform(p_val)
    if method == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_val.reshape(-1,1), y_val.astype(int))
        return lr.predict_proba(p_val.reshape(-1,1))[:,1]
    return p_val

def oof_predict_lgbm_purged_foldcal(X: pd.DataFrame, y: pd.Series, n_splits=5,
                                    purge=24, embargo=12, method="none",
                                    random_state=RANDOM_STATE) -> pd.Series:
    """
    OOF вероятности с purged/embargo и калибровкой по каждому фолду (fold-wise).
    Исключает утечку, в отличие от глобальной «на последнем окне».
    """
    proba = pd.Series(index=X.index, dtype=float)
    params = dict(
        n_estimators=4000, learning_rate=0.03, num_leaves=127,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective='binary', class_weight='balanced',
        force_col_wise=True, verbosity=-1, random_state=random_state
    )
    n = len(X)
    for tr, te in purged_splits(n, n_splits=n_splits, purge=purge, embargo=embargo):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xvl, yvl = X.iloc[te], y.iloc[te]
        clf, p_val = _fit_lgbm(Xtr, ytr, Xvl, yvl, params)
        p_val_cal = _calibrate_fold(np.asarray(p_val), yvl.values, method)
        proba.iloc[te] = p_val_cal
    return proba

def baseline_auc(X: pd.DataFrame, y: pd.Series) -> float | None:
    """Быстрый sanity-check: логистическая регрессия на последних 20% данных."""
    if len(X) < 2000:
        return None
    n = len(X); cut = int(n * 0.8)
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
    state, hold, cd = 0, 0, 0
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

def positions_hysteresis_dynamic_optimized(proba: pd.Series, prices: pd.Series,
                                          enter_long: float, exit_long: float,
                                          enter_short: float, exit_short: float,
                                          min_hold: int = 24, cooldown: int = 12,
                                          use_dynamic_sizing: bool = True,
                                          max_position_size: float = 1.0,
                                          dd_protection_threshold: float = -0.15) -> tuple:
    """
    ОПТИМИЗИРОВАННАЯ версия с динамическим управлением позицией
    """
    state, hold, cd = 0, 0, 0
    pos, pos_sizes, protection_flags = [], [], []
    
    # Предвычисляем все возвраты
    returns_history = prices.pct_change().dropna()
    n_bars = len(proba)
    
    # Предвычисляем волатильность для каждого периода (каждые 20 баров)
    vol_cache = {}
    for i in range(0, n_bars, 20):
        end_idx = min(i + 60, len(returns_history))
        if end_idx > 60:
            vol_cache[i] = returns_history.iloc[max(0, end_idx-60):end_idx].std()
        else:
            vol_cache[i] = 0.02  # значение по умолчанию
    
    # Упрощенное отслеживание equity для защиты от просадок
    equity_simple = 1.0
    equity_history = []
    protection_check_interval = 10  # проверяем защиту каждые 10 баров
    
    for i, p in enumerate(proba.values):
        # Адаптивный min_hold (проверяем волатильность реже)
        vol_key = (i // 20) * 20
        if vol_key in vol_cache:
            adaptive_min_hold = improved_min_hold_adaptive(vol_cache[vol_key], min_hold)
        else:
            adaptive_min_hold = min_hold
        
        # Проверяем защиту от просадок реже
        if i % protection_check_interval == 0 and len(equity_history) > 10:
            current_dd = max_drawdown(pd.Series(equity_history[-50:]))  # только последние 50 значений
            protection_active = current_dd < dd_protection_threshold
        else:
            protection_active = protection_flags[-1] if protection_flags else False
        
        protection_flags.append(protection_active)
        
        # Основная логика позиций (упрощенная)
        if cd > 0:
            if state != 0:
                hold += 1
                if state == 1 and hold >= adaptive_min_hold and (p < exit_long or protection_active):
                    state, hold, cd = 0, 0, cooldown
                elif state == -1 and hold >= adaptive_min_hold and (p > exit_short or protection_active):
                    state, hold, cd = 0, 0, cooldown
            else:
                cd -= 1
        else:
            if state == 0 and not protection_active:
                if p > enter_long:
                    state, hold = 1, 0
                elif p < enter_short:
                    state, hold = -1, 0
            elif state == 1:
                hold += 1
                if hold >= adaptive_min_hold and (p < exit_long or protection_active):
                    state, hold, cd = 0, 0, cooldown
            elif state == -1:
                hold += 1
                if hold >= adaptive_min_hold and (p > exit_short or protection_active):
                    state, hold, cd = 0, 0, cooldown
        
        # Упрощенный размер позиции
        if use_dynamic_sizing and state != 0 and i > 30:
            # Используем предвычисленную волатильность и упрощенный Kelly
            recent_vol = vol_cache.get(vol_key, 0.02)
            vol_adj = min(1.5, max(0.5, 0.02 / recent_vol)) if recent_vol > 0 else 1.0
            
            # Упрощенная корректировка на уверенность
            if p > 0.7 or p < 0.3:
                conf_adj = 1.1
            elif 0.45 < p < 0.55:
                conf_adj = 0.7
            else:
                conf_adj = 1.0
                
            pos_size = min(max_position_size, 0.3 * vol_adj * conf_adj)  # упрощенная формула
        else:
            pos_size = 1.0 if state != 0 else 0.0
        
        pos.append(state)
        pos_sizes.append(pos_size * abs(state) if state != 0 else 0.0)
        
        # Упрощенное обновление equity
        if i > 0 and i < len(returns_history):
            if len(pos) > 1:
                prev_pos = pos[-2] * (pos_sizes[-2] if len(pos_sizes) > 1 else 1.0)
                current_return = returns_history.iloc[i]
                equity_simple *= (1 + prev_pos * current_return * 0.1)  # масштабируем для стабильности
            equity_history.append(equity_simple)
    
    pos_series = pd.Series(pos, index=proba.index, dtype=int)
    size_series = pd.Series(pos_sizes, index=proba.index, dtype=float)
    protection_series = pd.Series(protection_flags, index=proba.index, dtype=bool)
    
    return pos_series, size_series, protection_series

# ----------------------------- Backtests -----------------------------
def _align_by_open_next(df: pd.DataFrame, proba: pd.Series):
    df = df[['o','h','l','c']].copy()
    proba = proba.dropna()
    idx = df.index.intersection(proba.index)
    df = df.loc[idx]
    proba = proba.loc[idx]
    o = df['o'].astype(float)
    ret = (o.shift(-1) / o - 1.0).dropna()
    proba = proba.loc[ret.index]
    return df, ret, proba

def backtest_hysteresis_dynamic(df: pd.DataFrame, proba: pd.Series, tf: str,
                               enter_long_q=0.85, exit_long_q=0.60,  # улучшенные пороги
                               enter_short_q=0.15, exit_short_q=0.35,
                               min_hold=24, cooldown=12,
                               use_dynamic_sizing=True, max_position_size=0.8,
                               dd_protection_threshold=-0.15,
                               fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> dict:
    """Улучшенный бэктест с динамическим управлением позицией и защитой от просадок"""
    df, ret, proba = _align_by_open_next(df, proba)
    if len(ret) < 10:
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

    # Используем ОПТИМИЗИРОВАННУЮ логику позиций
    pos_raw, pos_sizes, protection_flags = positions_hysteresis_dynamic_optimized(
        proba, df['c'], enter_long, exit_long, enter_short, exit_short,
        min_hold=min_hold, cooldown=cooldown,
        use_dynamic_sizing=use_dynamic_sizing,
        max_position_size=max_position_size,
        dd_protection_threshold=dd_protection_threshold
    )
    
    # Позиции с учетом размера (для исполнения)
    pos_exec = (pos_raw * pos_sizes).shift(1).reindex(ret.index).fillna(0)

    turns = pos_exec.diff().abs().fillna(0)
    cost_per_side = float(fee_per_side + slippage_per_side)
    costs = turns * cost_per_side

    strat_ret = pos_exec * ret - costs
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
                    min_hold=min_hold, cooldown=cooldown,
                    use_dynamic_sizing=use_dynamic_sizing, max_position_size=max_position_size),
        equity=eq, ret=strat_ret, pos=pos_raw, pos_sizes=pos_sizes, protection_flags=protection_flags
    )

def backtest_hysteresis_open_next(df: pd.DataFrame, proba: pd.Series, tf: str,
                                  enter_long_q=0.90, exit_long_q=0.70,
                                  enter_short_q=0.10, exit_short_q=0.30,
                                  min_hold=24, cooldown=12,
                                  fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> dict:
    df, ret, proba = _align_by_open_next(df, proba)
    if len(ret) < 10:
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

    pos_raw = positions_hysteresis(proba, enter_long, exit_long, enter_short, exit_short,
                                   min_hold=min_hold, cooldown=cooldown)
    pos_exec = pos_raw.shift(1).reindex(ret.index).fillna(0).astype(int)

    turns = pos_exec.diff().abs().fillna(0)
    cost_per_side = float(fee_per_side + slippage_per_side)
    costs = turns * cost_per_side

    strat_ret = pos_exec * ret - costs
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

def backtest_hysteresis_fixed_thresholds_open(df: pd.DataFrame, proba: pd.Series, tf: str,
                                              thresholds: dict, min_hold=24, cooldown=12,
                                              fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> dict:
    df, ret, proba = _align_by_open_next(df, proba)
    if len(ret) < 10:
        return dict(sharpe=0.0, cagr=0.0, max_dd=0.0, n_bars=0, turns=0,
                    long_share=0.0, short_share=0.0, neutral_share=1.0,
                    thresholds=thresholds, params=dict(min_hold=min_hold, cooldown=cooldown),
                    equity=pd.Series(dtype=float), ret=pd.Series(dtype=float), pos=pd.Series(dtype=int))

    enter_long  = float(thresholds["enter_long"])
    exit_long   = float(thresholds["exit_long"])
    enter_short = float(thresholds["enter_short"])
    exit_short  = float(thresholds["exit_short"])

    pos_raw = positions_hysteresis(proba, enter_long, exit_long, enter_short, exit_short,
                                   min_hold=min_hold, cooldown=cooldown)
    pos_exec = pos_raw.shift(1).reindex(ret.index).fillna(0).astype(int)

    turns = pos_exec.diff().abs().fillna(0)
    cost_per_side = float(fee_per_side + slippage_per_side)
    costs = turns * cost_per_side

    strat_ret = pos_exec * ret - costs
    eq = (1.0 + strat_ret).cumprod()

    bpy = bars_per_year(tf)
    sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * np.sqrt(bpy) if len(strat_ret) else 0.0
    years = len(strat_ret) / bpy if len(strat_ret) else 0.0
    cagr = (eq.iloc[-1] ** (1/years) - 1.0) if years > 0 and len(eq) >= 2 else 0.0
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

# ----------------------------- Threshold search -----------------------------
def search_thresholds_dynamic(proba: pd.Series, df_feat: pd.DataFrame, tf: str,
                             turnover_cap=DEFAULT_TURNOVER_CAP, max_dd_cap=0.30,  # более строгий лимит
                             use_dynamic_sizing=True,
                             fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE):
    """Улучшенный поиск порогов с акцентом на снижение просадок и повышение Sharpe"""
    # Более консервативные и лучше сбалансированные сетки
    grid_enter = [0.82, 0.85, 0.88]            # Уменьшили количество для ускорения
    grid_exit  = [0.55, 0.60, 0.65]            # Больший спред с входом
    grid_enter_s = [0.18, 0.15, 0.12]          # Уменьшили количество для ускорения
    grid_exit_s  = [0.35, 0.40, 0.45]          # Больший спред с входом
    grid_hold = [16, 24, 32]                    # Меньше вариантов, фокус на оптимальных
    cooldowns = [12, 24]                        # Уменьшили для ускорения

    best, best_stats = None, None
    best_loose, best_loose_stats = None, None

    # Подсчет общего количества итераций для прогресса
    total_iterations = len(grid_enter) * len(grid_exit) * len(grid_enter_s) * len(grid_exit_s) * len(grid_hold) * len(cooldowns)
    current_iteration = 0
    
    print(f"🔍 Поиск оптимальных порогов: {total_iterations} комбинаций...")
    
    for el in grid_enter:
        for xl in grid_exit:
            if xl >= el: continue
            for es in grid_enter_s:
                for xs in grid_exit_s:
                    if xs <= es: continue
                    for mh in grid_hold:
                        for cd in cooldowns:
                            current_iteration += 1
                            
                            # Показываем прогресс каждые 10 итераций
                            if current_iteration % 10 == 0:
                                progress = (current_iteration / total_iterations) * 100
                                print(f"  ⏳ Прогресс: {progress:.1f}% ({current_iteration}/{total_iterations})")
                            
                            stats = backtest_hysteresis_dynamic(
                                df_feat, proba, tf,
                                enter_long_q=el, exit_long_q=xl,
                                enter_short_q=es, exit_short_q=xs,
                                min_hold=mh, cooldown=cd,
                                use_dynamic_sizing=use_dynamic_sizing,
                                max_position_size=0.8,  # Консервативный размер
                                dd_protection_threshold=-0.15,
                                fee_per_side=fee_per_side, slippage_per_side=slippage_per_side
                            )
                            if stats['n_bars'] == 0: continue
                            tp_bar = stats['turns'] / max(1, stats['n_bars'])
                            
                            # Комбинированный скор: Sharpe важнее, но с учетом просадки
                            combined_score = stats['sharpe'] * (1 + stats['max_dd'])  # max_dd отрицательный
                            
                            if (best_loose is None) or (combined_score > best_loose_stats.get('combined_score', -999)):
                                stats['combined_score'] = combined_score
                                best_loose, best_loose_stats = (el, xl, es, xs, mh, cd), stats
                                print(f"    ✨ Новый лучший (без ограничений): Sharpe={stats['sharpe']:.3f}, DD={stats['max_dd']:.1%}")
                            
                            if tp_bar <= turnover_cap and stats['max_dd'] >= -max_dd_cap:
                                if (best is None) or (combined_score > best_stats.get('combined_score', -999)):
                                    stats['combined_score'] = combined_score
                                    best, best_stats = (el, xl, es, xs, mh, cd), stats
                                    print(f"    🎯 Новый лучший (с ограничениями): Sharpe={stats['sharpe']:.3f}, DD={stats['max_dd']:.1%}")
    
    print("✅ Поиск порогов завершен!")
    return best, best_stats, best_loose, best_loose_stats

def search_thresholds_grid(proba: pd.Series, df_feat: pd.DataFrame, tf: str,
                           turnover_cap=DEFAULT_TURNOVER_CAP, max_dd_cap=DEFAULT_MAX_DD_CAP,
                           fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE):
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
            if xl >= el: continue
            for es in grid_enter_s:
                for xs in grid_exit_s:
                    if xs <= es: continue
                    for mh in grid_hold:
                        for cd in cooldowns:
                            stats = backtest_hysteresis_open_next(
                                df_feat, proba, tf,
                                enter_long_q=el, exit_long_q=xl,
                                enter_short_q=es, exit_short_q=xs,
                                min_hold=mh, cooldown=cd,
                                fee_per_side=fee_per_side, slippage_per_side=slippage_per_side
                            )
                            if stats['n_bars'] == 0: continue
                            tp_bar = stats['turns'] / max(1, stats['n_bars'])
                            if (best_loose is None) or (stats['sharpe'] > best_loose_stats['sharpe']):
                                best_loose, best_loose_stats = (el, xl, es, xs, mh, cd), stats
                            if tp_bar <= turnover_cap and stats['max_dd'] >= -max_dd_cap:
                                if (best is None) or (stats['sharpe'] > best_stats['sharpe']):
                                    best, best_stats = (el, xl, es, xs, mh, cd), stats
    return best, best_stats, best_loose, best_loose_stats

def search_thresholds_bayes(proba: pd.Series, df_feat: pd.DataFrame, tf: str,
                            turnover_cap=DEFAULT_TURNOVER_CAP, max_dd_cap=DEFAULT_MAX_DD_CAP,
                            n_calls=40,
                            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE):
    if not _HAS_SKOPT:
        print("WARN: skopt не найден — использую grid-поиск.")
        return search_thresholds_grid(proba, df_feat, tf, turnover_cap, max_dd_cap, fee_per_side, slippage_per_side)

    def objective(params):
        el, xl, es, xs, mh, cd = params
        stats = backtest_hysteresis_open_next(
            df_feat, proba, tf,
            enter_long_q=el, exit_long_q=xl, enter_short_q=es, exit_short_q=xs,
            min_hold=int(mh), cooldown=int(cd),
            fee_per_side=fee_per_side, slippage_per_side=slippage_per_side
        )
        if stats["n_bars"] == 0 or stats["max_dd"] < -max_dd_cap:
            return 1e3
        tp_bar = stats["turns"]/max(1,stats["n_bars"])
        penalty = 50.0 * max(0.0, tp_bar - turnover_cap)
        return float(-(stats["sharpe"]) + penalty)

    space = [
        Real(0.75, 0.95, name="enter_long_q"),
        Real(0.55, 0.75, name="exit_long_q"),
        Real(0.05, 0.25, name="enter_short_q"),
        Real(0.25, 0.45, name="exit_short_q"),
        Integer(12, 72,   name="min_hold"),
        Integer(2,  48,   name="cooldown"),
    ]
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=RANDOM_STATE)
    el, xl, es, xs, mh, cd = res.x
    best_stats = backtest_hysteresis_open_next(
        df_feat, proba, tf, el, xl, es, xs, int(mh), int(cd),
        fee_per_side=fee_per_side, slippage_per_side=slippage_per_side
    )
    # для "loose" вернём то же (можно добавить пост-грid добор при желании)
    return (el, xl, es, xs, int(mh), int(cd)), best_stats, (el, xl, es, xs, int(mh), int(cd)), best_stats

# ----------------------------- Trades (open-exec) & JSON -----------------------------
def extract_trades_from_pos_open(df: pd.DataFrame, pos_exec: pd.Series,
                                 fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE) -> pd.DataFrame:
    df = df[['o','c']].copy()
    idx = df.index.intersection(pos_exec.index)
    if len(idx) == 0:
        return pd.DataFrame(columns=['side','entry_ts','exit_ts','entry_price','exit_price','gross_ret','net_ret','net_pct','bars','win'])

    o = df.loc[idx, 'o'].astype(float)
    ps = pos_exec.loc[idx].astype(int)

    cost_side = float(fee_per_side + slippage_per_side)
    trades = []; cur = None

    for i, ts in enumerate(idx):
        p = int(ps.iloc[i]); price_open = float(o.iloc[i])

        if cur is None:
            if p != 0:
                cur = dict(side=p, entry_ts=ts, entry_price=price_open, entry_i=i)
            continue

        if p == cur["side"]:
            continue

        side = cur["side"]; entry_price = float(cur["entry_price"])
        exit_ts = ts; exit_price = price_open

        gross = (exit_price / entry_price - 1.0) if side == 1 else (entry_price / exit_price - 1.0)
        net = (1.0 + gross) * (1.0 - cost_side)**2 - 1.0
        bars_held = i - int(cur["entry_i"])

        trades.append(dict(
            side="long" if side == 1 else "short",
            entry_ts=cur["entry_ts"], exit_ts=exit_ts,
            entry_price=entry_price, exit_price=exit_price,
            gross_ret=gross, net_ret=net, net_pct=net*100.0, bars=bars_held
        ))

        cur = None
        if p != 0:
            cur = dict(side=p, entry_ts=ts, entry_price=price_open, entry_i=i)

    if cur is not None and len(idx):
        ts = idx[-1]; price_open = float(o.iloc[-1])
        side = cur["side"]; entry_price = float(cur["entry_price"])
        gross = (price_open / entry_price - 1.0) if side == 1 else (entry_price / price_open - 1.0)
        net = (1.0 + gross) * (1.0 - cost_side)**2 - 1.0
        bars_held = len(idx) - 1 - int(cur["entry_i"])
        trades.append(dict(
            side="long" if side == 1 else "short",
            entry_ts=cur["entry_ts"], exit_ts=ts,
            entry_price=entry_price, exit_price=price_open,
            gross_ret=gross, net_ret=net, net_pct=net*100.0, bars=bars_held
        ))

    df_tr = pd.DataFrame(trades)
    if not df_tr.empty:
        df_tr["win"] = df_tr["net_ret"] > 0
        df_tr["entry_ts"] = pd.to_datetime(df_tr["entry_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_tr["exit_ts"]  = pd.to_datetime(df_tr["exit_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df_tr

def save_trades_json(trades: pd.DataFrame, path="trades_all.json", fields=None) -> str:
    if trades is None or trades.empty:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return path
    if fields is None:
        fields = ["entry_ts","entry_price","exit_ts","exit_price","side","net_pct","bars","win"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trades[fields].to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return path

# ----------------------------- Cashflows (депозиты/выводы) -----------------------------
def load_cashflows(path: str) -> pd.Series:
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
    if cflows is None or cflows.empty or len(bar_index) == 0:
        return pd.Series(dtype=float)
    mapped = {}
    for ts, amt in cflows.items():
        if ts in bar_index:
            mapped[ts] = mapped.get(ts, 0.0) + float(amt)
        else:
            pos = bar_index.searchsorted(ts)
            if pos < len(bar_index):
                key = bar_index[pos]
                mapped[key] = mapped.get(key, 0.0) + float(amt)
    if not mapped:
        return pd.Series(dtype=float)
    return pd.Series(mapped).sort_index()

def equity_with_cashflows(strat_ret: pd.Series, initial_capital: float, cflows_on_bars: pd.Series) -> tuple:
    value = float(initial_capital)
    vals = []
    for ts in strat_ret.index:
        if (cflows_on_bars is not None) and (ts in cflows_on_bars.index):
            value += float(cflows_on_bars.loc[ts])
        value *= (1.0 + float(strat_ret.loc[ts]))
        vals.append((ts, value))
    series = pd.Series([v for _, v in vals], index=[t for t, _ in vals]) if vals else pd.Series(dtype=float)
    dep = float(cflows_on_bars[cflows_on_bars > 0].sum()) if cflows_on_bars is not None and len(cflows_on_bars) else 0.0
    wd  = float(-cflows_on_bars[cflows_on_bars < 0].sum()) if cflows_on_bars is not None and len(cflows_on_bars) else 0.0
    final_value = float(series.iloc[-1]) if len(series) else initial_capital + dep - wd
    net_contrib = float(initial_capital + dep - wd)
    profit = final_value - net_contrib
    roi = (profit / net_contrib) if net_contrib > 0 else np.nan
    return series, dep, wd, final_value, profit, roi

# ----------------------------- Summary -----------------------------
def summarize_trades(trades: pd.DataFrame, timeframe: str) -> str:
    if trades is None or trades.empty:
        return "Сделок не найдено."

    total = len(trades)
    wins = trades[trades["win"]]
    losses = trades[~trades["win"]]

    def pct(x): return 100.0 * float(x)

    winrate = trades["win"].mean() if len(trades) else 0.0
    pf = (wins["net_ret"].sum() / abs(losses["net_ret"].sum())) if len(losses) and abs(losses["net_ret"].sum()) > 1e-12 else np.inf
    exp = trades["net_ret"].mean()

    mins = timeframe_to_minutes(timeframe)
    avg_bars = trades["bars"].mean() if len(trades) else 0.0
    avg_hours = avg_bars * mins / 60.0

    wl = trades["win"].astype(int).tolist()
    max_w, max_l, cur_w, cur_l = 0, 0, 0, 0
    for x in wl:
        if x == 1:
            cur_w += 1; max_w = max(max_w, cur_w); cur_l = 0
        else:
            cur_l += 1; max_l = max(max_l, cur_l); cur_w = 0

    lines = []
    lines.append(f"Всего сделок: {total}")
    lines.append(f"Winrate: {pct(winrate):.1f}%")
    lines.append(f"Profit factor: {pf:.2f}  |  Expectancy/сделка: {pct(exp):.2f}%")
    lines.append(f"Средняя длительность: {avg_bars:.1f} баров (~{avg_hours:.1f} ч)")
    lines.append(f"Серии — побед: {max_w} | убыточных: {max_l}")
    lines.append(f"Средняя прибыльная: {pct(wins['net_ret'].mean() if len(wins) else 0):.2f}%  |  Средняя убыточная: {pct(losses['net_ret'].mean() if len(losses) else 0):.2f}%")
    return "\n".join(lines)

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="AI crypto bot (LGBM, purged CV + fold-wise calibration, open-next exec, hysteresis, thresholds, trades JSON, cashflows, last-window/OOT)")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME)
    parser.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_BARS, help="target horizon in bars")
    parser.add_argument("--smooth-span", type=int, default=DEFAULT_SMOOTH_SPAN, help="EMA span for proba smoothing (bars)")
    parser.add_argument("--turnover-cap", type=float, default=DEFAULT_TURNOVER_CAP, help="max turns per bar")
    parser.add_argument("--max-dd-cap", type=float, default=DEFAULT_MAX_DD_CAP, help="abs(MaxDD) limit to accept thresholds (e.g. 0.40=40%)")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="стартовый капитал для денежного эквити")
    parser.add_argument("--cashflows", type=str, default="cashflows.json", help="путь к JSON с кэшфлоу [{'ts': 'YYYY-MM-DD HH:MM:SS', 'amount': 1000}, ...]")
    parser.add_argument("--last-days", type=int, default=DEFAULT_LAST_DAYS, help="оценить результат только за последние N дней (0=выкл)")

    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-purge", type=int, default=None, help="purge bars (по умолчанию = horizon)")
    parser.add_argument("--cv-embargo", type=int, default=None, help="embargo bars (по умолчанию = horizon//2)")
    parser.add_argument("--calibrate", type=str, default="none", choices=["none","isotonic","platt"], help="fold-wise калибровка вероятностей")

    parser.add_argument("--search", type=str, default="dynamic", choices=["grid","bayes","dynamic"], help="поиск порогов (по умолчанию: dynamic - улучшенный алгоритм)")
    parser.add_argument("--use-dynamic-sizing", action="store_true", default=True, help="использовать динамическое управление размером позиции")
    parser.add_argument("--max-position-size", type=float, default=0.8, help="максимальный размер позиции (доля от капитала)")
    parser.add_argument("--dd-protection", type=float, default=-0.15, help="порог защиты от просадок")
    parser.add_argument("--bayes-calls", type=int, default=40, help="итерации для bayes-поиска (skopt)")

    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    purge = args.cv_purge if args.cv_purge is not None else args.horizon
    embargo = args.cv_embargo if args.cv_embargo is not None else max(1, args.horizon // 2)

    print(f"Downloading: {args.symbol} {args.timeframe} (max {args.max_bars} bars)")
    ex = ccxt.binance(); ex.enableRateLimit = True

    df = fetch_ohlcv_all(ex, args.symbol, timeframe=args.timeframe, max_bars=args.max_bars)
    df = df.rename(columns=str.lower)
    if len(df) < 5_000:
        print(f"WARNING: bars={len(df)} — мало данных. Рекомендуется >= 10–20k.")
    print("Raw bars:", len(df))

    df_feat = build_features(df)
    y = make_labels_binary(df_feat, args.horizon, FEE_PER_SIDE, SLIPPAGE_PER_SIDE)

    feat_cols = ['ret1','ret3','ret5','ret20','rsi','sma_ratio','macd','macd_sig','macd_diff','bb_pos','vol20']
    X = df_feat[feat_cols].shift(1).dropna()  # shift(1) — защита от утечки
    y = y.loc[X.index]

    # Самодиагностика
    print("X shape:", X.shape, "  y balance (mean of class=1):", round(float(y.mean()), 3))
    nunique = X.nunique().to_dict()
    stds = X.std().to_dict()
    print("Unique per feature:", {k:int(v) for k,v in nunique.items()})
    print("Std per feature    :", {k:round(float(v),6) for k,v in stds.items()})

    # Базовый sanity-check
    auc = baseline_auc(X, y)
    if auc is not None:
        print(f"Baseline LogisticRegression AUC (last 20%): {auc:.3f}")

    # OOF-прогноз LightGBM (purged/embargo) + FOLD-WISE CALIBRATION
    print(f"Training LightGBM OOF (purged CV) ... purge={purge}, embargo={embargo}, splits={args.cv_splits}, calibration={args.calibrate}")
    proba = oof_predict_lgbm_purged_foldcal(
        X, y, n_splits=args.cv_splits, purge=purge, embargo=embargo, method=args.calibrate, random_state=RANDOM_STATE
    )
    print("Proba ready. Describe:\n", proba.describe())

    # Сглаживание вероятностей
    proba_s = proba.ewm(span=args.smooth_span, adjust=False).mean() if (args.smooth_span and args.smooth_span > 1) else proba
    if args.smooth_span and args.smooth_span > 1:
        print(f"Applied EMA smoothing span={args.smooth_span}. Proba_smoothed describe:\n", proba_s.describe())

    # PSI дрифт последнего месяца vs остальная история
    try:
        cut_psi = df_feat.index.max() - pd.Timedelta(days=30)
        psi_proba = psi(proba_s[proba_s.index <= cut_psi], proba_s[proba_s.index > cut_psi])
        print("PSI(proba last30d vs history):", round(psi_proba, 3))
    except Exception:
        pass

    # ===== Подбор порогов (на всей истории)
    if args.search == "bayes":
        best, best_stats, best_loose, best_loose_stats = search_thresholds_bayes(
            proba_s, df_feat, args.timeframe,
            turnover_cap=args.turnover_cap, max_dd_cap=args.max_dd_cap,
            n_calls=args.bayes_calls,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
    elif args.search == "dynamic":
        print("🚀 Используем улучшенный алгоритм с динамическим размером позиции и защитой от просадок...")
        best, best_stats, best_loose, best_loose_stats = search_thresholds_dynamic(
            proba_s, df_feat, args.timeframe,
            turnover_cap=args.turnover_cap, max_dd_cap=0.30,  # более строгий лимит
            use_dynamic_sizing=args.use_dynamic_sizing,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
    else:
        best, best_stats, best_loose, best_loose_stats = search_thresholds_grid(
            proba_s, df_feat, args.timeframe,
            turnover_cap=args.turnover_cap, max_dd_cap=args.max_dd_cap,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )

    if best is None and best_loose is None:
        print("\nНе удалось подобрать пороги (в т.ч. без ограничений).")
        print("Попробуй: --horizon 48 --smooth-span 24 --turnover-cap 0.02 --max-dd-cap 0.50")
        return

    def save_common_outputs(stats, el, xl, es, xs, mh, cd, loose_flag=False):
        tag = "LOOSE, cap violated" if loose_flag else "OOF, turnover≤cap & DD≤cap"
        print(f"\n=== Best thresholds ({tag}) ===")
        print(f"enter_long_q={el}, exit_long_q={xl}, enter_short_q={es}, exit_short_q={xs}, min_hold={mh}, cooldown={cd}")
        print("Thresholds:", stats['thresholds'])
        tp_bar = stats['turns']/max(1,stats['n_bars'])
        print("\n=== Backtest (OOF · hysteresis · open-next) ===")
        print(f"Bars: {stats['n_bars']}, Turns: {stats['turns']}  (avg {tp_bar:.3f} per bar{'' if loose_flag else f', cap={args.turnover_cap}'})")
        print(f"Shares (L/S/F): {stats['long_share']:.2%} / {stats['short_share']:.2%} / {stats['neutral_share']:.2%}")
        print(f"Sharpe: {stats['sharpe']:.2f}, CAGR: {stats['cagr']:.2%}, MaxDD: {stats['max_dd']:.2%}")

        if args.plot and len(stats['equity']):
            plt.figure(figsize=(10,5))
            stats['equity'].plot(label='Strategy')
            eq_bh = (df_feat['c'].loc[stats['equity'].index] / df_feat['c'].loc[stats['equity'].index][0])
            eq_bh.plot(label='Buy & Hold')
            plt.legend(); plt.title(f"Equity (OOF · open-next){' · LOOSE' if loose_flag else ''} {args.symbol} {args.timeframe}")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

        out = pd.DataFrame({'equity': stats['equity'], 'ret': stats['ret'], 'pos': stats['pos']})
        out.to_csv("backtest_oof_results.csv"); print("Saved: backtest_oof_results.csv")

        to_save = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "horizon": args.horizon,
            "smooth_span": args.smooth_span,
            "turnover_cap": args.turnover_cap,
            "max_dd_cap": args.max_dd_cap,
            "calibration": args.calibrate,
            "search": args.search,
            "bayes_calls": args.bayes_calls if args.search=="bayes" else None,
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

        pos_exec = stats['pos'].shift(1).reindex(df_feat.index).fillna(0).astype(int)
        trades = extract_trades_from_pos_open(df_feat[['o','c']], pos_exec, FEE_PER_SIDE, SLIPPAGE_PER_SIDE)
        trades.to_csv("trades_all.csv", index=False); print("Saved: trades_all.csv")
        json_path = save_trades_json(trades, "trades_all.json")
        print("Saved:", json_path)
        print("\n=== Trade summary ===")
        print(summarize_trades(trades, args.timeframe))

        o = df_feat['o'].reindex(pos_exec.index).astype(float)
        ret_bar = (o.shift(-1) / o - 1.0).dropna()
        pos_e = pos_exec.reindex(ret_bar.index).fillna(0).astype(int)
        turns = pos_e.diff().abs().fillna(0)
        cost_side = float(FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
        costs = turns * cost_side
        strat_ret = pos_e * ret_bar - costs

        cflows = load_cashflows(args.cashflows)
        cflows_on_bars = align_cashflows_to_index(cflows, strat_ret.index)
        cash_eq, dep, wd, final_value, profit, roi = equity_with_cashflows(strat_ret, args.initial_capital, cflows_on_bars)
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

    if best is not None:
        (el, xl, es, xs, mh, cd) = best
        save_common_outputs(best_stats, el, xl, es, xs, mh, cd, loose_flag=False)
    else:
        (el, xl, es, xs, mh, cd) = best_loose
        save_common_outputs(best_loose_stats, el, xl, es, xs, mh, cd, loose_flag=True)

    # ===== Оценка за последние N дней: калибруем до окна, фикс-пороги в окне
    if args.last_days and args.last_days > 0:
        cutoff = df_feat.index.max() - pd.Timedelta(days=args.last_days)

        proba_hist = proba_s.loc[proba_s.index <= cutoff]
        feat_hist  = df_feat.loc[df_feat.index <= cutoff]

        if args.search == "bayes":
            best_hist, stats_hist, best_loose_hist, stats_loose_hist = search_thresholds_bayes(
                proba_hist, feat_hist, args.timeframe,
                turnover_cap=args.turnover_cap, max_dd_cap=args.max_dd_cap,
                n_calls=args.bayes_calls,
                fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
            )
        elif args.search == "dynamic":
            best_hist, stats_hist, best_loose_hist, stats_loose_hist = search_thresholds_dynamic(
                proba_hist, feat_hist, args.timeframe,
                turnover_cap=args.turnover_cap, max_dd_cap=0.30,
                use_dynamic_sizing=args.use_dynamic_sizing,
                fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
            )
        else:
            best_hist, stats_hist, best_loose_hist, stats_loose_hist = search_thresholds_grid(
                proba_hist, feat_hist, args.timeframe,
                turnover_cap=args.turnover_cap, max_dd_cap=args.max_dd_cap,
                fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
            )

        if best_hist is None and best_loose_hist is None:
            print(f"\n[Last {args.last_days}d] Не удалось откалибровать пороги на истории до окна.")
        else:
            if best_hist is not None:
                el, xl, es, xs, mh, cd = best_hist
                th = stats_hist["thresholds"]
            else:
                el, xl, es, xs, mh, cd = best_loose_hist
                th = stats_loose_hist["thresholds"]

            proba_last = proba_s.loc[proba_s.index > cutoff]
            feat_last  = df_feat.loc[df_feat.index > cutoff]
            stats_last = backtest_hysteresis_fixed_thresholds_open(
                feat_last, proba_last, args.timeframe,
                thresholds=th, min_hold=mh, cooldown=cd,
                fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
            )

            print(f"\n=== Last {args.last_days} days (fixed thresholds from prior history) ===")
            print(f"Bars: {stats_last['n_bars']}, Turns: {stats_last['turns']}, Shares L/S/F: "
                  f"{stats_last['long_share']:.1%}/{stats_last['short_share']:.1%}/{stats_last['neutral_share']:.1%}")
            print(f"Sharpe: {stats_last['sharpe']:.2f}, CAGR: {stats_last['cagr']:.2%}, MaxDD: {stats_last['max_dd']:.2%}")

            pos_exec_last = stats_last['pos'].shift(1).reindex(feat_last.index).fillna(0).astype(int)
            trades_last = extract_trades_from_pos_open(feat_last[['o','c']], pos_exec_last, FEE_PER_SIDE, SLIPPAGE_PER_SIDE)
            trades_last.to_csv("trades_last_window.csv", index=False)
            _ = save_trades_json(trades_last, "trades_last_window.json")
            print("Saved: trades_last_window.csv, trades_last_window.json")
            print("\n--- Trade summary (last window) ---")
            print(summarize_trades(trades_last, args.timeframe))

            o_last = feat_last['o'].reindex(pos_exec_last.index).astype(float)
            ret_last = (o_last.shift(-1) / o_last - 1.0).dropna()
            pos_e_last = pos_exec_last.reindex(ret_last.index).fillna(0).astype(int)
            turns_last = pos_e_last.diff().abs().fillna(0)
            cost_side = float(FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
            costs_last = turns_last * cost_side
            strat_ret_last = pos_e_last * ret_last - costs_last

            cflows = load_cashflows(args.cashflows)
            cflows_on_bars = align_cashflows_to_index(cflows, strat_ret_last.index)
            cash_eq, dep, wd, final_value, profit, roi = equity_with_cashflows(strat_ret_last, args.initial_capital, cflows_on_bars)
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

    # ===== OOT: калибруем пороги на 80%, тестим на 20% (фиксируем пороги из калибровки)
    cut = int(len(proba_s) * 0.8)
    proba_calib, proba_test = proba_s.iloc[:cut], proba_s.iloc[cut:]
    feat_calib, feat_test   = df_feat.iloc[:cut], df_feat.iloc[cut:]

    if args.search == "bayes":
        best_cal, stats_cal, *_ = search_thresholds_bayes(
            proba_calib, feat_calib, args.timeframe,
            turnover_cap=args.turnover_cap, max_dd_cap=args.max_dd_cap,
            n_calls=args.bayes_calls,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
    elif args.search == "dynamic":
        best_cal, stats_cal, *_ = search_thresholds_dynamic(
            proba_calib, feat_calib, args.timeframe,
            turnover_cap=args.turnover_cap, max_dd_cap=0.30,
            use_dynamic_sizing=args.use_dynamic_sizing,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
    else:
        best_cal, stats_cal, *_ = search_thresholds_grid(
            proba_calib, feat_calib, args.timeframe,
            turnover_cap=args.turnover_cap, max_dd_cap=args.max_dd_cap,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )

    if best_cal is not None:
        el, xl, es, xs, mh, cd = best_cal
        stats_oot = backtest_hysteresis_open_next(
            feat_test, proba_test, args.timeframe,
            enter_long_q=el, exit_long_q=xl, enter_short_q=es, exit_short_q=xs,
            min_hold=mh, cooldown=cd,
            fee_per_side=FEE_PER_SIDE, slippage_per_side=SLIPPAGE_PER_SIDE
        )
        print("\n=== OOT (20%) with fixed thresholds from first 80% (open-next) ===")
        print(f"Sharpe: {stats_oot['sharpe']:.2f}, CAGR: {stats_oot['cagr']:.2%}, MaxDD: {stats_oot['max_dd']:.2%}, Turns/bar: {stats_oot['turns']/max(1,stats_oot['n_bars']):.3f}")
        if args.plot and len(stats_oot['equity']):
            plt.figure(figsize=(10,5))
            stats_oot['equity'].plot(label='Strategy (OOT)')
            eq_bh = (df_feat['c'].loc[stats_oot['equity'].index] / df_feat['c'].loc[stats_oot['equity'].index][0])
            eq_bh.plot(label='Buy & Hold')
            plt.legend(); plt.title("Equity OOT (fixed thresholds, open-next)")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    else:
        print("\nOOT: не удалось найти пороги на калибровочном участке в заданных лимитах.")

    # === Save final model for live inference ===
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
          "feat_cols": list(X.columns),
          "smooth_span": args.smooth_span}, "final_model_lgbm.pkl")
    print("Saved: final_model_lgbm.pkl")

if __name__ == "__main__":
    # macOS/libomp helper (если lightgbm ругается на libomp: brew install libomp)
    try:
        import platform, subprocess
        if platform.system() == "Darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
            prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
            os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}/lib"
    except Exception:
        pass
    main()
