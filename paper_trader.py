#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, math, argparse, warnings, datetime as dt
import numpy as np
import pandas as pd
import ccxt
import traceback

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
pd.set_option("display.max_columns", 200)

STATE_FILE = "paper_state.json"
TRADES_CSV = "paper_trades.csv"
EQUITY_CSV = "paper_equity.csv"
THRESHOLDS_JSON = "best_thresholds.json"

# ----------------------------- Utils -----------------------------
def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unknown timeframe: {tf}")

def bars_per_year(tf: str) -> float:
    minutes = timeframe_to_minutes(tf)
    return (365.0 * 24.0 * 60.0) / minutes

def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def ceil_to_next_bar(ts: dt.datetime, timeframe: str) -> dt.datetime:
    # Round up to the next bar boundary
    minutes = timeframe_to_minutes(timeframe)
    epoch_min = int(ts.timestamp() // 60)
    next_bucket = ((epoch_min // minutes) + 1) * minutes
    return dt.datetime.fromtimestamp(next_bucket * 60, tz=dt.timezone.utc)

def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    roll = equity.cummax()
    dd = equity / roll - 1.0
    return float(dd.min())

# ----------------------------- Data -----------------------------
def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df.sort_index()
    return df

# ----------------------------- Features & Labels -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ret1"]  = d["c"].pct_change()
    d["ret3"]  = d["c"].pct_change(3)
    d["ret5"]  = d["c"].pct_change(5)
    d["ret20"] = d["c"].pct_change(20)

    d["rsi"] = RSIIndicator(d["c"], window=14).rsi()

    d["sma20"] = SMAIndicator(d["c"], 20).sma_indicator()
    d["sma50"] = SMAIndicator(d["c"], 50).sma_indicator()
    d["sma_ratio"] = d["sma20"] / d["sma50"] - 1

    macd = MACD(d["c"])
    d["macd"]      = macd.macd()
    d["macd_sig"]  = macd.macd_signal()
    d["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(d["c"], window=20, window_dev=2.0)
    rng = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    d["bb_pos"] = (d["c"] - bb.bollinger_mavg()) / rng

    d["vol20"] = d["ret1"].rolling(20).std()

    d = d.dropna()
    return d

def make_labels(df: pd.DataFrame, horizon_bars: int, fee_per_side: float, slippage_per_side: float) -> pd.Series:
    fee_round_trip = 2.0 * (fee_per_side + slippage_per_side)
    fwd = df["c"].shift(-horizon_bars) / df["c"] - 1.0
    y = (fwd > fee_round_trip).astype(int)
    return y

# ----------------------------- Modeling -----------------------------
def oof_predict_lgbm(X: pd.DataFrame, y: pd.Series, n_splits=5, random_state=42) -> pd.Series:
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
        objective="binary",
        class_weight="balanced",
        force_col_wise=True,
        verbosity=-1,
        random_state=random_state
    )
    for tr, te in tscv.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xvl, yvl = X.iloc[te], y.iloc[te]
        clf = LGBMClassifier(**params)
        clf.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], eval_metric="binary_logloss")
        proba.iloc[te] = clf.predict_proba(Xvl)[:, 1]
    return proba

def smooth_proba(proba: pd.Series, span: int) -> pd.Series:
    if span is None or span <= 1:
        return proba.dropna()
    return proba.ewm(span=span, adjust=False).mean().dropna()

# ----------------------------- Hysteresis -----------------------------
def positions_hysteresis(proba: pd.Series,
                         enter_long: float, exit_long: float,
                         enter_short: float, exit_short: float,
                         min_hold: int = 24, cooldown: int = 12) -> pd.Series:
    state = 0
    hold = 0
    cd = 0
    pos_vals = []
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
        pos_vals.append(state)
    return pd.Series(pos_vals, index=proba.index, dtype=int)

def align_ret_and_proba(prices: pd.Series, proba: pd.Series):
    proba = proba.dropna()
    idx = proba.index.intersection(prices.index)
    if len(idx) < 3:
        return None, None
    prices = prices.loc[idx]
    proba = proba.loc[idx]
    ret = prices.pct_change().dropna()
    proba = proba.loc[ret.index]
    return ret, proba

# ----------------------------- Paper Engine -----------------------------
def load_thresholds(path: str):
    with open(path, "r") as f:
        cfg = json.load(f)
    th = cfg.get("thresholds", {})
    q = cfg.get("quantiles", {})
    params = {
        "symbol": cfg.get("symbol", "BTC/USDT"),
        "timeframe": cfg.get("timeframe", "15m"),
        "horizon": int(cfg.get("horizon", 24)),
        "smooth_span": int(cfg.get("smooth_span", 12)),
        "min_hold": int(cfg.get("min_hold", 24)),
        "cooldown": int(cfg.get("cooldown", 12)),
        "fee_per_side": float(cfg.get("fees", {}).get("fee_per_side", 0.0003)),
        "slippage_per_side": float(cfg.get("fees", {}).get("slippage_per_side", 0.0001)),
        "enter_long": float(th.get("enter_long")) if th.get("enter_long") is not None else None,
        "exit_long": float(th.get("exit_long")) if th.get("exit_long") is not None else None,
        "enter_short": float(th.get("enter_short")) if th.get("enter_short") is not None else None,
        "exit_short": float(th.get("exit_short")) if th.get("exit_short") is not None else None,
        "enter_long_q": float(q.get("enter_long_q", 0.90)),
        "exit_long_q": float(q.get("exit_long_q", 0.70)),
        "enter_short_q": float(q.get("enter_short_q", 0.10)),
        "exit_short_q": float(q.get("exit_short_q", 0.30)),
    }
    return params

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def load_state(default_equity: float = 1.0) -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
            return s
        except Exception:
            pass
    return {
        "last_ts": None,
        "last_pos": 0,
        "equity": default_equity
    }

def ensure_csv_headers():
    if not os.path.exists(TRADES_CSV):
        pd.DataFrame(columns=[
            "ts","symbol","side","pos_prev","pos_new","price","note"
        ]).to_csv(TRADES_CSV, index=False)
    if not os.path.exists(EQUITY_CSV):
        pd.DataFrame(columns=["ts","equity"]).to_csv(EQUITY_CSV, index=False)

def log_trade(ts: pd.Timestamp, symbol: str, side: str, pos_prev: int, pos_new: int, price: float, note: str):
    row = pd.DataFrame([{
        "ts": ts.isoformat(),
        "symbol": symbol,
        "side": side,
        "pos_prev": int(pos_prev),
        "pos_new": int(pos_new),
        "price": float(price),
        "note": note
    }])
    row.to_csv(TRADES_CSV, mode="a", header=False, index=False)

def log_equity(ts: pd.Timestamp, equity: float):
    row = pd.DataFrame([{"ts": ts.isoformat(), "equity": float(equity)}])
    row.to_csv(EQUITY_CSV, mode="a", header=False, index=False)

def backtest_step(prices: pd.Series, proba: pd.Series, tf: str,
                  enter_long=None, exit_long=None, enter_short=None, exit_short=None,
                  enter_long_q=0.90, exit_long_q=0.70, enter_short_q=0.10, exit_short_q=0.30,
                  min_hold=24, cooldown=12,
                  fee_per_side=0.0003, slippage_per_side=0.0001):
    """Строим позицию по всей истории и возвращаем текущую целевую позу, ретёрн последнего бара и обновлённое equity (теоретически)."""
    ret, proba = align_ret_and_proba(prices, proba)
    if ret is None or len(ret) < 10:
        return None

    # Если числовые пороги не заданы — возьмём по квантилям текущего proba
    if None in (enter_long, exit_long, enter_short, exit_short):
        q = proba.quantile
        enter_long  = float(q(enter_long_q))
        exit_long   = float(q(exit_long_q))
        enter_short = float(q(enter_short_q))
        exit_short  = float(q(exit_short_q))

    pos_series = positions_hysteresis(
        proba, enter_long, exit_long, enter_short, exit_short,
        min_hold=min_hold, cooldown=cooldown
    )
    # без утечки: позиция действует со следующего бара
    pos = pos_series.shift(1).reindex(ret.index).fillna(0).astype(int)

    # последний завершённый бар
    last_ts = ret.index[-1]
    last_ret = float(ret.iloc[-1])
    last_price = float(prices.loc[last_ts])

    desired_pos = int(pos.loc[last_ts])

    # метрики на всякий
    cost_per_side = fee_per_side + slippage_per_side
    turns = pos.diff().abs().fillna(0)
    costs = turns * cost_per_side
    strat_ret = pos * ret - costs
    eq = (1.0 + strat_ret).cumprod()
    bpy = bars_per_year(tf)
    sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * math.sqrt(bpy)
    mdd = max_drawdown(eq)

    return dict(
        last_ts=last_ts, last_price=last_price,
        desired_pos=desired_pos, pos_series=pos, ret_series=strat_ret, equity_series=eq,
        thresholds=dict(enter_long=enter_long, exit_long=exit_long, enter_short=enter_short, exit_short=exit_short),
        sharpe=float(sharpe), max_dd=float(mdd)
    )

# ----------------------------- Main Loop -----------------------------
def run_once(args):
    # 1) thresholds/config
    if not os.path.exists(THRESHOLDS_JSON):
        raise FileNotFoundError(f"{THRESHOLDS_JSON} not found. Run aibot.py and save thresholds first.")
    cfg = load_thresholds(THRESHOLDS_JSON)

    symbol = args.symbol or cfg["symbol"]
    timeframe = args.timeframe or cfg["timeframe"]
    horizon = args.horizon or cfg["horizon"]
    smooth_span = args.smooth_span if args.smooth_span is not None else cfg["smooth_span"]

    fee_per_side = cfg["fee_per_side"]
    slippage_per_side = cfg["slippage_per_side"]
    min_hold = cfg["min_hold"]
    cooldown = cfg["cooldown"]

    # 2) exchange
    ex = ccxt.binance()
    ex.enableRateLimit = True
    # Примечание: для paper мы просто читаем маркет-данные, ключи не нужны.

    # 3) данные
    df = fetch_ohlcv(ex, symbol, timeframe, limit=args.max_bars)
    df = df.rename(columns=str.lower)
    feats = build_features(df)
    y = make_labels(feats, horizon, fee_per_side, slippage_per_side)

    feat_cols = ['ret1','ret3','ret5','ret20','rsi','sma_ratio','macd','macd_sig','macd_diff','bb_pos','vol20']
    X = feats[feat_cols].shift(1).dropna()
    y = y.loc[X.index]

    # OOF прогноз (walk-forward)
    proba = oof_predict_lgbm(X, y, n_splits=5, random_state=42)
    proba_s = smooth_proba(proba, smooth_span)

    step = backtest_step(
        feats['c'], proba_s, timeframe,
        enter_long=cfg["enter_long"], exit_long=cfg["exit_long"],
        enter_short=cfg["enter_short"], exit_short=cfg["exit_short"],
        enter_long_q=cfg["enter_long_q"], exit_long_q=cfg["exit_long_q"],
        enter_short_q=cfg["enter_short_q"], exit_short_q=cfg["exit_short_q"],
        min_hold=min_hold, cooldown=cooldown,
        fee_per_side=fee_per_side, slippage_per_side=slippage_per_side
    )
    if step is None:
        print("Not enough aligned data yet. Try increasing --max-bars.")
        return

    ensure_csv_headers()
    state = load_state(default_equity=args.start_equity)

    # Рассчитываем теоретический equity до последнего бара
    if len(step["equity_series"]) > 0:
        eq_last = float(step["equity_series"].iloc[-1])
    else:
        eq_last = state.get("equity", args.start_equity)

    # решаем — есть ли смена позы
    last_pos = int(state.get("last_pos", 0))
    desired = int(step["desired_pos"])
    last_ts = step["last_ts"]
    last_price = step["last_price"]

    if state.get("last_ts") == str(last_ts) and last_pos == desired:
        note = "no_change (same bar already processed)"
        print(f"[{last_ts}] pos stays {desired}, price={last_price:.2f}, equity={eq_last:.4f} — {note}")
    else:
        if desired != last_pos:
            side = "enter_long" if desired == 1 else "enter_short" if desired == -1 else "flat"
            note = f"switch {last_pos} -> {desired}"
            log_trade(last_ts, symbol, side, last_pos, desired, last_price, note)
            print(f"[{last_ts}] {note} @ {last_price:.2f}; equity={eq_last:.4f}; sharpe={step['sharpe']:.2f} mdd={step['max_dd']:.2%}")
        else:
            print(f"[{last_ts}] hold {desired} @ {last_price:.2f}; equity={eq_last:.4f}; sharpe={step['sharpe']:.2f}")

    log_equity(last_ts, eq_last)
    # обновляем состояние
    state.update({"last_ts": str(last_ts), "last_pos": desired, "equity": eq_last})
    save_state(state)

def main():
    parser = argparse.ArgumentParser(description="Paper trader (signals only, no live orders)")
    parser.add_argument("--symbol", type=str, default=None, help="override symbol (else from JSON)")
    parser.add_argument("--timeframe", type=str, default=None, help="override timeframe (else from JSON)")
    parser.add_argument("--horizon", type=int, default=None, help="override horizon bars (else from JSON)")
    parser.add_argument("--smooth-span", type=int, default=None, help="override EMA span")
    parser.add_argument("--max-bars", type=int, default=6000, help="bars to fetch for training")
    parser.add_argument("--start-equity", type=float, default=1.0, help="starting equity (relative)")
    parser.add_argument("--loop", action="store_true", help="run forever, waking up on each new bar")
    parser.add_argument("--tick-seconds", type=int, default=30, help="polling interval for new bar check")
    args = parser.parse_args()

    if not os.path.exists(THRESHOLDS_JSON):
        raise FileNotFoundError(f"{THRESHOLDS_JSON} not found. Run aibot.py, then save thresholds.")

    if args.loop:
        print("Paper trader loop started. Ctrl+C to stop.")
        while True:
            try:
                run_once(args)
                # ждём до следующего закрытого бара
                t_now = now_utc()
                tf_next = ceil_to_next_bar(t_now, args.timeframe or load_thresholds(THRESHOLDS_JSON)["timeframe"])
                wait_s = max(5, (tf_next - t_now).total_seconds() + 2)  # небольшая подушка
                # если слишком большой интервал — опрашиваем раз в tick-seconds
                sleep_s = min(wait_s, args.tick_seconds)
                time.sleep(sleep_s)
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print("Error in loop:", e)
                traceback.print_exc()
                time.sleep(5)
    else:
        run_once(args)

if __name__ == "__main__":
    main()