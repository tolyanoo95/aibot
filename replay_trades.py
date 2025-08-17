#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import ccxt
from joblib import load

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands

# ----------------- defaults -----------------
DEFAULT_SYMBOL = "SOL/USDT"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_MAX_BARS = 10000
FEE_PER_SIDE_DEFAULT = 0.0003
SLIP_PER_SIDE_DEFAULT = 0.0001

# ----------------- utils -----------------
def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 1440
    raise ValueError(f"Unknown timeframe: {tf}")

def fetch_ohlcv_all(exchange, symbol, timeframe='15m',
                    since_ms: int | None = None, limit=1000, max_bars=60000):
    tf2min = timeframe_to_minutes(timeframe)
    ms_step = tf2min * 60_000

    # берём последние max_bars (якоримся к "сейчас")
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
    return d.dropna()

def positions_hysteresis_numeric(proba: pd.Series,
                                 enter_long: float, exit_long: float,
                                 enter_short: float, exit_short: float,
                                 min_hold: int, cooldown: int) -> pd.Series:
    state, hold, cd = 0, 0, 0
    out = []
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
        out.append(state)
    return pd.Series(out, index=proba.index, dtype=int)

def extract_trades_from_pos(prices: pd.Series, pos_exec: pd.Series,
                            fee_per_side=FEE_PER_SIDE_DEFAULT, slippage_per_side=SLIP_PER_SIDE_DEFAULT) -> pd.DataFrame:
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

        if p == cur["side"]:
            continue

        # закрываем текущую позицию
        side = cur["side"]
        entry_price = float(cur["entry_price"])
        exit_ts = ts
        exit_price = price

        gross = (exit_price / entry_price - 1.0) if side == 1 else (entry_price / exit_price - 1.0)
        net = (1.0 + gross) * (1.0 - cost_side) * (1.0 - cost_side) - 1.0
        bars_held = i - int(cur["entry_i"])

        trades.append(dict(
            side="long" if side == 1 else "short",
            entry_ts=cur["entry_ts"], exit_ts=exit_ts,
            entry_price=entry_price, exit_price=exit_price,
            gross_ret=gross, net_ret=net, net_pct=net*100,
            bars=bars_held
        ))

        cur = None
        if p != 0:  # flip
            cur = dict(side=p, entry_ts=ts, entry_price=price, entry_i=i)

    # закрываем на последней цене, если что-то осталось
    if cur is not None and len(idx):
        ts = idx[-1]
        price = float(px.iloc[-1])
        side = cur["side"]
        entry_price = float(cur["entry_price"])
        gross = (price / entry_price - 1.0) if side == 1 else (entry_price / price - 1.0)
        net = (1.0 + gross) * (1.0 - cost_side) * (1.0 - cost_side) - 1.0
        bars_held = len(idx) - 1 - int(cur["entry_i"])
        trades.append(dict(
            side="long" if side == 1 else "short",
            entry_ts=cur["entry_ts"], exit_ts=ts,
            entry_price=entry_price, exit_price=price,
            gross_ret=gross, net_ret=net, net_pct=net*100,
            bars=bars_held
        ))

    df_tr = pd.DataFrame(trades)
    if not df_tr.empty:
        df_tr["win"] = df_tr["net_ret"] > 0
        df_tr["entry_ts"] = pd.to_datetime(df_tr["entry_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_tr["exit_ts"]  = pd.to_datetime(df_tr["exit_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df_tr

def save_trades_json(trades: pd.DataFrame, path="trades_replay.json") -> str:
    if trades is None or trades.empty:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return path
    fields = ["entry_ts","entry_price","exit_ts","exit_price","side","net_pct","bars","win"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trades[fields].to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return path

def summarize_trades(trades: pd.DataFrame) -> str:
    if trades is None or trades.empty:
        return "Сделок не найдено."
    total = len(trades)
    winrate = trades["win"].mean()*100
    avg_win = trades.loc[trades["win"], "net_pct"].mean() if (trades["win"].any()) else 0.0
    avg_loss = trades.loc[~trades["win"], "net_pct"].mean() if (~trades["win"]).any() else 0.0
    pf = (trades.loc[trades["win"], "net_ret"].sum() /
          abs(trades.loc[~trades["win"], "net_ret"].sum())) if (~trades["win"]).any() else np.inf
    return (f"Всего сделок: {total} | Winrate: {winrate:.1f}% | "
            f"Avg win: {avg_win:.2f}% | Avg loss: {avg_loss:.2f}% | PF: {pf:.2f}")

# -------- cashflows (депозиты/выводы) --------
def load_cashflows(path: str) -> pd.Series:
    """cashflows.json -> Series(amount) по времени."""
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
    """К каждому кэшфлоу подбираем ближайший следующий бар (если позже последнего — игнорируем)."""
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
    """
    Денежное эквити с депозитами/выводами.
    Кэшфлоу применяется ПЕРЕД доходностью бара.
    Возвращает: (series, deposits_sum, withdrawals_sum, final_value, profit, roi)
    """
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

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Replay trades from saved model & thresholds (+cashflows)")
    ap.add_argument("--model", default="final_model_lgbm.pkl", help="файл модели (из aibot.py)")
    ap.add_argument("--thresholds", default="best_thresholds.json", help="файл порогов (из aibot.py)")
    ap.add_argument("--symbol", default=None, help="переписать символ (иначе из thresholds)")
    ap.add_argument("--timeframe", default=None, help="переписать TF (иначе из thresholds)")
    ap.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS)
    ap.add_argument("--smooth-span", type=int, default=None, help="переписать сглаживание (иначе из модели/порогов)")
    ap.add_argument("--initial-capital", type=float, default=10_000.0, help="стартовый депозит")
    ap.add_argument("--cashflows", type=str, default="cashflows.json", help="JSON с депозитами/выводами [{'ts': 'YYYY-MM-DD HH:MM:SS', 'amount': 1000}, ...]")
    ap.add_argument("--out-prefix", default="trades_replay", help="префикс для файлов вывода")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Не найден файл модели: {args.model}\nСначала запусти aibot.py и сохрани модель.")
        sys.exit(1)
    if not os.path.exists(args.thresholds):
        print(f"Не найден файл порогов: {args.thresholds}\nСначала запусти aibot.py для подбора порогов.")
        sys.exit(1)

    # читаем пороги/параметры
    th_all = json.load(open(args.thresholds, "r"))
    th = th_all["thresholds"]
    min_hold = int(th_all.get("min_hold", th_all.get("params", {}).get("min_hold", 24)))
    cooldown = int(th_all.get("cooldown", th_all.get("params", {}).get("cooldown", 12)))
    fee_side = float(th_all.get("fees", {}).get("fee_per_side", FEE_PER_SIDE_DEFAULT))
    slip_side = float(th_all.get("fees", {}).get("slippage_per_side", SLIP_PER_SIDE_DEFAULT))
    symbol = args.symbol or th_all.get("symbol", DEFAULT_SYMBOL)
    timeframe = args.timeframe or th_all.get("timeframe", DEFAULT_TIMEFRAME)

    # читаем модель
    pack = load(args.model)
    model = pack["model"]
    feat_cols = pack.get("feat_cols", ['ret1','ret3','ret5','ret20','rsi','sma_ratio','macd','macd_sig','macd_diff','bb_pos','vol20'])
    smooth_span = (args.smooth_span if args.smooth_span is not None
                   else pack.get("smooth_span", th_all.get("smooth_span", 12)))

    # биржа и данные
    ex = ccxt.binance(); ex.enableRateLimit = True
    print(f"Downloading: {symbol} {timeframe} (max {args.max_bars} bars)")
    df = fetch_ohlcv_all(ex, symbol, timeframe=timeframe, max_bars=args.max_bars)
    df = df.rename(columns=str.lower)
    if len(df) < 300:
        print(f"Слишком мало баров: {len(df)}"); sys.exit(1)

    # фичи и X (shift(1) против утечек)
    feat = build_features(df)
    X = feat[feat_cols].shift(1).dropna()
    prices = feat["c"].loc[X.index]

    # предсказания и сглаживание
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    proba_s = proba.ewm(span=int(smooth_span), adjust=False).mean() if smooth_span and smooth_span > 1 else proba

    # позиции и сделки
    pos_raw = positions_hysteresis_numeric(
        proba_s, th["enter_long"], th["exit_long"], th["enter_short"], th["exit_short"],
        min_hold=min_hold, cooldown=cooldown
    )
    pos_exec = pos_raw.shift(1).reindex(prices.index).fillna(0).astype(int)
    trades = extract_trades_from_pos(prices, pos_exec, fee_per_side=fee_side, slippage_per_side=slip_side)

    # === стратегия: помесячная доходность баров (для equity)
    ret = prices.pct_change().dropna()
    # выравниваем
    idx = ret.index.intersection(pos_exec.index)
    ret = ret.loc[idx]
    pos_e = pos_exec.loc[idx]
    turns = pos_e.diff().abs().fillna(0)
    cost_side = float(fee_side + slip_side)
    costs = turns * cost_side
    strat_ret = pos_e * ret - costs  # доходность стратегии на бар

    # кэшфлоу и equity
    cflows = load_cashflows(args.cashflows)
    cflows_on_bars = align_cashflows_to_index(cflows, strat_ret.index)
    cash_eq, dep, wd, final_value, profit, roi = equity_with_cashflows(
        strat_ret, args.initial_capital, cflows_on_bars
    )

    # выводы (сделки + equity + summary)
    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"
    trades.to_csv(csv_path, index=False)
    save_trades_json(trades, json_path)

    eq_path = "equity_cashflow.csv"
    pd.DataFrame({"value": cash_eq}).to_csv(eq_path)
    csum_path = "cashflow_summary.json"
    with open(csum_path, "w") as f:
        json.dump({
            "initial_capital": args.initial_capital,
            "total_deposits": dep,
            "total_withdrawals": wd,
            "net_contributions": args.initial_capital + dep - wd,
            "final_value": final_value,
            "net_profit": profit,
            "roi": roi,
            "start_ts": str(cash_eq.index[0]) if len(cash_eq) else None,
            "end_ts": str(cash_eq.index[-1]) if len(cash_eq) else None
        }, f, indent=2)

    print(f"Saved: {csv_path}, {json_path}")
    print(summarize_trades(trades))
    print(f"Saved: {eq_path}, {csum_path}")
    print(f"Cashflow-adjusted: initial={args.initial_capital:.2f}, "
          f"deposits={dep:.2f}, withdrawals={wd:.2f} -> final={final_value:.2f}, "
          f"profit={profit:.2f} (ROI {roi*100:.2f}%)")

if __name__ == "__main__":
    # macOS / libomp helper (на всякий случай для lightgbm)
    try:
        import platform, subprocess, os as _os
        if platform.system() == "Darwin" and "DYLD_LIBRARY_PATH" not in _os.environ:
            prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
            _os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}/lib"
    except Exception:
        pass
    main()