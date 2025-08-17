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

# ----------------- utils -----------------
def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 1440
    raise ValueError(f"Unknown timeframe: {tf}")

def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=timeframe_to_minutes(tf))

def fetch_recent_ohlcv(ex, symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
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

def decide_signal(proba_s: pd.Series, thresholds: dict, min_hold: int, cooldown: int):
    """Возвращает решение по последнему закрытому бару."""
    pos_raw = positions_hysteresis_numeric(
        proba_s,
        thresholds["enter_long"], thresholds["exit_long"],
        thresholds["enter_short"], thresholds["exit_short"],
        min_hold=min_hold, cooldown=cooldown
    )
    if len(pos_raw) < 2:
        return None
    prev_state = int(pos_raw.iloc[-2])
    next_state = int(pos_raw.iloc[-1])

    if prev_state != next_state:
        if next_state == 1:   action = "ENTER_LONG"
        elif next_state == -1:action = "ENTER_SHORT"
        else:                 action = "EXIT_TO_FLAT"
    else:
        if next_state == 1:   action = "HOLD_LONG"
        elif next_state == -1:action = "HOLD_SHORT"
        else:                 action = "STAY_FLAT"

    return dict(
        prev_state=prev_state,
        next_state=next_state,
        action=action,
        proba=float(proba_s.iloc[-1]),
        bar_open_ts=proba_s.index[-1]   # время ОТКРЫТИЯ бара-решения
    )

# ----------------- pretty print -----------------
def print_every_bar(sig: dict, symbol: str, decision_ts: pd.Timestamp,
                    decision_price: float, planned_exec_ts: pd.Timestamp):
    """Печатает решение на каждом закрытом баре."""
    action = sig["action"]
    if action in ("ENTER_LONG", "ENTER_SHORT"):
        print(f"АЛГОРИТМ ПРИНЯЛ РЕШЕНИЕ: {('ВОЙТИ LONG' if action=='ENTER_LONG' else 'ВОЙТИ SHORT')} "
              f"по {decision_price:.8f} @ {decision_ts}  — {symbol}")
    else:
        # Информативная строка, когда входа нет
        human = {"HOLD_LONG":"ДЕРЖАТЬ LONG", "HOLD_SHORT":"ДЕРЖАТЬ SHORT", "STAY_FLAT":"БЕЗ ДЕЙСТВИЙ", "EXIT_TO_FLAT":"ВЫЙТИ В НОЛЬ"}
        print(f"РЕШЕНИЕ: {human.get(action, action)} "
              f"(proba={sig['proba']:.4f}) @ {decision_ts}  — {symbol} - {decision_price:.8f}")
    # Всегда показываем, когда планируется исполнение
    print(f"  Плановое исполнение: {planned_exec_ts} (open следующего бара)")
    sys.stdout.flush()

# ----------------- JSONL logging -----------------
def append_jsonl(path: str, obj: dict):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Live decisions printer (каждый закрытый бар; входы показываются явно)")
    ap.add_argument("--model", default="final_model_lgbm.pkl", help="model file from aibot.py")
    ap.add_argument("--thresholds", default="best_thresholds.json", help="thresholds file from aibot.py")
    ap.add_argument("--symbol", default=None, help="override symbol (else from thresholds.json)")
    ap.add_argument("--timeframe", default=None, help="override timeframe (else from thresholds.json)")
    ap.add_argument("--history-bars", type=int, default=5000, help="bars for context & smoothing")
    ap.add_argument("--loop", action="store_true", help="run forever")
    ap.add_argument("--sleep", type=int, default=30, help="polling sleep seconds in loop mode")
    ap.add_argument("--log", default=None, help="JSONL файл с решениями каждого бара")
    ap.add_argument("--entries-log", default=None, help="JSONL файл только с входами (ENTER_LONG/ENTER_SHORT)")
    args = ap.parse_args()

    # файлы
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}\n→ сначала запусти aibot.py, чтобы создать final_model_lgbm.pkl")
        sys.exit(1)
    if not os.path.exists(args.thresholds):
        print(f"Thresholds file not found: {args.thresholds}\n→ сначала запусти aibot.py, чтобы создать best_thresholds.json")
        sys.exit(1)

    # пороги/параметры
    th_all = json.load(open(args.thresholds, "r"))
    th = th_all["thresholds"]
    min_hold = int(th_all.get("min_hold", th_all.get("params", {}).get("min_hold", 24)))
    cooldown = int(th_all.get("cooldown", th_all.get("params", {}).get("cooldown", 12)))
    smooth_span = int(th_all.get("smooth_span", 12))
    symbol = args.symbol or th_all.get("symbol", "XRP/USDT")
    timeframe = args.timeframe or th_all.get("timeframe", "15m")

    # модель
    pack = load(args.model)
    model = pack["model"]
    feat_cols = pack.get("feat_cols", ['ret1','ret3','ret5','ret20','rsi','sma_ratio','macd','macd_sig','macd_diff','bb_pos','vol20'])

    # биржа
    ex = ccxt.binance(); ex.enableRateLimit = True

    last_seen_bar_open = None
    pending_exec = None  # {'action': ..., 'planned_ts': Timestamp}

    def once():
        nonlocal last_seen_bar_open, pending_exec

        # данные
        df = fetch_recent_ohlcv(ex, symbol, timeframe, limit=max(600, args.history_bars))
        if len(df) < 200:
            print(f"Not enough bars ({len(df)})"); return

        # подтвердить исполнение, если наступил следующий бар
        if pending_exec is not None:
            planned_ts = pending_exec["planned_ts"]
            if planned_ts in df.index:
                executed_price = float(df.loc[planned_ts, "o"])
                print(f">>> ИСПОЛНЕНО: {pending_exec['action']} по {executed_price:.8f} @ {planned_ts} (open)")
                pending_exec = None

        # фичи и вероятность
        feat = build_features(df)
        X = feat[feat_cols].shift(1).dropna()
        if len(X) < 50: return
        proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
        proba_s = proba.ewm(span=smooth_span, adjust=False).mean() if smooth_span and smooth_span > 1 else proba

        # работаем только по новому закрытому бару
        current_bar_open = proba_s.index[-1]
        if last_seen_bar_open is not None and current_bar_open == last_seen_bar_open:
            return

        # решение
        sig = decide_signal(proba_s, th, min_hold=min_hold, cooldown=cooldown)
        if not sig: return

        tf_delta = timeframe_to_timedelta(timeframe)
        decision_bar_open_ts = sig["bar_open_ts"]
        decision_ts = decision_bar_open_ts + tf_delta                 # время закрытия текущего бара
        planned_exec_ts = decision_ts                                 # open следующего бара
        decision_price = float(df["c"].reindex(proba_s.index).iloc[-1])  # close текущего бара

        # печать — НА КАЖДОМ БАРЕ
        print_every_bar(sig, symbol, decision_ts, decision_price, planned_exec_ts)

        # лог JSONL (все решения)
        all_rec = {
            "symbol": symbol,
            "timeframe": timeframe,
            "decision_ts": str(decision_ts),
            "decision_bar_open_ts": str(decision_bar_open_ts),
            "decision_price": decision_price,
            "proba": sig["proba"],
            "action": sig["action"],
            "prev_state": sig["prev_state"],
            "next_state": sig["next_state"],
            "planned_exec_ts": str(planned_exec_ts),
            "thresholds": th,
            "smooth_span": smooth_span
        }
        append_jsonl(args.log, all_rec)

        # если вход — отдельная запись и план подтверждения
        if sig["action"] in ("ENTER_LONG", "ENTER_SHORT"):
            append_jsonl(args.entries_log, all_rec)
            pending_exec = {"action": sig["action"], "planned_ts": pd.Timestamp(planned_exec_ts)}

        last_seen_bar_open = current_bar_open

    # one-shot
    if not args.loop:
        once(); return

    # loop python3 live_signal.py --loop --sleep 300
    print(f"Started loop: symbol={symbol}, timeframe={timeframe}, smooth_span={smooth_span}, "
          f"min_hold={min_hold}, cooldown={cooldown}")
    while True:
        try:
            once()
        except Exception as e:
            print("ERROR:", repr(e))
        time.sleep(args.sleep)

if __name__ == "__main__":
    # macOS / libomp helper (на всякий случай для lightgbm)
    try:
        import platform, subprocess
        if platform.system() == "Darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
            prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
            os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}/lib"
    except Exception:
        pass
    main()