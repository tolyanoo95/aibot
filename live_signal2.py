#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import ccxt
from joblib import load
import warnings

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

warnings.filterwarnings("ignore", category=FutureWarning)

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

def fetch_ohlcv_all(ex, symbol: str, timeframe: str, max_bars: int = 2000) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=max_bars)
    df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
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

    return d

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
                    decision_price: float, planned_exec_ts: pd.Timestamp,
                    current_pos: int, sl_price: float | None = None, htf_status: str = "OFF"):
    """Печатает решение на каждом закрытом баре."""
    action = sig["action"]
    
    pos_map = {1: "LONG", -1: "SHORT", 0: "FLAT"}
    
    if action == "EXIT_BY_SL":
        side = "LONG" if current_pos == 1 else "SHORT"
        print(f"!!! STOP-LOSS: ВЫХОД ИЗ {side} по {sl_price:.8f} @ {decision_ts} — {symbol}")
    elif action in ("ENTER_LONG", "ENTER_SHORT"):
        print(f"АЛГОРИТМ ПРИНЯЛ РЕШЕНИЕ: {('ВОЙТИ LONG' if action=='ENTER_LONG' else 'ВОЙТИ SHORT')} "
              f"по {decision_price:.8f} @ {decision_ts}  — {symbol}")
    else:
        # Информативная строка, когда входа нет
        human = {"HOLD_LONG":"ДЕРЖАТЬ LONG", "HOLD_SHORT":"ДЕРЖАТЬ SHORT", 
                 "STAY_FLAT":"БЕЗ ДЕЙСТВИЙ", "EXIT_TO_FLAT":"ВЫЙТИ В НОЛЬ"}
        sl_info = f", SL={sl_price:.8f}" if sl_price is not None and not np.isnan(sl_price) else ", SL=nan"
        htf_info = f", HTF={htf_status}"
        print(f"РЕШЕНИЕ: {human.get(action, action)} (pos={pos_map[sig['next_state']]}{htf_info}{sl_info}, "
              f"proba={sig['proba']:.4f}) @ {decision_ts}  — {symbol} - {decision_price:.8f}")
        
    # Всегда показываем, когда планируется исполнение
    if action != "EXIT_BY_SL":
        print(f"  Плановое исполнение: {planned_exec_ts} (open следующего бара)")
    sys.stdout.flush()

# ----------------- JSONL logging -----------------
def append_jsonl(path: str, obj: dict):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------- State & Main Logic -----------------
class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.load_settings()
        self.ex = ccxt.binance(); self.ex.enableRateLimit = True
        
        # State
        self.current_pos = 0 # -1, 0, 1
        self.entry_price = np.nan
        self.sl_price = np.nan
        # Для Trailing SL
        self.peak_since_entry = np.nan
        self.trough_since_entry = np.nan
        self.last_seen_bar_open = None
        self.pending_exec = None

    def load_settings(self):
        if not os.path.exists(self.config["model"]):
            print(f"Model file not found: {self.config["model"]}\n→ сначала запусти aibot.py")
            sys.exit(1)
        if not os.path.exists(self.config["thresholds"]):
            print(f"Thresholds file not found: {self.config["thresholds"]}\n→ сначала запусти aibot.py")
            sys.exit(1)

        th_all = json.load(open(self.config["thresholds"], "r"))
        self.thresholds = th_all["thresholds"]
        self.min_hold = int(th_all.get("min_hold", 24))
        self.cooldown = int(th_all.get("cooldown", 12))
        self.smooth_span = int(th_all.get("smooth_span", 12))
        self.symbol = self.config.get("symbol") or th_all.get("symbol")
        self.timeframe = self.config.get("timeframe") or th_all.get("timeframe")
        
        # --- Stop-Loss: приоритет [CLI флаг -> JSON -> дефолт] ---
        sl_settings = th_all.get("stop_loss", {})
        # Сначала читаем из JSON, это база
        self.use_stop_loss = sl_settings.get("use", False)
        # Если в CLI был передан флаг --disable-stop-loss, он станет False. Если нет - останется True из set_defaults
        # Этот флаг должен переопределить значение из JSON.
        if self.config["use_stop_loss"] is not None and self.config["use_stop_loss"] != sl_settings.get("use"):
             self.use_stop_loss = self.config["use_stop_loss"]

        # Аналогично для multiplier: CLI имеет приоритет
        cli_sl_mult = self.config.get("sl_atr_multiplier")
        if cli_sl_mult is not None:
            self.sl_atr_multiplier = cli_sl_mult
        else:
            self.sl_atr_multiplier = sl_settings.get("atr_multiplier", 2.5)
        
        # --- HTF Filter: приоритет [CLI флаг -> JSON -> дефолт] ---
        htf_settings = th_all.get("htf_filter", {})
        # Сначала читаем из JSON
        self.use_htf_filter = htf_settings.get("use", False)
        # Затем смотрим, не переопределил ли его флаг CLI
        if self.config["use_htf_filter"] is not None and self.config["use_htf_filter"] != htf_settings.get("use"):
            self.use_htf_filter = self.config["use_htf_filter"]

        self.htf_timeframe = self.config.get("htf_timeframe") or htf_settings.get("timeframe", "4h")
        self.htf_supertrend_period = self.config.get("htf_supertrend_period") or htf_settings.get("supertrend_period", 14)
        self.htf_supertrend_multiplier = self.config.get("htf_supertrend_multiplier") or htf_settings.get("supertrend_multiplier", 2.5)

        pack = load(self.config["model"])
        self.model = pack["model"]
        # Используем финальный набор признаков
        self.feat_cols = [
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
        print(f"Settings loaded for {self.symbol} {self.timeframe}. SL_ATR={self.sl_atr_multiplier if self.use_stop_loss else 'OFF'}, HTF={self.htf_timeframe if self.use_htf_filter else 'OFF'}")

    def run_once(self):
        # --- Data Fetching ---
        df = fetch_recent_ohlcv(self.ex, self.symbol, self.timeframe, limit=max(600, self.config["history_bars"]))
        if len(df) < 200:
            print(f"Not enough bars ({len(df)})"); return

        # --- HTF Filter Calculation ---
        trend_is_up = None # True-up, False-down, None-не используется
        htf_status = "OFF"
        if self.use_htf_filter:
            main_tf_mins = timeframe_to_minutes(self.timeframe)
            htf_tf_mins = timeframe_to_minutes(self.htf_timeframe)
            htf_bars_needed = int((len(df) * main_tf_mins) / htf_tf_mins) + self.htf_supertrend_period + 5

            df_htf = fetch_ohlcv_all(self.ex, self.symbol, self.htf_timeframe, max_bars=htf_bars_needed)
            
            _, trend_direction = calculate_supertrend(
                df_htf, 
                atr_period=self.htf_supertrend_period, 
                atr_multiplier=self.htf_supertrend_multiplier
            )
            trend_is_up = trend_direction.iloc[-1]
            
            htf_status = "UP" if trend_is_up else "DOWN"
            #print(f"HTF({self.htf_timeframe}) Supertrend is {htf_status}.")
        
        # --- Execution Confirmation ---
        if self.pending_exec is not None:
            planned_ts = self.pending_exec["planned_ts"]
            if planned_ts in df.index:
                executed_price = float(df.loc[planned_ts, "o"])
                print(f">>> ИСПОЛНЕНО: {self.pending_exec['action']} по {executed_price:.8f} @ {planned_ts} (open)")
                
                if self.pending_exec['action'] in ("ENTER_LONG", "ENTER_SHORT"):
                    self.current_pos = 1 if self.pending_exec['action'] == "ENTER_LONG" else -1
                    self.entry_price = executed_price
                    # Set SL based on the ATR of the *decision* bar
                    atr_val = self.pending_exec['atr_on_decision']
                    if self.current_pos == 1:
                        self.sl_price = self.entry_price - self.sl_atr_multiplier * atr_val
                        self.peak_since_entry = self.entry_price # Инициализация пика
                    else:
                        self.sl_price = self.entry_price + self.sl_atr_multiplier * atr_val
                        self.trough_since_entry = self.entry_price # Инициализация впадины
                else: # EXIT
                    self.current_pos = 0
                    self.entry_price, self.sl_price = np.nan, np.nan
                    self.peak_since_entry, self.trough_since_entry = np.nan, np.nan # Сброс
                self.pending_exec = None

        # --- Stop-Loss Check (on previous, now-closed bar) ---
        if self.current_pos != 0 and self.use_stop_loss and not np.isnan(self.sl_price):
            last_closed_bar = df.iloc[-2]
            
            # Сначала обновляем Trailing Stop
            # ATR для трейлинга берем с *текущего* бара, чтобы он был адаптивным
            current_atr = build_features(df)['atr'].iloc[-1]
            if self.current_pos == 1:
                self.peak_since_entry = max(self.peak_since_entry, last_closed_bar['h'])
                new_sl = self.peak_since_entry - self.sl_atr_multiplier * current_atr
                self.sl_price = max(self.sl_price, new_sl)
            elif self.current_pos == -1:
                self.trough_since_entry = min(self.trough_since_entry, last_closed_bar['l'])
                new_sl = self.trough_since_entry + self.sl_atr_multiplier * current_atr
                self.sl_price = min(self.sl_price, new_sl)

            # Теперь проверяем срабатывание
            sl_hit = False
            if self.current_pos == 1 and last_closed_bar['l'] <= self.sl_price:
                sl_hit = True
            elif self.current_pos == -1 and last_closed_bar['h'] >= self.sl_price:
                sl_hit = True
            
            if sl_hit:
                # Immediate exit, no waiting for next bar open
                tf_delta = timeframe_to_timedelta(self.timeframe)
                decision_ts = last_closed_bar.name + tf_delta
                sig = {"action": "EXIT_BY_SL", "proba": np.nan, "next_state": 0}
                print_every_bar(sig, self.symbol, decision_ts, last_closed_bar['c'], None, self.current_pos, self.sl_price, htf_status=htf_status)
                
                # Log and reset state
                self.log_decision(sig, decision_ts, last_closed_bar.name, last_closed_bar['c'], None)
                self.current_pos = 0
                self.entry_price, self.sl_price = np.nan, np.nan
                self.peak_since_entry, self.trough_since_entry = np.nan, np.nan # Сброс
                self.last_seen_bar_open = df.index[-1] # Skip normal signal on this bar
                return

        # --- Feature & Proba Calculation ---
        feat = build_features(df)
        X = feat[self.feat_cols].shift(1).dropna()
        if len(X) < 50: return
        proba = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        proba_s = proba.ewm(span=self.smooth_span, adjust=False).mean() if self.smooth_span and self.smooth_span > 1 else proba

        # --- New Bar Check ---
        current_bar_open = proba_s.index[-1]
        if self.last_seen_bar_open is not None and current_bar_open == self.last_seen_bar_open:
            return

        # --- Signal Decision ---
        sig = decide_signal(proba_s, self.thresholds, self.min_hold, self.cooldown)
        if not sig: return
        
        # We can't act on a signal if we are already pending an action
        if self.pending_exec is not None:
            return

        # --- Apply HTF Filter ---
        if self.use_htf_filter and trend_is_up is not None:
            if sig['action'] == "ENTER_LONG" and not trend_is_up:
                print(f"HTF filter blocks LONG signal (HTF trend is DOWN)")
                sig['action'] = "STAY_FLAT"; sig['next_state'] = 0
            elif sig['action'] == "ENTER_SHORT" and trend_is_up:
                print(f"HTF filter blocks SHORT signal (HTF trend is UP)")
                sig['action'] = "STAY_FLAT"; sig['next_state'] = 0

        # --- Process Signal ---
        # Don't enter if already in position, don't exit if already flat
        if (sig['action'] in ("ENTER_LONG", "ENTER_SHORT") and self.current_pos != 0) or \
           (sig['action'] == "EXIT_TO_FLAT" and self.current_pos == 0):
            sig['action'] = "HOLD_LONG" if self.current_pos == 1 else ("HOLD_SHORT" if self.current_pos == -1 else "STAY_FLAT")
            sig['next_state'] = self.current_pos

        tf_delta = timeframe_to_timedelta(self.timeframe)
        decision_bar_open_ts = sig["bar_open_ts"]
        decision_ts = decision_bar_open_ts + tf_delta
        planned_exec_ts = decision_ts
        decision_price = float(df["c"].reindex(proba_s.index).iloc[-1])
        
        print_every_bar(sig, self.symbol, decision_ts, decision_price, planned_exec_ts, self.current_pos, self.sl_price, htf_status=htf_status)
        
        self.log_decision(sig, decision_ts, decision_bar_open_ts, decision_price, planned_exec_ts)

        if sig["action"] in ("ENTER_LONG", "ENTER_SHORT", "EXIT_TO_FLAT"):
            atr_on_decision = feat['atr'].reindex(proba_s.index).iloc[-1]
            self.pending_exec = {
                "action": sig["action"], 
                "planned_ts": pd.Timestamp(planned_exec_ts),
                "atr_on_decision": atr_on_decision
            }

        self.last_seen_bar_open = current_bar_open

    def log_decision(self, sig, decision_ts, decision_bar_open_ts, decision_price, planned_exec_ts):
        log_rec = {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "decision_ts": str(decision_ts),
            "decision_bar_open_ts": str(decision_bar_open_ts),
            "decision_price": decision_price,
            "proba": sig.get("proba"), "action": sig["action"],
            "current_pos_before": self.current_pos,
            "next_state_model": sig.get("next_state"),
            "planned_exec_ts": str(planned_exec_ts) if planned_exec_ts else None,
            "sl_price_before": self.sl_price
        }
        append_jsonl(self.config["log"], log_rec)
        if sig["action"] in ("ENTER_LONG", "ENTER_SHORT"):
            append_jsonl(self.config["entries_log"], log_rec)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Live decisions printer (каждый закрытый бар; входы показываются явно)")
    ap.add_argument("--model", default="final_model_lgbm.pkl", help="model file from aibot.py")
    ap.add_argument("--thresholds", default="best_thresholds.json", help="thresholds file from aibot.py")
    ap.add_argument("--symbol", type=str, default="SOL/USDT", help="Symbol to trade")
    ap.add_argument("--timeframe", type=str, default="15m", help="Trading timeframe")
    
    # By default, filters are ON if no flag is specified. Use flags to disable them.
    ap.add_argument("--disable-htf-filter", dest="use_htf_filter", action="store_false", help="Отключить фильтр по старшему ТФ (Supertrend)")
    ap.add_argument("--enable-htf-filter", dest="use_htf_filter", action="store_true", help="Принудительно включить фильтр по старшему ТФ")
    ap.set_defaults(use_htf_filter=None) # Убираем дефолт, чтобы различать "не указано" и "указано"
    ap.add_argument("--htf-timeframe", type=str, default=None, help="Старший ТФ для фильтра")
    ap.add_argument("--htf-supertrend-period", type=int, default=None, help="Период ATR для Supertrend-фильтра HTF")
    ap.add_argument("--htf-supertrend-multiplier", type=float, default=None, help="Множитель ATR для Supertrend-фильтра HTF")
    
    # Stop-Loss
    ap.add_argument("--disable-stop-loss", dest="use_stop_loss", action="store_false", help="Отключить динамический стоп-лосс по ATR")
    ap.add_argument("--enable-stop-loss", dest="use_stop_loss", action="store_true", help="Принудительно включить стоп-лосс")
    ap.set_defaults(use_stop_loss=None) # Убираем дефолт
    ap.add_argument("--sl-atr-multiplier", type=float, default=None, help="Переопределить множитель ATR для стоп-лосса")
    
    # Добавляем недостающие аргументы для консистентности
    ap.add_argument("--history-bars", type=int, default=500, help="bars for context & smoothing")
    ap.add_argument("--loop", action="store_true", help="run forever")
    ap.add_argument("--sleep", type=int, default=30, help="polling sleep seconds in loop mode")
    ap.add_argument("--log", default=None, help="JSONL файл с решениями каждого бара")
    ap.add_argument("--entries-log", default=None, help="JSONL файл только с входами (ENTER_LONG/ENTER_SHORT)")

    args = ap.parse_args()

    # Собираем конфиг из файлов и аргументов
    config = {
        "model": args.model,
        "thresholds": args.thresholds,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "history_bars": args.history_bars,
        "log": args.log,
        "entries_log": args.entries_log,
        "use_stop_loss": args.use_stop_loss,
        "sl_atr_multiplier": args.sl_atr_multiplier,
        "use_htf_filter": args.use_htf_filter,
        "htf_timeframe": args.htf_timeframe,
        "htf_supertrend_period": args.htf_supertrend_period,
        "htf_supertrend_multiplier": args.htf_supertrend_multiplier
    }

    trader = LiveTrader(config)
    
    # one-shot
    if not args.loop:
        trader.run_once(); return

    # loop
    print(f"Started loop: symbol={trader.symbol}, timeframe={trader.timeframe}")
    while True:
        try:
            trader.run_once()
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
