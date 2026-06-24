"""BTC 5-minute candle probability predictor

Heuristic/probabilistic bot using ccxt to fetch market data (candles, orderbook, trades)
and combining technical indicators + orderflow imbalance to output a probability that
the next 5-minute BTC/USDT candle will close green.

Usage:
  - Install dependencies: pip install ccxt pandas numpy
  - Set environment variables to choose exchange/symbol if desired:
      EXCHANGE (default: "binance")
      SYMBOL (default: "BTC/USDT")
  - Run: python main.py

Notes:
  - This is an interpretable heuristic model (sigmoid of weighted features).
  - Weights are heuristics and should be trained/tuned with historical labeled data
    for better probabilities.
  - Watch out for exchange rate limits; adjust fetch intervals if you hit limits.
"""

import os
import time
import math
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# --- Configuration ---
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = '5m'
OHLCV_LIMIT = 200
ORDERBOOK_DEPTH = 20
TRADE_LOOKBACK_SECONDS = 60 * 3  # last 3 minutes of trades for orderflow

# Heuristic model weights (feature order described in compute_features)
# These are starting heuristics. Consider training these using historical labels.
WEIGHTS = np.array([0.6, 0.8, 0.4, 0.3, 1.2, 1.0, 1.5])
BIAS = -0.05


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))


def open_exchange(name: str):
    exchange_cls = getattr(ccxt, name)
    exchange = exchange_cls({
        'enableRateLimit': True,
        # add API keys in environment if you want private endpoints (not needed here)
    })
    return exchange


def fetch_ohlcv(exchange, symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    return df


def fetch_orderbook(exchange, symbol, depth=ORDERBOOK_DEPTH):
    ob = exchange.fetch_order_book(symbol, depth)
    return ob


def fetch_recent_trades(exchange, symbol, since=None, limit=1000):
    try:
        trades = exchange.fetch_trades(symbol, since=since, limit=limit)
        return trades
    except Exception:
        # Not all exchanges support since/limit similarly; fallback
        return exchange.fetch_trades(symbol, limit=limit)


def orderbook_imbalance(orderbook):
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    bid_vol = sum([b[1] for b in bids[:ORDERBOOK_DEPTH]])
    ask_vol = sum([a[1] for a in asks[:ORDERBOOK_DEPTH]])
    if bid_vol + ask_vol == 0:
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)


def trade_imbalance(trades):
    # trades are dicts with amount and side (maybe 'side' or 'takerSide' depending on exchange)
    buy_vol = 0.0
    sell_vol = 0.0
    for t in trades:
        amount = float(t.get('amount', t.get('size', 0) or 0))
        # guess side
        side = t.get('side') or t.get('takerSide') or t.get('maker')
        # normalize side detection
        if side in ('buy', 'Buy', 'b', 'taker', 'buy_market'):
            buy_vol += amount
        elif side in ('sell', 'Sell', 's', 'maker', 'sell_market'):
            sell_vol += amount
        else:
            # Some exchanges don't provide side; use price vs market-ask/bid heuristic if available
            # fallback: ignore
            pass
    if buy_vol + sell_vol == 0:
        return 0.0
    return (buy_vol - sell_vol) / (buy_vol + sell_vol)


def compute_features(df, orderbook, trades):
    # df is ohlcv with latest at the bottom
    features = []

    # 1) last 5m return
    last_return = (df['close'].iloc[-1] - df['open'].iloc[-1]) / (df['open'].iloc[-1] + 1e-12)
    features.append(last_return)

    # 2) short-term momentum: return of last 3 candles
    recent_return = (df['close'].iloc[-1] - df['close'].iloc[-4]) / (df['close'].iloc[-4] + 1e-12)
    features.append(recent_return)

    # 3) EMA slope (10 vs 30) of close
    e10 = ema(df['close'], span=10)
    e30 = ema(df['close'], span=30)
    ema_slope = (e10.iloc[-1] - e30.iloc[-1]) / (df['close'].iloc[-1] + 1e-12)
    features.append(ema_slope)

    # 4) RSI on closes (14)
    r = rsi(df['close'], period=14)
    rsi_last = (r.iloc[-1] - 50) / 50.0  # normalized around 0
    features.append(rsi_last)

    # 5) orderbook imbalance (bids vs asks)
    ob_imb = orderbook_imbalance(orderbook)
    features.append(ob_imb)

    # 6) recent trade imbalance
    trade_imb = trade_imbalance(trades)
    features.append(trade_imb)

    # 7) volume spike: last candle volume vs median of previous N
    med_vol = np.median(df['volume'].iloc[-20:-1]) if len(df) > 21 else np.median(df['volume'].iloc[:-1]) if len(df) > 2 else df['volume'].iloc[-1]
    vol_spike = (df['volume'].iloc[-1] - med_vol) / (med_vol + 1e-12)
    features.append(vol_spike)

    return np.array(features, dtype=float)


def predict_probability(features, weights=WEIGHTS, bias=BIAS):
    # simple linear + sigmoid
    score = np.dot(weights, features) + bias
    prob = sigmoid(score)
    return float(prob), float(score)


def align_to_next_timeframe(timeframe='5m'):
    now = datetime.now(timezone.utc)
    minutes = (now.minute // 5) * 5
    # compute next 5m boundary
    next_min = ((now.minute // 5) + 1) * 5
    if next_min >= 60:
        # next hour
        next_dt = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_min, second=0, microsecond=0)
    wait_seconds = (next_dt - now).total_seconds()
    return wait_seconds


def main():
    print(f"Starting BTC 5m prediction bot on {EXCHANGE_NAME} {SYMBOL}")
    exchange = open_exchange(EXCHANGE_NAME)

    # Warm up: fetch historic candles
    try:
        df = fetch_ohlcv(exchange, SYMBOL)
    except Exception as e:
        print("Error fetching initial OHLCV:", e)
        return

    while True:
        try:
            # align so predictions are made right after a closed 5m candle
            wait = align_to_next_timeframe(TIMEFRAME)
            print(f"Waiting {int(wait)}s until next 5m boundary...")
            time.sleep(max(1, wait))

            df = fetch_ohlcv(exchange, SYMBOL)
            orderbook = fetch_orderbook(exchange, SYMBOL)

            # fetch trades in last few minutes
            since_ms = int((datetime.now(timezone.utc).timestamp() - TRADE_LOOKBACK_SECONDS) * 1000)
            trades = fetch_recent_trades(exchange, SYMBOL, since=since_ms)

            features = compute_features(df, orderbook, trades)
            prob, score = predict_probability(features)

            # print readable info
            now = datetime.now().astimezone()
            print("-------------------------------------------------------------")
            print(now.isoformat(), f"Prediction next {TIMEFRAME} candle will CLOSE GREEN: {prob*100:.2f}%")
            print(f"Raw score: {score:.4f}")
            print("Features:")
            print(f" last_return: {features[0]:.6f}")
            print(f" recent_3c_return: {features[1]:.6f}")
            print(f" ema_slope: {features[2]:.6f}")
            print(f" rsi_norm: {features[3]:.6f}")
            print(f" orderbook_imb: {features[4]:.6f}")
            print(f" trade_imb: {features[5]:.6f}")
            print(f" vol_spike: {features[6]:.6f}")
            print("-------------------------------------------------------------")

            # optionally: persist predictions or send to webhook, exchange, or UI here

            # sleep a bit to avoid immediate re-fetching, loop will re-align at top
            time.sleep(2)

        except KeyboardInterrupt:
            print("Stopping by user")
            break
        except Exception as e:
            print("Error in main loop:", e)
            # backoff for a little while on errors
            time.sleep(10)


if __name__ == '__main__':
    main()
