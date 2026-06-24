"""BTC 5-minute candle probability predictor with training & backtest

This updated main.py adds:
 - A simple training pipeline that fits a logistic regression (numpy GD) on historical
   OHLCV-derived features and persists the model and scaler to disk (model.npz).
 - A backtest mode that evaluates the trained model on a held-out slice and reports
   accuracy and simple metrics.
 - Improved trade side detection using orderbook mid-price when trade side isn't provided.
 - CSV logging of live predictions to predictions.csv for future analysis.
 - CLI interface: --train, --backtest, --live (default).

Notes & caveats:
 - Historical orderbook snapshots aren't generally available via public exchange APIs,
   so training uses only OHLCV-derived features; orderbook/trade features are set to 0
   during training. Live predictions will use orderbook/trade features if available.
 - This uses a simple logistic regression trained with gradient descent in numpy to avoid
   heavyweight dependencies. For best results, consider using scikit-learn/XGBoost and
   a more extensive labeled dataset.
 - Always backtest thoroughly before using predictions for trading.
"""

import os
import time
import math
import json
import argparse
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# --- Configuration ---
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = '5m'
OHLCV_LIMIT = 2000  # increase for training/backtesting
ORDERBOOK_DEPTH = 30
TRADE_LOOKBACK_SECONDS = 60 * 3  # last 3 minutes of trades for orderflow
MODEL_PATH = 'model.npz'
PREDICTION_LOG = 'predictions.csv'

# Default heuristic weights (used if no trained model found)
DEFAULT_WEIGHTS = np.array([0.6, 0.8, 0.4, 0.3, 1.2, 1.0, 1.5])
DEFAULT_BIAS = -0.05


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


def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def open_exchange(name: str):
    exchange_cls = getattr(ccxt, name)
    exchange = exchange_cls({
        'enableRateLimit': True,
    })
    return exchange


def fetch_ohlcv(exchange, symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.reset_index(drop=True)
    return df


def fetch_orderbook(exchange, symbol, depth=ORDERBOOK_DEPTH):
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        return ob
    except Exception:
        return {'bids': [], 'asks': []}


def fetch_recent_trades(exchange, symbol, since=None, limit=1000):
    try:
        trades = exchange.fetch_trades(symbol, since=since, limit=limit)
        return trades
    except Exception:
        try:
            return exchange.fetch_trades(symbol, limit=limit)
        except Exception:
            return []


def orderbook_imbalance(orderbook, depths=(5, 10, 20)):
    # compute imbalance at multiple depths and return a small vector (averaged)
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])

    results = []
    for d in depths:
        bid_vol = sum([b[1] for b in bids[:d]])
        ask_vol = sum([a[1] for a in asks[:d]])
        if bid_vol + ask_vol == 0:
            results.append(0.0)
        else:
            results.append((bid_vol - ask_vol) / (bid_vol + ask_vol))
    # return a single scalar collapsed (weighted) for the live feature
    return float(np.mean(results))


def trade_imbalance(trades, orderbook=None):
    buy_vol = 0.0
    sell_vol = 0.0
    mid = None
    if orderbook and orderbook.get('bids') and orderbook.get('asks'):
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid = (best_bid + best_ask) / 2.0

    for t in trades:
        amount = float(t.get('amount', t.get('size', 0) or 0))
        price = t.get('price') or t.get('priceUsd') or None
        side = t.get('side') or t.get('takerSide') or None

        if not side:
            # infer from price vs mid if possible
            if price is not None and mid is not None:
                try:
                    p = float(price)
                    side = 'buy' if p >= mid else 'sell'
                except Exception:
                    side = None

        if side in ('buy', 'Buy', 'b', 'taker'):
            buy_vol += amount
        elif side in ('sell', 'Sell', 's', 'maker'):
            sell_vol += amount
        else:
            # ignore
            pass

    if buy_vol + sell_vol == 0:
        return 0.0
    return (buy_vol - sell_vol) / (buy_vol + sell_vol)


def compute_features(df, orderbook=None, trades=None):
    # Build the same 7 features as before; if orderbook/trades unavailable, those are 0
    features = []

    # last 5m return
    last_return = (df['close'].iloc[-1] - df['open'].iloc[-1]) / (df['open'].iloc[-1] + 1e-12)
    features.append(last_return)

    # short-term momentum: return of last 3 candles
    idx = len(df) - 1
    prev_idx = max(0, idx - 3)
    recent_return = (df['close'].iloc[idx] - df['close'].iloc[prev_idx]) / (df['close'].iloc[prev_idx] + 1e-12)
    features.append(recent_return)

    # EMA slope (10 vs 30) of close
    e10 = ema(df['close'], span=10)
    e30 = ema(df['close'], span=30)
    ema_slope = (e10.iloc[-1] - e30.iloc[-1]) / (df['close'].iloc[-1] + 1e-12)
    features.append(ema_slope)

    # RSI on closes (14)
    r = rsi(df['close'], period=14)
    rsi_last = (r.iloc[-1] - 50) / 50.0
    features.append(rsi_last)

    # orderbook imbalance
    ob_imb = 0.0
    if orderbook:
        ob_imb = orderbook_imbalance(orderbook)
    features.append(ob_imb)

    # trade imbalance
    t_imb = 0.0
    if trades:
        t_imb = trade_imbalance(trades, orderbook=orderbook)
    features.append(t_imb)

    # volume spike
    med_vol = np.median(df['volume'].iloc[-21:-1]) if len(df) > 21 else np.median(df['volume'].iloc[:-1]) if len(df) > 2 else df['volume'].iloc[-1]
    vol_spike = (df['volume'].iloc[-1] - med_vol) / (med_vol + 1e-12)
    features.append(vol_spike)

    return np.array(features, dtype=float)


def predict_probability(features, model=None):
    # model: dict with weights, bias, mean, std
    if model is None:
        # fallback to heuristic
        w = DEFAULT_WEIGHTS
        b = DEFAULT_BIAS
        score = float(np.dot(w, features) + b)
        return float(sigmoid(score)), score

    mean = model['mean']
    std = model['std']
    w = model['weights']
    b = model['bias']

    # ensure same feature length
    x = (features - mean) / (std + 1e-12)
    score = float(np.dot(w, x) + b)
    return float(sigmoid(score)), score


def save_model(weights, bias, mean, std, path=MODEL_PATH):
    np.savez(path, weights=weights, bias=bias, mean=mean, std=std)


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    npz = np.load(path)
    return {
        'weights': npz['weights'],
        'bias': float(npz['bias']),
        'mean': npz['mean'],
        'std': npz['std']
    }


def build_dataset_from_ohlcv(df):
    # df must be reasonably long. We'll generate one sample per candle excluding the last one
    rows = []
    for i in range(30, len(df) - 1):
        window = df.iloc[:i+1].copy()
        # compute features up to index i
        f = []
        last_return = (window['close'].iloc[-1] - window['open'].iloc[-1]) / (window['open'].iloc[-1] + 1e-12)
        f.append(last_return)
        prev_idx = max(0, len(window) - 4)
        recent_return = (window['close'].iloc[-1] - window['close'].iloc[prev_idx]) / (window['close'].iloc[prev_idx] + 1e-12)
        f.append(recent_return)
        e10 = ema(window['close'], span=10)
        e30 = ema(window['close'], span=30)
        ema_slope = (e10.iloc[-1] - e30.iloc[-1]) / (window['close'].iloc[-1] + 1e-12)
        f.append(ema_slope)
        r = rsi(window['close'], period=14)
        rsi_last = (r.iloc[-1] - 50) / 50.0
        f.append(rsi_last)
        # orderbook/trade features not available historically -> 0
        f.append(0.0)
        f.append(0.0)
        med_vol = np.median(window['volume'].iloc[-21:-1]) if len(window) > 21 else np.median(window['volume'].iloc[:-1]) if len(window) > 2 else window['volume'].iloc[-1]
        vol_spike = (window['volume'].iloc[-1] - med_vol) / (med_vol + 1e-12)
        f.append(vol_spike)

        # target is whether next candle closed green
        next_open = df['open'].iloc[i+1]
        next_close = df['close'].iloc[i+1]
        y = 1 if next_close > next_open else 0

        rows.append((f, y))

    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    return X, y


def train_logistic_gd(X, y, epochs=2000, lr=0.1, l2=1e-4, verbose=True):
    n, d = X.shape
    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Xs = (X - mean) / (std + 1e-12)

    # init
    w = np.zeros(d)
    b = 0.0

    for epoch in range(epochs):
        z = Xs.dot(w) + b
        p = sigmoid(z)
        # gradients
        error = p - y
        gw = (Xs.T.dot(error) / n) + l2 * w
        gb = error.mean()
        w -= lr * gw
        b -= lr * gb

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + (l2 / 2) * np.sum(w * w)
            print(f"Epoch {epoch}/{epochs} loss={loss:.6f}")

    return w, b, mean, std


def backtest_model(model, X, y):
    mean = model['mean']
    std = model['std']
    w = model['weights']
    b = model['bias']
    Xs = (X - mean) / (std + 1e-12)
    probs = sigmoid(Xs.dot(w) + b)
    preds = (probs >= 0.5).astype(int)

    acc = (preds == y).mean()
    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    print(f"Backtest samples: {len(y)} accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f}")


def log_prediction(timestamp, features, prob, model_name='trained'):
    cols = ['ts', 'model', 'prob'] + [f'f{i}' for i in range(len(features))]
    row = [timestamp.isoformat(), model_name, f"{prob:.6f}"] + [f"{v:.6f}" for v in features]
    write_header = not os.path.exists(PREDICTION_LOG)
    with open(PREDICTION_LOG, 'a') as f:
        if write_header:
            f.write(','.join(cols) + '\n')
        f.write(','.join(row) + '\n')


def align_to_next_timeframe(timeframe='5m'):
    now = datetime.now(timezone.utc)
    next_min = ((now.minute // 5) + 1) * 5
    if next_min >= 60:
        next_dt = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_min, second=0, microsecond=0)
    wait_seconds = (next_dt - now).total_seconds()
    return wait_seconds


def run_live(exchange_name, symbol):
    print(f"Starting live BTC 5m prediction on {exchange_name} {symbol}")
    exchange = open_exchange(exchange_name)

    model = load_model()
    if model:
        print("Loaded trained model from disk")
    else:
        print("No trained model found - falling back to heuristic weights")

    while True:
        try:
            wait = align_to_next_timeframe(TIMEFRAME)
            print(f"Waiting {int(wait)}s until next 5m boundary...")
            time.sleep(max(1, wait))

            df = fetch_ohlcv(exchange, symbol, limit=200)
            orderbook = fetch_orderbook(exchange, symbol)
            since_ms = int((datetime.now(timezone.utc).timestamp() - TRADE_LOOKBACK_SECONDS) * 1000)
            trades = fetch_recent_trades(exchange, symbol, since=since_ms)

            features = compute_features(df, orderbook=orderbook, trades=trades)
            prob, score = predict_probability(features, model=model)

            now = datetime.now().astimezone()
            print("-------------------------------------------------------------")
            print(now.isoformat(), f"Prediction next {TIMEFRAME} candle CLOSE GREEN: {prob*100:.2f}%")
            print(f"Raw score: {score:.4f}")
            for i, v in enumerate(features):
                print(f" f{i}: {v:.6f}")
            print("-------------------------------------------------------------")

            # log for future analysis
            log_prediction(now, features, prob, model_name='trained' if model else 'heuristic')

            # sleep a short while before re-aligning
            time.sleep(2)

        except KeyboardInterrupt:
            print("Stopping by user")
            break
        except Exception as e:
            print("Error in live loop:", e)
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train model from historical OHLCV')
    parser.add_argument('--backtest', action='store_true', help='Backtest saved model on historical OHLCV')
    parser.add_argument('--live', action='store_true', help='Run live prediction (default)')
    parser.add_argument('--exchange', default=EXCHANGE_NAME)
    parser.add_argument('--symbol', default=SYMBOL)
    args = parser.parse_args()

    exchange = open_exchange(args.exchange)

    if args.train:
        print("Fetching historical OHLCV for training...")
        df = fetch_ohlcv(exchange, args.symbol, limit=OHLCV_LIMIT)
        X, y = build_dataset_from_ohlcv(df)
        print(f"Built dataset X={X.shape} y={y.shape}")
        # split
        split = int(len(y) * 0.8)
        Xtr, ytr = X[:split], y[:split]
        Xval, yval = X[split:], y[split:]

        w, b, mean, std = train_logistic_gd(Xtr, ytr, epochs=1200, lr=0.2, l2=1e-4, verbose=True)
        save_model(w, b, mean, std)
        print("Saved model to", MODEL_PATH)

        model = load_model()
        if model:
            print("Evaluating on validation set...")
            backtest_model(model, Xval, yval)

    elif args.backtest:
        model = load_model()
        if not model:
            print("No model found. Run --train first")
            return
        df = fetch_ohlcv(exchange, args.symbol, limit=OHLCV_LIMIT)
        X, y = build_dataset_from_ohlcv(df)
        # use last 20% as test
        split = int(len(y) * 0.8)
        Xval, yval = X[split:], y[split:]
        backtest_model(model, Xval, yval)

    else:
        # default live
        run_live(args.exchange, args.symbol)


if __name__ == '__main__':
    main()
