"""Real-time prediction service using Binance trades websocket + ccxt orderbook polling.

Run with:
  pip install -r requirements.txt
  uvicorn realtime_service:app --host 0.0.0.0 --port 8000

Endpoints:
  GET /predict -> current probability and features

Notes:
 - Relies on functions from main.py (compute_features, fetch_ohlcv, fetch_orderbook, load_model, predict_probability)
 - Uses Binance trade websocket to collect recent trades and ccxt to poll orderbook and OHLCV frequently.
"""

import os
import time
import threading
import json
from collections import deque
from datetime import datetime, timezone

import ccxt
import pandas as pd
from websocket import WebSocketApp
from fastapi import FastAPI

# import helpers from main.py
from main import fetch_ohlcv, fetch_orderbook, compute_features, load_model, predict_probability, open_exchange

# Configuration
EXCHANGE = os.getenv('EXCHANGE', 'binance')
SYMBOL = os.getenv('SYMBOL', 'BTC/USDT')
TRADE_LOOKBACK_SECONDS = int(os.getenv('TRADE_LOOKBACK_SECONDS', 60 * 3))
ORDERBOOK_POLL_SECONDS = float(os.getenv('ORDERBOOK_POLL_SECONDS', 1.0))
OHLCV_LIMIT = int(os.getenv('OHLCV_LIMIT', 300))

# Globals shared with FastAPI
trade_deque = deque()
trade_lock = threading.Lock()
current_features = None
current_prob = None
last_update = None
model = None

app = FastAPI()

# Binance websocket symbol format
def binance_trade_stream_symbol(symbol):
    s = symbol.replace('/', '').lower()
    return f"{s}@trade"


def on_trade_message(ws, message):
    try:
        msg = json.loads(message)
        # Binance trade stream format: p=price, q=quantity, T=tradeTime, m=isBuyerMaker
        price = float(msg.get('p'))
        qty = float(msg.get('q'))
        ts = int(msg.get('T'))
        is_buyer_maker = msg.get('m')
        side = 'sell' if is_buyer_maker else 'buy'
        trade = {'price': price, 'amount': qty, 'timestamp': ts, 'side': side}
        with trade_lock:
            trade_deque.append(trade)
            # drop old
            cutoff = int(time.time() * 1000) - TRADE_LOOKBACK_SECONDS * 1000
            while trade_deque and trade_deque[0]['timestamp'] < cutoff:
                trade_deque.popleft()
    except Exception as e:
        print('Error parsing trade message:', e)


def on_error(ws, error):
    print('WebSocket error:', error)


def on_close(ws, close_status_code, close_msg):
    print('WebSocket closed', close_status_code, close_msg)


def on_open(ws):
    print('WebSocket connection opened')


def start_trade_ws(symbol):
    stream = binance_trade_stream_symbol(symbol)
    url = f"wss://stream.binance.com:9443/ws/{stream}"
    ws = WebSocketApp(url, on_message=on_trade_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    # run forever
    wst = threading.Thread(target=ws.run_forever, kwargs={'ping_interval': 20, 'ping_timeout': 10})
    wst.daemon = True
    wst.start()
    return ws


def run_realtime_loop(exchange_name, symbol):
    global current_features, current_prob, last_update, model
    print(f'Starting realtime loop for {exchange_name} {symbol}')
    exchange = open_exchange(exchange_name)
    model = load_model()
    if model:
        print('Loaded trained model for realtime')
    else:
        print('No trained model found; realtime will use heuristic fallback')

    # Start websocket for trades (Binance only currently)
    try:
        start_trade_ws(symbol)
    except Exception as e:
        print('Failed to start websocket trade stream:', e)

    while True:
        try:
            # fetch OHLCV and orderbook
            df = fetch_ohlcv(exchange, symbol, limit=OHLCV_LIMIT)
            orderbook = fetch_orderbook(exchange, symbol)

            # build trades list from deque
            with trade_lock:
                trades = list(trade_deque)

            # convert trades to format compute_features expects (list of dicts with amount and side)
            trades_fmt = []
            for t in trades:
                trades_fmt.append({'amount': t['amount'], 'price': t['price'], 'side': t['side']})

            features = compute_features(df, orderbook=orderbook, trades=trades_fmt)
            prob, score = predict_probability(features, model=model)

            current_features = features.tolist()
            current_prob = float(prob)
            last_update = datetime.now(timezone.utc).isoformat()

            # print for operator
            print(f"{last_update} realtime prob: {prob*100:.2f}% score={score:.4f}")

            time.sleep(ORDERBOOK_POLL_SECONDS)

        except Exception as e:
            print('Error in realtime loop:', e)
            time.sleep(2)


@app.get('/predict')
def get_prediction():
    return {
        'symbol': SYMBOL,
        'probability': current_prob,
        'features': current_features,
        'last_update': last_update,
    }


# Start background thread when module imported (so uvicorn will spawn it)
_thread = threading.Thread(target=run_realtime_loop, args=(EXCHANGE, SYMBOL), daemon=True)
_thread.start()
