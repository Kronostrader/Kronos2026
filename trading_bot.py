import ccxt
import pandas as pd
import numpy as np
import time

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.linear_model import LogisticRegression

# Exchange setup
exchange = ccxt.binance()

symbol = "BTC/USDT"
timeframe = "5m"
limit = 500


def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    return df


def add_indicators(df):
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    df["ema_9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(df["close"], window=21).ema_indicator()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["return"] = df["close"].pct_change()

    # Target: next candle direction
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna()

    return df


def train_model(df):
    features = ["rsi", "ema_9", "ema_21", "macd", "macd_signal", "return"]

    X = df[features]
    y = df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, features


def predict(model, features, df):
    latest = df[features].iloc[-1].values.reshape(1, -1)

    prob_up = model.predict_proba(latest)[0][1]
    prob_down = 1 - prob_up

    return prob_up, prob_down


def main():
    while True:
        try:
            df = fetch_data()
            df = add_indicators(df)

            model, features = train_model(df)

            prob_up, prob_down = predict(model, features, df)

            print("\n========================")
            print("BTC 5M PREDICTION")
            print(f"UP probability:   {prob_up:.2%}")
            print(f"DOWN probability: {prob_down:.2%}")

            if prob_up > 0.55:
                print("🟢 BIAS: LONG")
            elif prob_down > 0.55:
                print("🔴 BIAS: SHORT")
            else:
                print("⚪ NO EDGE")

            print("========================\n")

        except Exception as e:
            print("Error:", e)

        time.sleep(60)


if __name__ == "__main__":
    main()
