"""
Data Loader Module
Fetches historical OHLCV data for backtesting
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


def fetch_btc_data(symbol: str = "BTC-USD", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch BTC historical OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Trading pair symbol
        period: Data period (1y, 2y, 5y)
        interval: Candle interval (1d, 1h, 4h)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"📥 Fetching {symbol} data ({period}, {interval} interval)...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    # Normalize columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index.name = 'date'
    df.dropna(inplace=True)
    
    print(f"✅ Fetched {len(df)} candles | From: {df.index[0].date()} To: {df.index[-1].date()}")
    return df


def add_technical_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Add technical indicators based on strategy parameters.
    
    Args:
        df: OHLCV DataFrame
        params: Strategy parameters dict
    
    Returns:
        DataFrame with indicators
    """
    df = df.copy()
    
    ema_fast = int(params.get('ema_fast', 10))
    ema_slow = int(params.get('ema_slow', 30))
    rsi_period = int(params.get('rsi_period', 14))
    atr_period = int(params.get('atr_period', 14))
    bb_period = int(params.get('bb_period', 20))
    bb_std = float(params.get('bb_std', 2.0))

    # EMA
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=atr_period).mean()

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    bb_std_val = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * bb_std_val)
    df['bb_lower'] = df['bb_mid'] - (bb_std * bb_std_val)

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Volume MA
    df['volume_ma'] = df['volume'].rolling(window=20).mean()

    df.dropna(inplace=True)
    return df


def generate_synthetic_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing when API is unavailable.
    Uses Geometric Brownian Motion.
    """
    print("🔧 Generating synthetic BTC data (GBM simulation)...")
    
    np.random.seed(42)
    dt = 1 / 365
    mu = 0.5       # drift (annual return)
    sigma = 0.8    # volatility (annual)
    
    prices = [30000.0]
    for _ in range(n_periods - 1):
        shock = np.random.normal(mu * dt, sigma * np.sqrt(dt))
        prices.append(prices[-1] * np.exp(shock))
    
    closes = np.array(prices)
    
    df = pd.DataFrame({
        'open': closes * (1 + np.random.uniform(-0.01, 0.01, n_periods)),
        'high': closes * (1 + np.random.uniform(0.005, 0.025, n_periods)),
        'low': closes * (1 - np.random.uniform(0.005, 0.025, n_periods)),
        'close': closes,
        'volume': np.random.uniform(1000, 10000, n_periods) * closes / 30000
    }, index=pd.date_range(start='2022-01-01', periods=n_periods, freq='D'))
    
    df.index.name = 'date'
    print(f"✅ Generated {len(df)} synthetic candles")
    return df


def load_data(use_live: bool = True, **kwargs) -> pd.DataFrame:
    """Main data loading function with fallback to synthetic."""
    if use_live:
        try:
            return fetch_btc_data(**kwargs)
        except Exception as e:
            print(f"⚠️  Live data failed: {e}")
            print("📊 Falling back to synthetic data...")
            return generate_synthetic_data()
    else:
        return generate_synthetic_data()


if __name__ == "__main__":
    df = load_data()
    print(df.tail())
    print(f"\nShape: {df.shape}")
