import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from vibe_trading.data.db import Database

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        # Use Binance public exchange client for free historical OHLCV data bootstrapping
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        # Use Binance futures client for derivatives metrics (funding rates, open interest)
        self.futures_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int = None, limit: int = 500) -> pd.DataFrame:
        """Fetches OHLCV from the exchange."""
        # Convert CCXT symbol format if needed (e.g., BTC/USD or BTC/USDT)
        # Spot market uses USDT or USD
        exchange_symbol = symbol.replace("USD", "USDT") if "USDT" not in symbol else symbol
        
        try:
            logger.info(f"Fetching {limit} candles for {exchange_symbol} on {timeframe}...")
            candles = self.exchange.fetch_ohlcv(exchange_symbol, timeframe, since=since_ms, limit=limit)
            if not candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} ({timeframe}): {e}")
            return pd.DataFrame()

    def bootstrap(self, db: Database, symbols: list, timeframes: list):
        """Downloads historical candles for the warm-up window."""
        # Check if database already has data to prevent duplicate downloading
        db.connect()
        for symbol in symbols:
            for tf in timeframes:
                # Check count
                count_res = db.conn.execute(
                    "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?", (symbol, tf)
                ).fetchone()
                
                if count_res and count_res[0] > 100:
                    logger.info(f"Existing candles found for {symbol} on {tf}: {count_res[0]}. Skipping bootstrap.")
                    continue
                
                # Determine how far back to bootstrap
                if tf == '1d':
                    days_back = 365 * 2  # 2 years
                else:  # '4h' or other shorter timeframes
                    days_back = 30 * 6   # 6 months
                
                since_time = datetime.utcnow() - timedelta(days=days_back)
                since_ms = int(since_time.timestamp() * 1000)
                
                all_dfs = []
                current_since = since_ms
                
                # Paginate downloads (ccxt returns max 500-1000 per request)
                while True:
                    df = self.fetch_ohlcv(symbol, tf, since_ms=current_since, limit=1000)
                    if df.empty:
                        break
                    
                    all_dfs.append(df)
                    
                    # Set next since_ms to last candle timestamp + 1 ms
                    last_ts = int(df['timestamp'].iloc[-1].timestamp() * 1000)
                    if last_ts <= current_since:
                        break
                    
                    current_since = last_ts + 1000
                    
                    # Respect rate limits
                    time.sleep(self.exchange.rateLimit / 1000)
                    if len(df) < 500:
                        break
                
                if all_dfs:
                    full_df = pd.concat(all_dfs).drop_duplicates(subset=['timestamp'])
                    # Write to DuckDB
                    logger.info(f"Writing {len(full_df)} candles to DuckDB for {symbol} on {tf}...")
                    for _, row in full_df.iterrows():
                        db.conn.execute("""
                            INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (symbol, tf, row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']))
        db.close()

    def incremental_update(self, db: Database, symbols: list, timeframes: list, limit: int = 20):
        """Fetches the latest candles incrementally and writes to the DB."""
        db.connect()
        for symbol in symbols:
            for tf in timeframes:
                df = self.fetch_ohlcv(symbol, tf, limit=limit)
                if not df.empty:
                    for _, row in df.iterrows():
                        db.conn.execute("""
                            INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (symbol, tf, row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']))
        db.close()

    def fetch_funding_rate_and_oi(self, symbol: str) -> dict:
        """Fetches funding rate and open interest for a symbol."""
        exchange_symbol = symbol.replace("USD", "USDT") if "USDT" not in symbol else symbol
        # CCXT futures markets require symbol formatted as asset/USDT:USDT (e.g. BTC/USDT:USDT)
        futures_symbol = f"{exchange_symbol.split('/')[0]}/USDT:USDT" if "/" in exchange_symbol else f"{exchange_symbol}:USDT"
        
        result = {
            "funding_rate": "0.00% (neutral)",
            "open_interest_trend": "flat (stable)"
        }
        
        try:
            # 1. Fetch funding rate
            funding_info = self.futures_exchange.fetch_funding_rate(futures_symbol)
            rate = funding_info.get('fundingRate', 0)
            rate_pct = rate * 100
            
            # Categorize funding rate
            if rate_pct >= 0.05:
                regime = "extremely_high_long_crowding"
            elif rate_pct >= 0.02:
                regime = "high_long_crowding"
            elif rate_pct <= -0.05:
                regime = "extremely_high_short_crowding"
            elif rate_pct <= -0.02:
                regime = "high_short_crowding"
            else:
                regime = "neutral"
                
            result["funding_rate"] = f"{rate_pct:.4f}% ({regime})"
            
            # 2. Fetch Open Interest (some endpoints might have limits, so wrap in try-except)
            try:
                oi_info = self.futures_exchange.fetch_open_interest(futures_symbol)
                oi_val = oi_info.get('openInterest', 0)
                # Note: To determine the trend, we'd need historical OI, which is often not supported
                # on public spots. We can just log the value or check if it's high.
                # For simplicity, if we don't have historical, we label it as 'active'.
                result["open_interest_trend"] = f"{oi_val:,.0f} USD value (active)"
            except Exception as e:
                logger.warning(f"Could not fetch Open Interest for {symbol}: {e}")
                result["open_interest_trend"] = "data_unavailable"
                
        except Exception as e:
            logger.error(f"Error fetching derivatives data for {symbol}: {e}")
            
        return result
