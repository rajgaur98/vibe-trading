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
        # 1. First, check which symbol/timeframe pairs actually need bootstrapping
        to_bootstrap = []
        db.connect()
        try:
            for symbol in symbols:
                for tf in timeframes:
                    count_res = db.conn.execute(
                        "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?", (symbol, tf)
                    ).fetchone()
                    
                    if not count_res or count_res[0] <= 100:
                        to_bootstrap.append((symbol, tf))
                    else:
                        logger.info(f"Existing candles found for {symbol} on {tf}: {count_res[0]}. Skipping bootstrap.")
        finally:
            db.close()

        # 2. Loop through and download historical candles (network IO, DB is CLOSED!)
        for symbol, tf in to_bootstrap:
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
                logger.info(f"Writing {len(full_df)} candles to DuckDB for {symbol} on {tf}...")
                
                # 3. Open connection, write in bulk, and close immediately
                db.connect()
                try:
                    for _, row in full_df.iterrows():
                        db.conn.execute("""
                            INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (symbol, tf, row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']))
                finally:
                    db.close()

    def incremental_update(self, db: Database, symbols: list, timeframes: list, limit: int = 20):
        """Fetches the latest candles incrementally and writes to the DB."""
        # 1. Fetch candles for all symbol/timeframe pairs (network calls first, DB is CLOSED!)
        fetched_data = []
        for symbol in symbols:
            for tf in timeframes:
                df = self.fetch_ohlcv(symbol, tf, limit=limit)
                if not df.empty:
                    fetched_data.append((symbol, tf, df))
        
        # 2. Write all candles to DB in a single short-lived connection
        if fetched_data:
            db.connect()
            try:
                for symbol, tf, df in fetched_data:
                    for _, row in df.iterrows():
                        db.conn.execute("""
                            INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (symbol, tf, row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']))
            finally:
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

    def fetch_trending_symbols(self, limit: int = 10) -> list:
        """
        Fetches popular/trending symbols.
        Tries CoinGecko trending search coins first, falling back to Binance quote volume.
        """
        import urllib.request
        import json
        
        # 1. Try CoinGecko search/trending first
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            logger.info(f"Attempting to fetch trending coins from CoinGecko: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
            coingecko_coins = data.get("coins", [])
            trending_symbols = []
            
            # Load exchange markets to verify listing and symbols mapping
            markets = self.exchange.load_markets()
            stablecoins = {"USDC", "USDT", "BUSD", "DAI", "EUR", "FDUSD", "TUSD", "USD1", "USD", "GBP", "TRY", "PAX", "USDS"}
            
            for coin in coingecko_coins:
                item = coin.get("item", {})
                symbol = item.get("symbol", "").upper()
                binance_pair = f"{symbol}/USDT"
                
                # Filter stablecoins
                if symbol in stablecoins:
                    continue
                    
                if binance_pair in markets:
                    trending_symbols.append(binance_pair)
                    if len(trending_symbols) >= limit:
                        break
                        
            if trending_symbols:
                logger.info(f"Successfully fetched trending symbols from CoinGecko: {trending_symbols}")
                return trending_symbols
                
        except Exception as e:
            logger.warning(f"Failed to fetch trending coins from CoinGecko, falling back to volume-based tickers: {e}")
            
        # 2. Fallback to volume-based sorting (top USDT volume pairs on Binance)
        try:
            logger.info("Fetching tickers from Binance to determine top volume pairs...")
            tickers = self.exchange.fetch_tickers()
            usdt_tickers = []
            stablecoins = {"USDC", "USDT", "BUSD", "DAI", "EUR", "FDUSD", "TUSD", "USD1", "USD", "GBP", "TRY", "PAX", "USDS"}
            
            for symbol, ticker in tickers.items():
                if not symbol.endswith("/USDT"):
                    continue
                base = symbol.split("/")[0]
                if base in stablecoins:
                    continue
                if "UP/" in symbol or "DOWN/" in symbol or "BULL/" in symbol or "BEAR/" in symbol:
                    continue
                    
                usdt_tickers.append({
                    "symbol": symbol,
                    "quoteVolume": ticker.get("quoteVolume", 0)
                })
                
            usdt_tickers.sort(key=lambda x: x["quoteVolume"] or 0, reverse=True)
            trending_symbols = [x["symbol"] for x in usdt_tickers[:limit]]
            logger.info(f"Successfully fetched volume-based trending symbols from Binance: {trending_symbols}")
            return trending_symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch volume-based trending symbols: {e}")
            # Return static defaults as a last-resort fallback
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "NEAR/USDT", "BNB/USDT"]

    def bootstrap_if_needed(self, db: Database, symbols: list, timeframes: list):
        """Checks candle counts for symbols and runs bootstrap if they are missing or have insufficient data (< 200 candles)."""
        db.connect()
        symbols_to_bootstrap = []
        try:
            for symbol in symbols:
                needs_bootstrap = False
                for tf in timeframes:
                    count_res = db.conn.execute(
                        "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?", (symbol, tf)
                    ).fetchone()
                    count = count_res[0] if count_res else 0
                    if count < 200:
                        needs_bootstrap = True
                        logger.info(f"Symbol {symbol} has only {count} {tf} candles in DB (required >= 200). Needs bootstrapping.")
                        break
                if needs_bootstrap:
                    symbols_to_bootstrap.append(symbol)
        finally:
            db.close()
            
        if symbols_to_bootstrap:
            logger.info(f"Starting bootstrap for needed symbols: {symbols_to_bootstrap}")
            self.bootstrap(db, symbols_to_bootstrap, timeframes)
