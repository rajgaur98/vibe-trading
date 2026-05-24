import pandas as pd
import numpy as np
from datetime import datetime
import talib
from scipy.signal import find_peaks
import logging
from vibe_trading.data.db import Database
from vibe_trading.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)

class FeaturePipeline:
    def __init__(self, db: Database):
        self.db = db
        self.fetcher = DataFetcher()

    def run(self, symbol: str, timestamp: datetime) -> dict:
        """
        Calculates all features for a given symbol at a specific candle timestamp
        and returns a Pydantic-compatible snapshot dictionary.
        """
        self.db.connect()
        try:
            # 1. Fetch historical candles up to the target timestamp from DuckDB
            # We need enough history for indicator warm-ups (e.g. 200 days for 1D, 200 candles for 4h)
            df_4h = self._get_candles(symbol, "4h", timestamp, limit=300)
            df_1d = self._get_candles(symbol, "1d", timestamp, limit=300)

            if len(df_4h) < 200 or len(df_1d) < 200:
                logger.warning(f"Not enough candles fetched to calculate features for {symbol} at {timestamp}. Required: 200. Got: 4h={len(df_4h)}, 1d={len(df_1d)}")
                return {}

            # 2. Calculate Indicators for 4h and 1d Timeframes
            feats_4h = self._calculate_indicators(df_4h)
            feats_1d = self._calculate_indicators(df_1d)

            # Get latest rows (the candle being evaluated)
            latest_4h = feats_4h.iloc[-1]
            latest_1d = feats_1d.iloc[-1]

            # 3. Detect Support and Resistance levels using scipy.signal.find_peaks
            sr_levels = self._detect_support_resistance(df_4h)
            current_price = float(latest_4h['close'])
            
            # Calculate support & resistance distances
            support_price, support_dist_pct, support_proximity = self._get_closest_level(current_price, sr_levels['supports'])
            resistance_price, resistance_dist_pct, resistance_proximity = self._get_closest_level(current_price, sr_levels['resistances'])

            # 4. Candlestick patterns via TA-Lib
            candlestick_pattern = self._recognize_candlesticks(df_4h)

            # 5. Fetch Derivatives context
            derivatives = self.fetcher.fetch_funding_rate_and_oi(symbol)

            # 6. Check macro event
            # (Simply mock major events or check static calendar. Here we can check calendar day)
            is_macro_event = self._is_macro_event(timestamp)

            # Assemble the snapshot dictionary (categorizing raw floats to avoid LLM math)
            snapshot = {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": current_price,  # placeholder or raw values if needed
                "high": float(latest_4h['high']),
                "low": float(latest_4h['low']),
                "close": current_price,
                "volume": float(latest_4h['volume']),
                
                # Trend stack features
                "rsi_14": float(latest_4h['rsi_14']),
                "rsi_regime": self._get_rsi_regime(latest_4h['rsi_14']),
                
                "macd": float(latest_4h['macd']),
                "macd_signal": float(latest_4h['macd_signal']),
                "macd_hist": float(latest_4h['macd_hist']),
                "macd_regime": self._get_macd_regime(latest_4h['macd_hist']),
                
                "adx_14": float(latest_4h['adx_14']),
                "adx_regime": "strong_trend" if latest_4h['adx_14'] >= 25.0 else "weak_trend",
                
                "obv": float(latest_4h['obv']),
                "obv_trend": self._get_obv_trend(feats_4h),
                
                # S/R levels
                "support_price": support_price,
                "support_distance_pct": support_dist_pct,
                "support_proximity": support_proximity,
                
                "resistance_price": resistance_price,
                "resistance_distance_pct": resistance_dist_pct,
                "resistance_proximity": resistance_proximity,
                
                # Patterns and macro
                "candlestick_pattern": candlestick_pattern,
                "funding_rate": derivatives["funding_rate"],
                "open_interest_trend": derivatives["open_interest_trend"],
                "is_macro_event_today": is_macro_event
            }

            return snapshot
        finally:
            self.db.close()

    def _get_candles(self, symbol: str, timeframe: str, timestamp: datetime, limit: int) -> pd.DataFrame:
        """Fetches candles up to a specific timestamp from DuckDB."""
        should_close = False
        if not self.db.conn:
            self.db.connect()
            should_close = True
        try:
            # DuckDB query to get historical rows up to our current point in time
            res = self.db.conn.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = ? AND timeframe = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, timeframe, timestamp, limit)).fetchall()
            
            if not res:
                return pd.DataFrame()
                
            # Reverse because we want oldest to newest
            res.reverse()
            df = pd.DataFrame(res, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        finally:
            if should_close:
                self.db.close()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs standard TA-Lib indicator calculations on the candles."""
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values.astype(float)

        # 1. Moving Averages
        df['ma20'] = talib.SMA(closes, timeperiod=20)
        df['ma50'] = talib.SMA(closes, timeperiod=50)
        df['ma200'] = talib.SMA(closes, timeperiod=200)

        # 2. Oscillators
        df['rsi_14'] = talib.RSI(closes, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        df['adx_14'] = talib.ADX(highs, lows, closes, timeperiod=14)

        # 3. Volume
        df['obv'] = talib.OBV(closes, volumes)

        # Fill NaNs from warmups
        df = df.fillna(0.0)
        return df

    def _detect_support_resistance(self, df: pd.DataFrame) -> dict:
        """Identifies support and resistance zones using scipy signal find_peaks."""
        highs = df['high'].values
        lows = df['low'].values
        
        # Scipy find_peaks expects a 1D signal. 
        # Resistance: peaks in highs. Support: peaks in -lows (troughs).
        # We set standard parameters for prominence and distance
        res_indices, _ = find_peaks(highs, distance=10, prominence=0.01 * np.mean(highs))
        sup_indices, _ = find_peaks(-lows, distance=10, prominence=0.01 * np.mean(lows))
        
        resistances = sorted([float(highs[i]) for i in res_indices])
        supports = sorted([float(lows[i]) for i in sup_indices])
        
        return {
            "supports": supports,
            "resistances": resistances
        }

    def _get_closest_level(self, current_price: float, levels: list) -> tuple:
        """Finds the closest support/resistance price, distance pct, and proximity label."""
        if not levels:
            return 0.0, 0.0, "unknown"
            
        distances = [abs(current_price - l) for l in levels]
        min_idx = np.argmin(distances)
        closest_price = levels[min_idx]
        
        dist_pct = (abs(current_price - closest_price) / current_price) * 100
        
        if dist_pct < 0.5:
            proximity = "immediate_contact"
        elif dist_pct < 2.0:
            proximity = "very_close"
        elif dist_pct < 5.0:
            proximity = "near"
        else:
            proximity = "far"
            
        return closest_price, float(dist_pct), proximity

    def _recognize_candlesticks(self, df: pd.DataFrame) -> str:
        """Checks for engulfing, morning/evening star patterns using TA-Lib."""
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Run multiple TA-lib pattern recognizers (using last candle results)
        engulfing = talib.CDLENGULFING(opens, highs, lows, closes)[-1]
        morning_star = talib.CDLMORNINGSTAR(opens, highs, lows, closes)[-1]
        evening_star = talib.CDLEVENINGSTAR(opens, highs, lows, closes)[-1]
        hammer = talib.CDLHAMMER(opens, highs, lows, closes)[-1]
        shooting_star = talib.CDLSHOOTINGSTAR(opens, highs, lows, closes)[-1]

        patterns = []
        if engulfing != 0:
            patterns.append("engulfing_bullish" if engulfing > 0 else "engulfing_bearish")
        if morning_star != 0:
            patterns.append("morning_star")
        if evening_star != 0:
            patterns.append("evening_star")
        if hammer != 0:
            patterns.append("hammer_bullish")
        if shooting_star != 0:
            patterns.append("shooting_star_bearish")

        return ", ".join(patterns) if patterns else "none"

    def _get_rsi_regime(self, rsi: float) -> str:
        if rsi >= 70.0:
            return "overbought"
        elif rsi <= 30.0:
            return "oversold"
        elif rsi >= 60.0:
            return "approaching_overbought"
        elif rsi <= 40.0:
            return "approaching_oversold"
        else:
            return "neutral"

    def _get_macd_regime(self, macd_hist: float) -> str:
        if macd_hist > 0:
            return "bullish_momentum_expanding"
        elif macd_hist < 0:
            return "bearish_momentum_expanding"
        else:
            return "flat"

    def _get_obv_trend(self, df: pd.DataFrame) -> str:
        """Looks at the last 5 OBV values to determine trend."""
        obv = df['obv'].values
        if len(obv) < 5:
            return "flat"
        
        diff = np.diff(obv[-5:])
        positive_diffs = np.sum(diff > 0)
        
        if positive_diffs >= 4:
            return "accumulation"
        elif positive_diffs <= 1:
            return "distribution"
        else:
            return "neutral"

    def _is_macro_event(self, dt: datetime) -> bool:
        """Checks if a major event occurs on the given date (e.g. FOMC days placeholder)."""
        # In a real environment, we'd query an API or static list.
        # For our v1 spec, we check if the day of week is Wednesday (which is FOMC release day).
        return dt.weekday() == 2  # Wednesday is 2 in Python datetime (0-indexed starting Monday)
