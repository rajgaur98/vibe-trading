import pytest
from unittest.mock import MagicMock, patch
from vibe_trading.data.fetcher import DataFetcher

def test_fetch_trending_symbols_fallback():
    fetcher = DataFetcher()
    
    # Mock CCXT Binance fetch_tickers
    mock_tickers = {
        "BTC/USDT": {"quoteVolume": 1000000.0, "close": 50000.0},
        "ETH/USDT": {"quoteVolume": 500000.0, "close": 3000.0},
        "USDC/USDT": {"quoteVolume": 2000000.0, "close": 1.0}, # stablecoin
        "SOL/USDT": {"quoteVolume": 800000.0, "close": 100.0},
        "BTCUP/USDT": {"quoteVolume": 100000.0, "close": 2.0}, # leveraged token
        "ETH/BTC": {"quoteVolume": 300000.0, "close": 0.06}, # non-USDT quote
    }
    
    fetcher.exchange.fetch_tickers = MagicMock(return_value=mock_tickers)
    
    # Force failure on CoinGecko to test volume fallback
    with patch("urllib.request.urlopen", side_effect=Exception("API Down")):
        symbols = fetcher.fetch_trending_symbols(limit=3)
        
    # Expected top 3 tradeable sorted by quoteVolume (stablecoins/leveraged excluded):
    # 1. BTC/USDT (1000000.0)
    # 2. SOL/USDT (800000.0)
    # 3. ETH/USDT (500000.0)
    assert symbols == ["BTC/USDT", "SOL/USDT", "ETH/USDT"]

def test_fetch_trending_symbols_coingecko():
    fetcher = DataFetcher()
    
    # Mock markets
    mock_markets = {
        "NEAR/USDT": {},
        "BTC/USDT": {},
        "ETH/USDT": {},
    }
    fetcher.exchange.load_markets = MagicMock(return_value=mock_markets)
    
    # Mock CoinGecko return json
    mock_response = MagicMock()
    mock_response.read.return_value = b"""
    {
      "coins": [
        {"item": {"symbol": "NEAR", "name": "NEAR Protocol"}},
        {"item": {"symbol": "USDC", "name": "USD Coin"}},
        {"item": {"symbol": "BTC", "name": "Bitcoin"}},
        {"item": {"symbol": "HYPE", "name": "Hyperliquid"}},
        {"item": {"symbol": "ETH", "name": "Ethereum"}}
      ]
    }
    """
    mock_response.__enter__.return_value = mock_response
    
    with patch("urllib.request.urlopen", return_value=mock_response):
        symbols = fetcher.fetch_trending_symbols(limit=3)
        
    # Expected symbols: PENGU/USDT not in mock market, USDC excluded stablecoin, HYPE not listed,
    # NEAR/USDT, BTC/USDT, ETH/USDT are listed on mock Binance.
    assert symbols == ["NEAR/USDT", "BTC/USDT", "ETH/USDT"]
