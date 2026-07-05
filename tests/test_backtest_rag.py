from unittest.mock import MagicMock, patch

from vibe_trading.eval.backtest import BacktestEngine


def test_engine_defaults_to_no_rag():
    engine = BacktestEngine(MagicMock(), ["BTC/USDT"])
    assert engine.journal_rag is False
    assert engine.replay_journal is None


def test_live_decision_threads_precedents_and_records(monkeypatch):
    """With journal_rag on, the live-agent path retrieves precedents, passes them
    to the trader, and records the new decision into the replay journal."""
    from vibe_trading.journal import RetrievalResult

    engine = BacktestEngine(MagicMock(), ["BTC/USDT"], journal_rag=True)
    engine.replay_journal = MagicMock()
    engine.replay_journal.retrieve_for.return_value = RetrievalResult([0.1], [])

    fake_analyst = MagicMock()
    fake_trader = MagicMock()
    fake_trader.decide.return_value = {"decision_id": "dec-1", "action": "long",
                                       "symbol": "BTC/USDT",
                                       "stop_loss_strategy": "1.5_atr",
                                       "take_profit_strategy": "next_resistance",
                                       "risk_reward_ratio": 2.0,
                                       "hold_period_bias": "medium",
                                       "reasoning_summary": "r"}
    with patch("vibe_trading.agents.analyst.TechnicalVolumeAnalyst",
               return_value=fake_analyst), \
         patch("vibe_trading.agents.trader.HeadTrader", return_value=fake_trader), \
         patch("vibe_trading.data.fetcher.DataFetcher"):
        snapshot = {"close": 100.0, "rsi_14": 50.0, "obv_trend": "flat",
                    "macd_regime": "neutral"}
        from datetime import datetime
        proposal = engine._get_decision("BTC/USDT", snapshot,
                                        datetime(2026, 6, 1), use_live_agents=True)
    assert proposal["decision_id"] == "dec-1"
    engine.replay_journal.retrieve_for.assert_called_once()
    assert fake_trader.decide.call_args.kwargs["precedents"] == []
    engine.replay_journal.record_decision.assert_called_once()
    rec_args = engine.replay_journal.record_decision.call_args.args
    assert rec_args[0] == "dec-1"


def test_mock_path_rejects_journal_rag():
    import pytest
    engine = BacktestEngine(MagicMock(), ["BTC/USDT"], journal_rag=True)
    engine.replay_journal = MagicMock()
    with pytest.raises(ValueError):
        engine._get_decision("BTC/USDT", {"close": 1.0, "rsi_14": 50.0,
                                          "obv_trend": "flat",
                                          "macd_regime": "neutral"},
                             None, use_live_agents=False)
