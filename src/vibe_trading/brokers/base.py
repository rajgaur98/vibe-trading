from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseBroker(ABC):
    @abstractmethod
    def get_balance(self) -> float:
        """Returns the current liquid account balance in USD."""
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Returns a list of active open positions.
        Format: [{'symbol': 'BTC/USD', 'side': 'long', 'entry_price': 50000.0, 'size_usd': 100.0, 'stop_price': 48000.0, 'take_profit_price': 55000.0}]
        """
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        action: str,  # 'long' or 'short'
        size_usd: float,
        stop_price: float,
        take_profit_price: float,
        entry_price: float = 0.0,
        decision_id: str = None,
    ) -> Dict[str, Any]:
        """Submits an entry order with OCO stop and profit targets.

        `entry_price`: when > 0, the position is filled at submission (live paper trading
        path). When 0.0, the broker may lazily fill `entry_price` on the next
        `update_positions` tick (backtester path that simulates next-candle fill).

        `decision_id`: the decision_log id that produced this order. It is persisted on
        the open position and carried through to the resulting closed trade, so the
        outcome can be joined back to the decision that opened it (None when unknown).
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Exits an active position at market price."""
        pass

    @abstractmethod
    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Checks for SL/TP fills based on latest prices.
        Mainly utilized by paper trading and backtesters.
        """
        pass

    def get_mark_price(self, symbol: str) -> Optional[float]:
        """Execution-critical price for `symbol`, or None if this broker has no
        live price source. The scheduler uses this in LIVE_TESTNET to align entry/
        proximity decisions to the traded instrument's mark; brokers that return
        None cause the scheduler to fall back to the DuckDB spot close.
        """
        return None
