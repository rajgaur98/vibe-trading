from abc import ABC, abstractmethod
from typing import Dict, Any, List

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
        take_profit_price: float
    ) -> Dict[str, Any]:
        """Submits an entry order with OCO stop and profit targets."""
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
