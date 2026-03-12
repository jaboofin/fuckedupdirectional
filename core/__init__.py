"""Core package exports.

Keep optional dependency imports lazy/guarded so unit tests for pure modules
(e.g. regime filter, position sizer) do not fail at import time.
"""

try:
    from core.polymarket_client import PolymarketClient, BinaryMarket, TradeRecord
except ModuleNotFoundError:  # optional runtime deps like aiohttp may be absent in tests
    PolymarketClient = None
    BinaryMarket = None
    TradeRecord = None

from core.risk_manager import RiskManager
from core.trade_logger import TradeLogger

__all__ = [
    "PolymarketClient",
    "BinaryMarket",
    "TradeRecord",
    "RiskManager",
    "TradeLogger",
]
