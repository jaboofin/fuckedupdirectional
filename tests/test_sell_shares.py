"""
Tests for Phase 2 — sell execution path in polymarket_client.py.

Tests mock the CLOB SDK since it's not installed in test environments.

Run:  python tests/test_sell_shares.py
"""

import sys
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

sys.path.insert(0, ".")


class MockConfig:
    """Minimal config to construct PolymarketClient."""
    class polymarket:
        clob_api_url = "https://clob.polymarket.com"
        gamma_api_url = "https://gamma-api.polymarket.com"
        chain_id = 137
        rpc_url = "https://polygon-rpc.com"
        private_key = "0xfake"
        funder = "0xfake_funder"
        sig_type = 1
        market_slug_pattern = "btc-price"
        market_interval_minutes = 15
        order_type = "market"
        max_slippage_pct = 2.0
        min_liquidity_usd = 50.0
        sync_live_bankroll = False
        live_bankroll_poll_secs = 60
        fee_cache_ttl_secs = 60
        fee_fallback_pct = 1.56


def run_async(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestGetBestBid(unittest.TestCase):

    def setUp(self):
        from core.polymarket_client import PolymarketClient
        self.client = PolymarketClient(MockConfig())

    def test_best_bid_from_orderbook(self):
        """get_best_bid returns highest bid with depth info."""
        mock_book = {
            "bids": [
                {"price": 0.55, "size": 100},
                {"price": 0.54, "size": 200},
                {"price": 0.53, "size": 150},
            ],
            "asks": [{"price": 0.57, "size": 50}],
            "tick_size": "0.01",
        }
        with patch.object(self.client, "get_order_book", return_value=mock_book):
            result = self.client.get_best_bid("tok123")

        self.assertIsNotNone(result)
        self.assertEqual(result["price"], 0.55)
        self.assertEqual(result["size"], 100)
        self.assertEqual(result["levels"], 3)

    def test_best_bid_empty_book(self):
        """get_best_bid returns None for empty orderbook."""
        with patch.object(self.client, "get_order_book", return_value={"bids": [], "asks": []}):
            result = self.client.get_best_bid("tok123")
        self.assertIsNone(result)

    def test_best_bid_no_book(self):
        """get_best_bid returns None when orderbook fetch fails."""
        with patch.object(self.client, "get_order_book", return_value=None):
            result = self.client.get_best_bid("tok123")
        self.assertIsNone(result)

    def test_best_bid_single_level(self):
        """Works with a single bid level."""
        mock_book = {
            "bids": [{"price": 0.60, "size": 50}],
            "asks": [],
            "tick_size": "0.01",
        }
        with patch.object(self.client, "get_order_book", return_value=mock_book):
            result = self.client.get_best_bid("tok123")

        self.assertEqual(result["price"], 0.60)
        self.assertEqual(result["size"], 50)
        self.assertEqual(result["levels"], 1)


class TestSellShares(unittest.TestCase):

    def setUp(self):
        # Inject SDK constants into module namespace (py-clob-client not installed)
        import core.polymarket_client as pm
        pm.HAS_CLOB_SDK = True
        pm.OrderArgs = MagicMock()
        pm.SELL = "SELL"
        pm.BUY = "BUY"
        pm.OrderType = MagicMock()
        pm.OrderType.FOK = "FOK"
        pm.OrderType.GTC = "GTC"

        from core.polymarket_client import PolymarketClient
        self.client = PolymarketClient(MockConfig())
        self.client._clob_initialized = True
        self.client._clob = MagicMock()

    def test_sell_no_bids(self):
        """sell_shares returns None when no bids on orderbook."""
        with patch.object(self.client, "get_best_bid", return_value=None):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="stop_loss")
            )
        self.assertIsNone(result)

    def test_sell_below_min_price(self):
        """sell_shares skips when best bid < min_price."""
        bid_info = {"price": 0.40, "size": 100, "depth_usd": 40.0, "levels": 3}
        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, min_price=0.45, reason="take_profit")
            )
        self.assertIsNone(result)

    def test_sell_fok_success(self):
        """sell_shares succeeds on FOK fill."""
        bid_info = {"price": 0.65, "size": 100, "depth_usd": 65.0, "levels": 3}

        # Mock fee rate
        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        # Mock FOK success
        self.client._clob.create_order.return_value = "signed_order"
        self.client._clob.post_order.return_value = {
            "success": True, "status": "matched",
            "orderID": "order123", "transactionsHashes": ["0xabc"],
        }
        # Mock fill verification
        self.client._clob.get_order.return_value = {"status": "matched"}

        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="take_profit")
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["exit_price"], 0.65)
        self.assertEqual(result["shares_sold"], 10.0)
        self.assertEqual(result["order_id"], "order123")
        self.assertTrue(result["verified"])

    def test_sell_fok_rejected_gtc_fallback(self):
        """sell_shares falls back to GTC when FOK rejected."""
        bid_info = {"price": 0.65, "size": 5, "depth_usd": 3.25, "levels": 1}

        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        # FOK throws thin-book exception
        call_count = [0]
        def mock_create_order(args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("couldn't be fully filled")
            return "signed_order_gtc"

        self.client._clob.create_order.side_effect = mock_create_order
        self.client._clob.post_order.return_value = {
            "success": True, "status": "matched",
            "orderID": "order456", "transactionsHashes": ["0xdef"],
        }
        self.client._clob.get_order.return_value = {"status": "matched"}

        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="trailing_stop")
            )

        self.assertIsNotNone(result)
        # GTC fallback price = 0.65 - 0.01 = 0.64
        self.assertEqual(result["exit_price"], 0.64)
        self.assertTrue(result["verified"])

    def test_sell_phantom_fill_detected(self):
        """sell_shares returns None on phantom fill (CLOB says matched but verification fails)."""
        bid_info = {"price": 0.65, "size": 100, "depth_usd": 65.0, "levels": 3}

        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        self.client._clob.create_order.return_value = "signed"
        self.client._clob.post_order.return_value = {
            "success": True, "status": "matched",
            "orderID": "order789", "transactionsHashes": ["0xghi"],
        }
        # Verification fails both times
        self.client._clob.get_order.return_value = {"status": "cancelled"}

        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="stop_loss")
            )

        self.assertIsNone(result)

    def test_sell_minimum_shares(self):
        """sell_shares enforces minimum 5 shares."""
        bid_info = {"price": 0.65, "size": 100, "depth_usd": 65.0, "levels": 3}

        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        self.client._clob.create_order.return_value = "signed"
        self.client._clob.post_order.return_value = {
            "success": True, "status": "matched",
            "orderID": "order_min", "transactionsHashes": ["0x"],
        }
        self.client._clob.get_order.return_value = {"status": "matched"}

        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 3.0, reason="take_profit")
            )

        self.assertIsNotNone(result)
        # Should have been bumped to 5.0
        self.assertEqual(result["shares_sold"], 5.0)

    def test_sell_reason_in_logging(self):
        """Different reason tags produce correct log prefixes."""
        # This is a structural test — just verifying no crashes with various reasons
        bid_info = {"price": 0.65, "size": 100, "depth_usd": 65.0, "levels": 3}

        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        self.client._clob.create_order.return_value = "signed"
        self.client._clob.post_order.return_value = {
            "success": True, "status": "matched",
            "orderID": "order_tp", "transactionsHashes": ["0x"],
        }
        self.client._clob.get_order.return_value = {"status": "matched"}

        for reason in ["take_profit", "stop_loss", "trailing_stop", ""]:
            with patch.object(self.client, "get_best_bid", return_value=bid_info):
                result = run_async(
                    self.client.sell_shares("tok123", 10.0, reason=reason)
                )
                self.assertIsNotNone(result)


class TestSellSharesEdgeCases(unittest.TestCase):

    def setUp(self):
        import core.polymarket_client as pm
        pm.HAS_CLOB_SDK = True
        pm.OrderArgs = MagicMock()
        pm.SELL = "SELL"
        pm.BUY = "BUY"
        pm.OrderType = MagicMock()
        pm.OrderType.FOK = "FOK"
        pm.OrderType.GTC = "GTC"

        from core.polymarket_client import PolymarketClient
        self.client = PolymarketClient(MockConfig())
        self.client._clob_initialized = True
        self.client._clob = MagicMock()

    def test_sell_zero_bid_price(self):
        """sell_shares returns None when best bid is zero."""
        bid_info = {"price": 0.0, "size": 100, "depth_usd": 0.0, "levels": 1}
        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="stop_loss")
            )
        self.assertIsNone(result)

    def test_sell_clob_exception(self):
        """sell_shares returns None on unexpected CLOB exception."""
        bid_info = {"price": 0.65, "size": 100, "depth_usd": 65.0, "levels": 3}

        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        self.client._clob.create_order.side_effect = RuntimeError("CLOB down")

        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="take_profit")
            )
        self.assertIsNone(result)

    def test_sell_order_rejected(self):
        """sell_shares returns None when CLOB rejects the order."""
        bid_info = {"price": 0.65, "size": 100, "depth_usd": 65.0, "levels": 3}

        async def mock_fee(*args):
            return 0
        self.client.get_fee_rate_bps = mock_fee

        self.client._clob.create_order.return_value = "signed"
        self.client._clob.post_order.return_value = {
            "success": False, "status": "rejected",
            "errorMsg": "Insufficient balance",
        }

        with patch.object(self.client, "get_best_bid", return_value=bid_info):
            result = run_async(
                self.client.sell_shares("tok123", 10.0, reason="stop_loss")
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
