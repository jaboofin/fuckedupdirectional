"""
Tests for core/early_exit.py — Phase 1 pure logic.

Run:  python -m pytest tests/test_early_exit.py -v
  or: python tests/test_early_exit.py
"""

import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

from core.early_exit import (
    EarlyExitManager,
    ExitSignal,
    ExitType,
    TrackedPosition,
    compute_polymarket_fee,
    compute_round_trip_fee,
    _get_conviction_multipliers,
)
from config.settings import EarlyExitConfig


class TestFeeComputation(unittest.TestCase):
    """Polymarket fee formula: price * (1 - price) * 0.0625"""

    def test_fee_at_50_cents(self):
        fee = compute_polymarket_fee(0.50)
        self.assertAlmostEqual(fee, 0.015625, places=4)

    def test_fee_at_extremes(self):
        self.assertAlmostEqual(compute_polymarket_fee(0.0), 0.0)
        self.assertAlmostEqual(compute_polymarket_fee(1.0), 0.0)

    def test_fee_symmetry(self):
        # fee(0.3) == fee(0.7) due to p*(1-p) symmetry
        self.assertAlmostEqual(
            compute_polymarket_fee(0.3), compute_polymarket_fee(0.7), places=6
        )

    def test_fee_at_53_cents(self):
        # 0.53 * 0.47 * 0.0625 = 0.01556...
        fee = compute_polymarket_fee(0.53)
        self.assertAlmostEqual(fee, 0.53 * 0.47 * 0.0625, places=6)

    def test_round_trip(self):
        rt = compute_round_trip_fee(0.50, 0.60)
        buy_fee = compute_polymarket_fee(0.50)
        sell_fee = compute_polymarket_fee(0.60)
        self.assertAlmostEqual(rt, buy_fee + sell_fee, places=6)

    def test_round_trip_spec_example(self):
        # Spec: buy 0.53, sell 0.68 → RT ≈ 0.0291
        rt = compute_round_trip_fee(0.53, 0.68)
        self.assertAlmostEqual(rt, 0.0291, delta=0.001)


class TestConvictionTiers(unittest.TestCase):

    def test_marginal_tier(self):
        tp, trail = _get_conviction_multipliers(0.65)
        self.assertEqual(tp, 0.7)
        self.assertEqual(trail, 0.8)

    def test_standard_tier(self):
        tp, trail = _get_conviction_multipliers(0.75)
        self.assertEqual(tp, 1.0)
        self.assertEqual(trail, 1.0)

    def test_high_tier(self):
        tp, trail = _get_conviction_multipliers(0.85)
        self.assertEqual(tp, 1.2)
        self.assertEqual(trail, 1.1)

    def test_very_high_tier(self):
        tp, trail = _get_conviction_multipliers(0.92)
        self.assertEqual(tp, 1.5)
        self.assertEqual(trail, 1.3)

    def test_extreme_tier(self):
        tp, trail = _get_conviction_multipliers(0.97)
        self.assertEqual(tp, 2.0)
        self.assertEqual(trail, 1.5)

    def test_below_minimum(self):
        tp, trail = _get_conviction_multipliers(0.50)
        self.assertEqual(tp, 0.7)
        self.assertEqual(trail, 0.8)

    def test_boundary_values(self):
        # 0.70 should be Standard, not Marginal
        tp, _ = _get_conviction_multipliers(0.70)
        self.assertEqual(tp, 1.0)
        # 0.69 should be Marginal
        tp, _ = _get_conviction_multipliers(0.69)
        self.assertEqual(tp, 0.7)


class TestConfigValidation(unittest.TestCase):

    def test_defaults(self):
        cfg = EarlyExitConfig()
        self.assertEqual(cfg.tp_offset, 0.10)
        self.assertEqual(cfg.sl_offset, 0.08)
        self.assertEqual(cfg.trail_offset, 0.06)

    def test_clamps_to_range(self):
        cfg = EarlyExitConfig(tp_offset=0.50, sl_offset=0.01, poll_interval_secs=1.0)
        self.assertEqual(cfg.tp_offset, 0.25)  # clamped to max
        self.assertEqual(cfg.sl_offset, 0.04)  # clamped to min
        self.assertEqual(cfg.poll_interval_secs, 3.0)  # clamped to min


class TestRegistration(unittest.TestCase):

    def setUp(self):
        self.cfg = EarlyExitConfig(enabled=True)
        self.mgr = EarlyExitManager(self.cfg)

    def test_register_basic(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.assertEqual(pos.entry_price, 0.53)
        self.assertEqual(pos.conviction, 0.75)
        self.assertFalse(pos.exited)
        self.assertIn("t1", self.mgr.get_tracked())

    def test_tp_price_standard_conviction(self):
        # Standard tier (0.75): tp_mult = 1.0, so TP = 0.53 + 0.10 = 0.63
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.assertIsNotNone(pos.tp_price)
        # TP should be around 0.63, possibly adjusted up for min_net_profit
        self.assertGreaterEqual(pos.tp_price, 0.60)
        self.assertLessEqual(pos.tp_price, 0.70)

    def test_tp_price_marginal_conviction(self):
        # Marginal tier (0.65): tp_mult = 0.7, so TP offset = 0.10 * 0.7 = 0.07
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.65, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.assertIsNotNone(pos.tp_price)
        # TP should be around 0.60 (0.53 + 0.07)
        self.assertGreaterEqual(pos.tp_price, 0.58)
        self.assertLessEqual(pos.tp_price, 0.66)

    def test_tp_nets_positive(self):
        """TP target must net positive after round-trip fees."""
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        rt_fee = compute_round_trip_fee(pos.entry_price, pos.tp_price)
        net = (pos.tp_price - pos.entry_price) - rt_fee
        self.assertGreaterEqual(net, self.cfg.tp_min_net_profit)

    def test_sl_price(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # SL = 0.53 - 0.08 = 0.45
        self.assertAlmostEqual(pos.sl_price, 0.45, places=2)

    def test_sl_min_price_floor(self):
        # If entry is low, SL should not go below sl_min_price (0.10)
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.15,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # SL = 0.15 - 0.08 = 0.07, but clamped to min 0.10
        self.assertEqual(pos.sl_price, 0.10)

    def test_trail_params(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Trail activation = 0.53 + 0.04 = 0.57
        self.assertAlmostEqual(pos.trail_activation_price, 0.57, places=2)
        # Standard tier: trail_mult = 1.0, trail_offset = 0.06
        self.assertAlmostEqual(pos.trail_offset, 0.06, places=2)
        self.assertFalse(pos.trail_active)

    def test_no_duplicate_registration(self):
        pos1 = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        pos2 = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.60,
            conviction=0.80, size_usd=10.0, market_close_ts=time.time() + 900,
        )
        # Should return original, not overwrite
        self.assertEqual(pos2.entry_price, 0.53)

    def test_tp_disabled(self):
        cfg = EarlyExitConfig(enabled=True, tp_enabled=False)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.assertIsNone(pos.tp_price)

    def test_sl_disabled(self):
        cfg = EarlyExitConfig(enabled=True, sl_enabled=False)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.assertIsNone(pos.sl_price)

    def test_conviction_scale_disabled(self):
        cfg = EarlyExitConfig(enabled=True, tp_conviction_scale=False)
        mgr = EarlyExitManager(cfg)
        pos_low = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.65, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        pos_high = mgr.register_position(
            trade_id="t2", token_id="tok2", entry_price=0.53,
            conviction=0.95, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Without scaling, both should get same TP
        self.assertEqual(pos_low.tp_price, pos_high.tp_price)


class TestTakeProfit(unittest.TestCase):

    def setUp(self):
        self.cfg = EarlyExitConfig(
            enabled=True, min_hold_secs=0,  # Disable min hold for tests
            sl_enabled=False, trail_enabled=False,  # Isolate TP
        )
        self.mgr = EarlyExitManager(self.cfg)

    def test_tp_triggers(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Bid at TP target
        sig = self.mgr.check_position("t1", pos.tp_price)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.TAKE_PROFIT)
        self.assertGreater(sig.net_pnl_per_share, 0)

    def test_tp_above_target(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", pos.tp_price + 0.05)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.TAKE_PROFIT)

    def test_tp_not_reached(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", pos.tp_price - 0.01)
        self.assertIsNone(sig)

    def test_tp_net_profit_positive(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", pos.tp_price)
        self.assertGreaterEqual(sig.net_pnl_per_share, self.cfg.tp_min_net_profit)


class TestStopLoss(unittest.TestCase):

    def setUp(self):
        self.cfg = EarlyExitConfig(
            enabled=True, min_hold_secs=0,
            tp_enabled=False, trail_enabled=False,  # Isolate SL
        )
        self.mgr = EarlyExitManager(self.cfg)

    def test_sl_triggers(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", pos.sl_price)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.STOP_LOSS)
        self.assertLess(sig.gross_pnl_per_share, 0)

    def test_sl_below_target(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", pos.sl_price - 0.05)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.STOP_LOSS)

    def test_sl_not_reached(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", pos.sl_price + 0.01)
        self.assertIsNone(sig)

    def test_sl_saves_vs_full_loss(self):
        """Spec Example 3: SL should save ~80% of the loss."""
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", 0.45)
        self.assertIsNotNone(sig)
        # Loss with SL: ~0.08 + fees
        sl_loss = abs(sig.net_pnl_per_share)
        # Full loss: 0.53 + entry fee
        full_loss = 0.53 + compute_polymarket_fee(0.53)
        self.assertLess(sl_loss, full_loss * 0.3)  # SL loss < 30% of full loss

    def test_sl_time_decay(self):
        """SL should tighten in the final 60 seconds."""
        cfg = EarlyExitConfig(
            enabled=True, min_hold_secs=0,
            tp_enabled=False, trail_enabled=False,
            sl_time_decay=True,
        )
        mgr = EarlyExitManager(cfg)
        # Market closes in 30 seconds
        close_ts = time.time() + 30
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=close_ts,
        )
        base_sl = pos.sl_price  # 0.45

        # At 30s remaining, decay_factor = 0.5 + 0.5*(30/60) = 0.75
        # Decayed offset = 0.08 * 0.75 = 0.06
        # Effective SL = 0.53 - 0.06 = 0.47
        # Should trigger at 0.47 (which would NOT trigger base SL of 0.45)
        sig = mgr.check_position("t1", 0.47)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.STOP_LOSS)
        self.assertIn("time-decayed", sig.reason)


class TestTrailingStop(unittest.TestCase):

    def setUp(self):
        self.cfg = EarlyExitConfig(
            enabled=True, min_hold_secs=0,
            tp_enabled=False, sl_enabled=False,  # Isolate trail
            trail_tighten_near_close=False,
        )
        self.mgr = EarlyExitManager(self.cfg)

    def test_trail_not_active_initially(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.assertFalse(pos.trail_active)

    def test_trail_activates(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Price moves above activation (0.57)
        self.mgr.check_position("t1", 0.58)
        self.assertTrue(pos.trail_active)
        self.assertAlmostEqual(pos.trail_peak, 0.58, places=2)

    def test_trail_does_not_activate_below_threshold(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.check_position("t1", 0.56)
        self.assertFalse(pos.trail_active)

    def test_trail_peak_updates(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.check_position("t1", 0.58)  # activate
        self.mgr.check_position("t1", 0.62)  # new peak
        self.assertAlmostEqual(pos.trail_peak, 0.62, places=2)
        self.mgr.check_position("t1", 0.60)  # drops but no new peak
        self.assertAlmostEqual(pos.trail_peak, 0.62, places=2)

    def test_trail_triggers_on_drop(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.check_position("t1", 0.58)  # activate
        self.mgr.check_position("t1", 0.72)  # peak
        # Trail trigger = 0.72 - 0.06 = 0.66
        sig = self.mgr.check_position("t1", 0.66)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.TRAILING_STOP)
        self.assertAlmostEqual(sig.peak_price, 0.72, places=2)

    def test_trail_holds_above_trigger(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.check_position("t1", 0.58)  # activate
        self.mgr.check_position("t1", 0.72)  # peak
        sig = self.mgr.check_position("t1", 0.67)  # above trigger of 0.66
        self.assertIsNone(sig)

    def test_trail_tighten_near_close(self):
        cfg = EarlyExitConfig(
            enabled=True, min_hold_secs=0,
            tp_enabled=False, sl_enabled=False,
            trail_tighten_near_close=True,
        )
        mgr = EarlyExitManager(cfg)
        close_ts = time.time() + 30  # 30s left
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=close_ts,
        )
        mgr.check_position("t1", 0.58)
        mgr.check_position("t1", 0.72)
        # Normal trigger = 0.72 - 0.06 = 0.66
        # Tightened (0.6x) = 0.72 - 0.036 = 0.684
        sig = mgr.check_position("t1", 0.68)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.TRAILING_STOP)


class TestEvaluationPriority(unittest.TestCase):
    """TP is checked before SL before Trail."""

    def setUp(self):
        self.cfg = EarlyExitConfig(enabled=True, min_hold_secs=0)
        self.mgr = EarlyExitManager(self.cfg)

    def test_tp_wins_over_trail(self):
        """If price is at TP and trail would also trigger, TP wins."""
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Activate trail high, then price equals TP
        self.mgr.check_position("t1", 0.90)  # activate trail at high peak
        sig = self.mgr.check_position("t1", pos.tp_price)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.TAKE_PROFIT)


class TestMinHold(unittest.TestCase):

    def test_min_hold_blocks_exit(self):
        cfg = EarlyExitConfig(enabled=True, min_hold_secs=30)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Immediately check — should be blocked by min hold
        sig = mgr.check_position("t1", pos.tp_price + 0.10)
        self.assertIsNone(sig)

    def test_min_hold_allows_after_time(self):
        cfg = EarlyExitConfig(enabled=True, min_hold_secs=1)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Fake the registration time to be 2 seconds ago
        pos.registered_at = time.time() - 2
        sig = mgr.check_position("t1", pos.tp_price + 0.10)
        self.assertIsNotNone(sig)


class TestPositionManagement(unittest.TestCase):

    def setUp(self):
        self.cfg = EarlyExitConfig(enabled=True, min_hold_secs=0)
        self.mgr = EarlyExitManager(self.cfg)

    def test_mark_exited(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.mark_exited("t1", ExitType.TAKE_PROFIT, 0.65)
        self.assertTrue(pos.exited)
        self.assertEqual(pos.exit_type, ExitType.TAKE_PROFIT)
        # Exited position should not appear in get_tracked()
        self.assertNotIn("t1", self.mgr.get_tracked())
        # But should appear in get_all_tracked()
        self.assertIn("t1", self.mgr.get_all_tracked())

    def test_exited_position_not_checked(self):
        pos = self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.mark_exited("t1", ExitType.TAKE_PROFIT, 0.65)
        sig = self.mgr.check_position("t1", 0.90)
        self.assertIsNone(sig)

    def test_remove_position(self):
        self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.remove_position("t1")
        self.assertNotIn("t1", self.mgr.get_tracked())
        self.assertNotIn("t1", self.mgr.get_all_tracked())

    def test_remove_nonexistent(self):
        # Should not raise
        self.mgr.remove_position("nonexistent")

    def test_check_nonexistent(self):
        sig = self.mgr.check_position("nonexistent", 0.50)
        self.assertIsNone(sig)

    def test_zero_bid(self):
        self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        sig = self.mgr.check_position("t1", 0.0)
        self.assertIsNone(sig)

    def test_get_status(self):
        self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        status = self.mgr.get_status()
        self.assertEqual(status["active_positions"], 1)
        self.assertIn("t1", status["positions"])

    def test_exit_stats_tracking(self):
        self.mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        self.mgr.mark_exited("t1", ExitType.TAKE_PROFIT, 0.65)
        status = self.mgr.get_status()
        self.assertEqual(status["exits"]["take_profit"], 1)
        self.assertGreater(status["total_exit_pnl"], 0)


class TestSpecExamples(unittest.TestCase):
    """Validate the worked examples from the spec document."""

    def test_example_1_take_profit(self):
        """Entry 0.53, conviction 0.67 (Marginal), TP should be ~0.60."""
        cfg = EarlyExitConfig(enabled=True, min_hold_secs=0, sl_enabled=False, trail_enabled=False)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.67, size_usd=5.0, market_close_ts=time.time() + 900,
        )
        # Marginal tier: tp_mult = 0.7, offset = 0.10 * 0.7 = 0.07
        # TP = 0.53 + 0.07 = 0.60 (if it passes fee check)
        self.assertIsNotNone(pos.tp_price)
        self.assertGreaterEqual(pos.tp_price, 0.58)
        self.assertLessEqual(pos.tp_price, 0.65)

        # Bid hits 0.61 → should trigger
        sig = mgr.check_position("t1", 0.61)
        if pos.tp_price <= 0.61:
            self.assertIsNotNone(sig)
            self.assertEqual(sig.exit_type, ExitType.TAKE_PROFIT)
            self.assertGreater(sig.net_pnl_per_share, 0)

    def test_example_3_stop_loss(self):
        """Entry 0.53, SL = 0.45. Drop to 0.45 triggers, saves ~80%."""
        cfg = EarlyExitConfig(enabled=True, min_hold_secs=0, tp_enabled=False, trail_enabled=False)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.53,
            conviction=0.75, size_usd=2.60, market_close_ts=time.time() + 900,
        )
        self.assertAlmostEqual(pos.sl_price, 0.45, places=2)

        sig = mgr.check_position("t1", 0.45)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_type, ExitType.STOP_LOSS)
        # Loss should be much less than full loss of ~0.5456/share
        self.assertLess(abs(sig.net_pnl_per_share), 0.20)

    def test_example_4_nothing_triggers(self):
        """Price hovers, nothing triggers, returns None every time."""
        cfg = EarlyExitConfig(enabled=True, min_hold_secs=0)
        mgr = EarlyExitManager(cfg)
        pos = mgr.register_position(
            trade_id="t1", token_id="tok1", entry_price=0.48,
            conviction=0.75, size_usd=5.0, market_close_ts=time.time() + 300,
        )
        for bid in [0.48, 0.50, 0.49, 0.51, 0.50, 0.49]:
            sig = mgr.check_position("t1", bid)
            self.assertIsNone(sig)


class TestSharesEstimate(unittest.TestCase):

    def test_shares_basic(self):
        pos = TrackedPosition(
            trade_id="t1", token_id="tok1", side="BUY",
            entry_price=0.53, conviction=0.75, size_usd=5.0,
            registered_at=time.time(), market_close_ts=time.time() + 900,
        )
        # 5.0 / 0.53 ≈ 9.43
        self.assertAlmostEqual(pos.shares_estimate, 5.0 / 0.53, places=2)

    def test_shares_zero_entry(self):
        pos = TrackedPosition(
            trade_id="t1", token_id="tok1", side="BUY",
            entry_price=0.0, conviction=0.75, size_usd=5.0,
            registered_at=time.time(), market_close_ts=time.time() + 900,
        )
        self.assertEqual(pos.shares_estimate, 0.0)


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
