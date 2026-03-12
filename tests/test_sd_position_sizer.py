"""
Test Suite for SimpleDirect Module 4: Confidence-Scaled Position Sizing
Adapted from GRIDPHANTOM-TARB test_position_sizer.py

Base pct defaults aligned with TARB live config:
  per_market_budget_pct = 0.15 (full bankroll)
  small_bankroll_budget_pct = 0.08 (< $200)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.position_sizer import PositionSizer, SizerConfig


class TestBase(unittest.TestCase):
    def _make_sizer(self, **overrides):
        config = SizerConfig(**overrides)
        return PositionSizer(config)


# 1. CONFIDENCE TIER MULTIPLIERS (7 tests)
class TestConfidenceTiers(TestBase):
    def test_marginal_confidence(self):
        sizer = self._make_sizer()
        size = sizer.compute_size(0.65, 1000.0)
        self.assertEqual(size, 50.0)  # 1000*0.15*0.5=75 -> cap 50

    def test_standard_confidence(self):
        sizer = self._make_sizer()
        self.assertEqual(sizer.compute_size(0.75, 1000.0), 50.0)

    def test_high_confidence(self):
        sizer = self._make_sizer()
        self.assertEqual(sizer.compute_size(0.85, 1000.0), 50.0)

    def test_very_high_confidence(self):
        sizer = self._make_sizer()
        self.assertEqual(sizer.compute_size(0.92, 1000.0), 50.0)

    def test_extreme_confidence(self):
        sizer = self._make_sizer()
        self.assertEqual(sizer.compute_size(0.97, 1000.0), 50.0)

    def test_below_minimum_confidence(self):
        sizer = self._make_sizer()
        self.assertEqual(sizer.compute_size(0.55, 1000.0), 0.0)

    def test_exact_tier_boundaries(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.60, 1000.0), 1000*0.15*0.5, places=2)
        self.assertAlmostEqual(sizer.compute_size(0.70, 1000.0), 1000*0.15*0.75, places=2)
        self.assertAlmostEqual(sizer.compute_size(0.80, 1000.0), 1000*0.15*1.0, places=2)
        self.assertAlmostEqual(sizer.compute_size(0.90, 1000.0), 1000*0.15*1.3, places=2)
        self.assertAlmostEqual(sizer.compute_size(0.95, 1000.0), 1000*0.15*1.6, places=2)


# 2. BANKROLL-TIERED CAPS (8 tests)
class TestBankrollCaps(TestBase):
    def test_small_bankroll_caps_at_1x(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        size = sizer.compute_size(0.97, 100.0)
        self.assertAlmostEqual(size, 100*0.08*1.0, places=2)

    def test_small_bankroll_allows_downscaling(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.65, 100.0), 100*0.08*0.5, places=2)

    def test_small_bankroll_high_confidence_no_boost(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        size = sizer.compute_size(0.92, 150.0)
        self.assertAlmostEqual(size, 150*0.08*1.0, places=2)

    def test_mid_bankroll_caps_at_1_3x(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        size = sizer.compute_size(0.97, 350.0)
        base_pct = 0.08 + (150/300)*0.07  # 0.115
        self.assertAlmostEqual(size, round(350*base_pct*1.3, 2), places=2)

    def test_mid_bankroll_allows_1_3x(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        size = sizer.compute_size(0.92, 350.0)
        base_pct = 0.08 + (150/300)*0.07
        self.assertAlmostEqual(size, round(350*base_pct*1.3, 2), places=2)

    def test_full_bankroll_no_cap(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.97, 600.0), 600*0.15*1.6, places=2)

    def test_at_small_boundary_exact(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.97, 200.0), 200*0.08*1.3, places=2)

    def test_at_mid_boundary_exact(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.97, 500.0), 500*0.15*1.6, places=2)


# 3. BASE PERCENTAGE INTERPOLATION (5 tests)
class TestBasePctInterpolation(TestBase):
    def test_small_bankroll_uses_small_pct(self):
        info = self._make_sizer().get_sizing_info(0.85, 50.0)
        self.assertEqual(info["base_pct"], 0.08)

    def test_full_bankroll_uses_full_pct(self):
        info = self._make_sizer().get_sizing_info(0.85, 600.0)
        self.assertEqual(info["base_pct"], 0.15)

    def test_mid_bankroll_interpolates(self):
        info = self._make_sizer().get_sizing_info(0.85, 350.0)
        self.assertAlmostEqual(info["base_pct"], 0.115, places=4)

    def test_interpolation_at_quarter(self):
        info = self._make_sizer().get_sizing_info(0.85, 275.0)
        self.assertAlmostEqual(info["base_pct"], 0.0975, places=4)

    def test_interpolation_at_three_quarter(self):
        info = self._make_sizer().get_sizing_info(0.85, 425.0)
        self.assertAlmostEqual(info["base_pct"], 0.1325, places=4)


# 4. DRAWDOWN INTEGRATION (6 tests)
class TestDrawdownIntegration(TestBase):
    def test_green_tier_no_change(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.85, 1000.0, 1.0), 150.0, places=2)

    def test_yellow_tier_halves(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.85, 1000.0, 0.5), 75.0, places=2)

    def test_orange_tier_quarters(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.85, 1000.0, 0.25), 37.5, places=2)

    def test_red_tier_zero(self):
        self.assertEqual(self._make_sizer().compute_size(0.85, 1000.0, 0.0), 0.0)

    def test_double_compression_small_bankroll(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.65, 100.0, 0.5), 100*0.08*0.5*0.5, places=2)

    def test_triple_compression_worst_case(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(0.65, 100.0, 0.25), 100*0.08*0.5*0.25, places=2)


# 5. HARD LIMITS (5 tests)
class TestHardLimits(TestBase):
    def test_max_order_cap(self):
        self.assertEqual(self._make_sizer(max_order_usd=50.0).compute_size(0.97, 1000.0), 50.0)

    def test_min_order_floor(self):
        sizer = self._make_sizer(min_order_usd=5.0, max_order_usd=10000.0)
        self.assertEqual(sizer.compute_size(0.65, 10.0, 0.25), 0.0)

    def test_custom_max_order(self):
        self.assertEqual(self._make_sizer(max_order_usd=25.0).compute_size(0.85, 500.0), 25.0)

    def test_size_just_above_min(self):
        sizer = self._make_sizer(min_order_usd=1.0, max_order_usd=10000.0)
        self.assertEqual(sizer.compute_size(0.65, 25.0), 1.0)  # 25*0.08*0.5=1.0

    def test_size_just_below_min(self):
        sizer = self._make_sizer(min_order_usd=1.0, max_order_usd=10000.0)
        self.assertEqual(sizer.compute_size(0.65, 24.0), 0.0)  # 24*0.08*0.5=0.96


# 6. EDGE CASES (6 tests)
class TestEdgeCases(TestBase):
    def test_zero_bankroll(self):
        self.assertEqual(self._make_sizer().compute_size(0.85, 0.0), 0.0)

    def test_negative_bankroll(self):
        self.assertEqual(self._make_sizer().compute_size(0.85, -50.0), 0.0)

    def test_zero_confidence(self):
        self.assertEqual(self._make_sizer().compute_size(0.0, 1000.0), 0.0)

    def test_negative_confidence(self):
        self.assertEqual(self._make_sizer().compute_size(-0.5, 1000.0), 0.0)

    def test_confidence_exactly_1(self):
        sizer = self._make_sizer(max_order_usd=10000.0)
        self.assertAlmostEqual(sizer.compute_size(1.0, 1000.0), 1000*0.15*1.6, places=2)

    def test_very_large_bankroll(self):
        self.assertEqual(self._make_sizer(max_order_usd=50.0).compute_size(0.85, 1_000_000.0), 50.0)


# 7. CONFIG VALIDATION (5 tests)
class TestConfigValidation(TestBase):
    def test_valid_defaults(self):
        SizerConfig().validate()

    def test_small_pct_gt_full_pct(self):
        with self.assertRaises(ValueError):
            SizerConfig(small_bankroll_budget_pct=0.50, per_market_budget_pct=0.25).validate()

    def test_bankroll_caps_inverted(self):
        with self.assertRaises(ValueError):
            SizerConfig(small_bankroll_cap=500.0, mid_bankroll_cap=200.0).validate()

    def test_min_order_gt_max_order(self):
        with self.assertRaises(ValueError):
            SizerConfig(min_order_usd=100.0, max_order_usd=50.0).validate()

    def test_negative_min_order(self):
        with self.assertRaises(ValueError):
            SizerConfig(min_order_usd=-1.0).validate()


# 8. SIZING INFO (4 tests)
class TestSizingInfo(TestBase):
    def test_info_contains_all_fields(self):
        info = self._make_sizer().get_sizing_info(0.85, 500.0)
        expected = {"confidence","confidence_tier","confidence_mult","capped_mult","was_capped","bankroll","bankroll_tier","base_pct","drawdown_mult","final_size"}
        self.assertEqual(set(info.keys()), expected)

    def test_info_small_bankroll_shows_capped(self):
        info = self._make_sizer().get_sizing_info(0.97, 100.0)
        self.assertTrue(info["was_capped"])
        self.assertEqual(info["confidence_mult"], 1.6)
        self.assertEqual(info["capped_mult"], 1.0)
        self.assertEqual(info["bankroll_tier"], "Small")

    def test_info_full_bankroll_not_capped(self):
        info = self._make_sizer().get_sizing_info(0.85, 600.0)
        self.assertFalse(info["was_capped"])
        self.assertEqual(info["bankroll_tier"], "Full")

    def test_info_tier_labels(self):
        sizer = self._make_sizer()
        self.assertEqual(sizer.get_sizing_info(0.65, 500.0)["confidence_tier"], "Marginal")
        self.assertEqual(sizer.get_sizing_info(0.75, 500.0)["confidence_tier"], "Standard")
        self.assertEqual(sizer.get_sizing_info(0.85, 500.0)["confidence_tier"], "High")
        self.assertEqual(sizer.get_sizing_info(0.92, 500.0)["confidence_tier"], "Very High")
        self.assertEqual(sizer.get_sizing_info(0.97, 500.0)["confidence_tier"], "Extreme")


# 9. REALISTIC SCENARIOS (7 tests)
class TestRealisticScenarios(TestBase):
    def test_60_bankroll_green_high(self):
        self.assertEqual(self._make_sizer().compute_size(0.85, 60.0, 1.0), 4.80)

    def test_60_bankroll_green_marginal(self):
        self.assertEqual(self._make_sizer().compute_size(0.65, 60.0, 1.0), 2.40)

    def test_60_bankroll_yellow_high(self):
        self.assertEqual(self._make_sizer().compute_size(0.85, 51.0, 0.5), 2.04)

    def test_100_bankroll_green_high(self):
        self.assertEqual(self._make_sizer().compute_size(0.85, 100.0, 1.0), 8.0)

    def test_100_bankroll_green_extreme_capped(self):
        self.assertEqual(self._make_sizer().compute_size(0.97, 100.0, 1.0), 8.0)

    def test_100_bankroll_orange_marginal(self):
        # 75*0.08*0.5*0.25=0.75 < 1.0 min -> 0
        self.assertEqual(self._make_sizer().compute_size(0.65, 75.0, 0.25), 0.0)

    def test_500_bankroll_unlocks_sizing_up(self):
        # 500*0.15*1.6=120 -> cap 50
        self.assertEqual(self._make_sizer().compute_size(0.97, 500.0, 1.0), 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
