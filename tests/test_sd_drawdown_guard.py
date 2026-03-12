"""
Test Suite for SimpleDirect Module 3: Cumulative Drawdown Kill Switch
Adapted from GRIDPHANTOM-TARB test_drawdown_guard.py (73 tests)

Covers: tier classification, escalation, recovery/hysteresis, persistence,
        HWM tracking, halted file, acknowledge-drawdown, edge cases,
        config validation, status reporting, and integration scenarios.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.drawdown_guard import (
    DrawdownConfig,
    DrawdownGuard,
    DrawdownState,
    DrawdownTier,
)


class TestBase(unittest.TestCase):
    """Base class with temp directory setup for persistence tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.hwm_path = os.path.join(self.tmp_dir, "hwm_state.json")
        self.halted_path = os.path.join(self.tmp_dir, "HALTED")
        self.mock_time = 1000000.0

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _time_fn(self):
        return self.mock_time

    def _advance_time(self, seconds=1.0):
        self.mock_time += seconds

    def _make_config(self, **overrides):
        defaults = {
            "hwm_state_path": self.hwm_path,
            "halted_file_path": self.halted_path,
        }
        defaults.update(overrides)
        return DrawdownConfig(**defaults)

    def _make_guard(self, bankroll=1000.0, acknowledge=False, **config_overrides):
        config = self._make_config(**config_overrides)
        return DrawdownGuard(
            config=config,
            initial_bankroll=bankroll,
            acknowledge_drawdown=acknowledge,
            time_fn=self._time_fn,
        )


# ============================================================================
# 1. BASIC TIER CLASSIFICATION (5 tests)
# ============================================================================

class TestTierClassification(TestBase):

    def test_green_no_drawdown(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(1000.0)
        self.assertEqual(tier, DrawdownTier.GREEN)

    def test_green_small_drawdown(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(900.0)
        self.assertEqual(tier, DrawdownTier.GREEN)

    def test_yellow_at_threshold(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(850.0)
        self.assertEqual(tier, DrawdownTier.YELLOW)

    def test_orange_at_threshold(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(750.0)
        self.assertEqual(tier, DrawdownTier.ORANGE)

    def test_red_at_threshold(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(600.0)
        self.assertEqual(tier, DrawdownTier.RED)


# ============================================================================
# 2. TIER ESCALATION (6 tests)
# ============================================================================

class TestTierEscalation(TestBase):

    def test_green_to_yellow(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)

    def test_green_to_orange_skip_yellow(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)

    def test_green_to_red_skip_all(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)
        self.assertTrue(guard.is_halted())

    def test_yellow_to_orange(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(740.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)

    def test_yellow_to_red(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        guard.update_bankroll(550.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)
        self.assertTrue(guard.is_halted())

    def test_orange_to_red(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        guard.update_bankroll(590.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)
        self.assertTrue(guard.is_halted())


# ============================================================================
# 3. RECOVERY & HYSTERESIS (7 tests)
# ============================================================================

class TestRecoveryHysteresis(TestBase):

    def test_yellow_recovery_to_green(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(910.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_yellow_no_recovery_in_hysteresis_band(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        guard.update_bankroll(880.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)

    def test_orange_recovery_to_green(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)
        guard.update_bankroll(860.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_orange_no_recovery_in_hysteresis_band(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        guard.update_bankroll(800.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)

    def test_red_never_auto_recovers(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)
        guard.update_bankroll(990.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)
        self.assertTrue(guard.is_halted())

    def test_recovery_after_new_hwm_resets_baseline(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        guard.update_bankroll(910.0)
        guard.update_bankroll(1100.0)
        self.assertEqual(guard.state.high_water_mark, 1100.0)
        guard.update_bankroll(930.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)

    def test_hysteresis_prevents_oscillation(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(845.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(855.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(845.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(905.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)


# ============================================================================
# 4. HIGH-WATER MARK TRACKING (5 tests)
# ============================================================================

class TestHWMTracking(TestBase):

    def test_hwm_set_on_init(self):
        guard = self._make_guard(1000.0)
        self.assertEqual(guard.state.high_water_mark, 1000.0)

    def test_hwm_updates_on_new_high(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(1050.0)
        self.assertEqual(guard.state.high_water_mark, 1050.0)

    def test_hwm_unchanged_on_loss(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(950.0)
        self.assertEqual(guard.state.high_water_mark, 1000.0)

    def test_hwm_tracks_through_multiple_peaks(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(1100.0)
        guard.update_bankroll(1050.0)
        guard.update_bankroll(1200.0)
        guard.update_bankroll(1150.0)
        self.assertEqual(guard.state.high_water_mark, 1200.0)

    def test_hwm_incremental_profits(self):
        guard = self._make_guard(1000.0)
        for i in range(10):
            guard.update_bankroll(1000.0 + (i + 1) * 5.0)
        self.assertEqual(guard.state.high_water_mark, 1050.0)


# ============================================================================
# 5. SIZE MULTIPLIER & CONFIDENCE (7 tests)
# ============================================================================

class TestSizeAndConfidence(TestBase):

    def test_green_full_size(self):
        guard = self._make_guard(1000.0)
        self.assertEqual(guard.get_size_multiplier(), 1.0)

    def test_yellow_half_size(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self.assertEqual(guard.get_size_multiplier(), 0.5)

    def test_orange_quarter_size(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        self.assertEqual(guard.get_size_multiplier(), 0.25)

    def test_red_zero_size(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        self.assertEqual(guard.get_size_multiplier(), 0.0)

    def test_orange_confidence_floor(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        self.assertEqual(guard.get_min_confidence(), 0.90)

    def test_yellow_no_confidence_floor(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self.assertEqual(guard.get_min_confidence(), 0.0)

    def test_green_no_confidence_floor(self):
        guard = self._make_guard(1000.0)
        self.assertEqual(guard.get_min_confidence(), 0.0)


# ============================================================================
# 6. PERSISTENCE (7 tests)
# ============================================================================

class TestPersistence(TestBase):

    def test_state_saved_on_init(self):
        self._make_guard(1000.0)
        self.assertTrue(os.path.exists(self.hwm_path))

    def test_state_saved_on_update(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(900.0)
        with open(self.hwm_path) as f:
            data = json.load(f)
        self.assertEqual(data["current_bankroll"], 900.0)

    def test_state_survives_restart(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self._advance_time(10)
        guard2 = self._make_guard(0.0)
        self.assertEqual(guard2.state.high_water_mark, 1000.0)
        self.assertEqual(guard2.state.current_bankroll, 840.0)
        self.assertEqual(guard2.state.tier, DrawdownTier.YELLOW)

    def test_hwm_persists_across_restart(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(1200.0)
        guard.update_bankroll(1100.0)
        guard2 = self._make_guard(0.0)
        self.assertEqual(guard2.state.high_water_mark, 1200.0)

    def test_red_tier_persists(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        guard2 = self._make_guard(0.0)
        self.assertTrue(guard2.is_halted())
        self.assertEqual(guard2.state.tier, DrawdownTier.RED)

    def test_corrupt_state_file_starts_fresh(self):
        with open(self.hwm_path, "w") as f:
            f.write("NOT JSON")
        guard = self._make_guard(500.0)
        self.assertEqual(guard.state.high_water_mark, 500.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_missing_state_dir_created(self):
        nested_path = os.path.join(self.tmp_dir, "nested", "deep", "hwm.json")
        guard = self._make_guard(1000.0, hwm_state_path=nested_path)
        self.assertTrue(os.path.exists(nested_path))


# ============================================================================
# 7. HALTED FILE (5 tests)
# ============================================================================

class TestHaltedFile(TestBase):

    def test_halted_file_created_on_red(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        self.assertTrue(os.path.exists(self.halted_path))

    def test_halted_file_contains_info(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        with open(self.halted_path) as f:
            content = f.read()
        self.assertIn("TRADING HALTED", content)
        self.assertIn("$1000.00", content)
        self.assertIn("$500.00", content)

    def test_halted_file_not_created_for_yellow(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        self.assertFalse(os.path.exists(self.halted_path))

    def test_halted_file_not_created_for_orange(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(740.0)
        self.assertFalse(os.path.exists(self.halted_path))

    def test_halted_file_removed_on_acknowledge(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        self.assertTrue(os.path.exists(self.halted_path))
        guard2 = self._make_guard(0.0, acknowledge=True)
        self.assertFalse(os.path.exists(self.halted_path))
        self.assertFalse(guard2.is_halted())


# ============================================================================
# 8. ACKNOWLEDGE DRAWDOWN (5 tests)
# ============================================================================

class TestAcknowledgeDrawdown(TestBase):

    def test_acknowledge_resets_to_orange(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        guard2 = self._make_guard(0.0, acknowledge=True)
        self.assertEqual(guard2.state.tier, DrawdownTier.ORANGE)
        self.assertFalse(guard2.is_halted())

    def test_no_acknowledge_stays_halted(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        guard2 = self._make_guard(0.0, acknowledge=False)
        self.assertTrue(guard2.is_halted())
        self.assertEqual(guard2.state.tier, DrawdownTier.RED)

    def test_acknowledge_when_not_halted_no_effect(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        guard2 = self._make_guard(0.0, acknowledge=True)
        self.assertEqual(guard2.state.tier, DrawdownTier.YELLOW)

    def test_acknowledge_preserves_hwm(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(1500.0)
        guard.update_bankroll(800.0)
        guard2 = self._make_guard(0.0, acknowledge=True)
        self.assertEqual(guard2.state.high_water_mark, 1500.0)

    def test_acknowledge_allows_trading_then_can_re_halt(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        guard2 = self._make_guard(0.0, acknowledge=True)
        self.assertFalse(guard2.is_halted())
        guard2.update_bankroll(400.0)
        self.assertTrue(guard2.is_halted())
        self.assertEqual(guard2.state.tier, DrawdownTier.RED)


# ============================================================================
# 9. STATE OBJECT (4 tests)
# ============================================================================

class TestDrawdownState(TestBase):

    def test_drawdown_pct_normal(self):
        state = DrawdownState(high_water_mark=1000.0, current_bankroll=850.0)
        self.assertAlmostEqual(state.drawdown_pct(), 0.15)

    def test_drawdown_pct_zero_hwm(self):
        state = DrawdownState(high_water_mark=0.0, current_bankroll=0.0)
        self.assertEqual(state.drawdown_pct(), 0.0)

    def test_to_dict_roundtrip(self):
        state = DrawdownState(
            high_water_mark=1234.56,
            current_bankroll=987.65,
            tier=DrawdownTier.ORANGE,
            last_updated=12345.0,
            tier_entry_time=12300.0,
            halted=False,
        )
        d = state.to_dict()
        state2 = DrawdownState.from_dict(d)
        self.assertEqual(state.high_water_mark, state2.high_water_mark)
        self.assertEqual(state.current_bankroll, state2.current_bankroll)
        self.assertEqual(state.tier, state2.tier)
        self.assertEqual(state.halted, state2.halted)

    def test_from_dict_defaults(self):
        state = DrawdownState.from_dict({})
        self.assertEqual(state.high_water_mark, 0.0)
        self.assertEqual(state.tier, DrawdownTier.GREEN)


# ============================================================================
# 10. CONFIG VALIDATION (6 tests)
# ============================================================================

class TestConfigValidation(TestBase):

    def test_valid_defaults(self):
        config = DrawdownConfig()
        config.validate()

    def test_invalid_threshold_order(self):
        config = DrawdownConfig(yellow_threshold=0.30, orange_threshold=0.25)
        with self.assertRaises(ValueError):
            config.validate()

    def test_yellow_gt_one(self):
        config = DrawdownConfig(yellow_threshold=1.5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_size_mult(self):
        config = DrawdownConfig(yellow_size_mult=0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_orange_mult_gt_yellow(self):
        config = DrawdownConfig(yellow_size_mult=0.3, orange_size_mult=0.5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_recovery_must_be_less_than_threshold(self):
        config = DrawdownConfig(yellow_recovery=0.20, yellow_threshold=0.15)
        with self.assertRaises(ValueError):
            config.validate()


# ============================================================================
# 11. STATUS REPORTING (3 tests)
# ============================================================================

class TestStatusReporting(TestBase):

    def test_status_green(self):
        guard = self._make_guard(1000.0)
        status = guard.get_status()
        self.assertEqual(status["tier"], "GREEN")
        self.assertFalse(status["halted"])
        self.assertEqual(status["size_multiplier"], 1.0)
        self.assertEqual(status["drawdown_pct"], 0.0)

    def test_status_yellow(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(840.0)
        status = guard.get_status()
        self.assertEqual(status["tier"], "YELLOW")
        self.assertEqual(status["size_multiplier"], 0.5)
        self.assertEqual(status["drawdown_pct"], 16.0)

    def test_status_red(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(500.0)
        status = guard.get_status()
        self.assertEqual(status["tier"], "RED")
        self.assertTrue(status["halted"])
        self.assertEqual(status["size_multiplier"], 0.0)
        self.assertEqual(status["high_water_mark"], 1000.0)


# ============================================================================
# 12. EDGE CASES (5 tests)
# ============================================================================

class TestEdgeCases(TestBase):

    def test_zero_initial_bankroll(self):
        guard = self._make_guard(0.0)
        self.assertEqual(guard.state.drawdown_pct(), 0.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_bankroll_goes_to_zero(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(0.0)
        self.assertEqual(tier, DrawdownTier.RED)
        self.assertTrue(guard.is_halted())

    def test_negative_bankroll(self):
        guard = self._make_guard(1000.0)
        tier = guard.update_bankroll(-50.0)
        self.assertEqual(tier, DrawdownTier.RED)

    def test_very_small_drawdowns(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(999.99)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_exact_boundary_values(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(850.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(750.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)
        guard.update_bankroll(600.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)


# ============================================================================
# 13. INTEGRATION SCENARIOS (5 tests)
# ============================================================================

class TestIntegrationScenarios(TestBase):

    def test_slow_death_scenario(self):
        """The exact scenario M3 prevents: losing ~19% per day across 3 days."""
        guard = self._make_guard(1000.0)
        guard.update_bankroll(810.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        self.assertEqual(guard.get_size_multiplier(), 0.5)
        guard.update_bankroll(650.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)
        self.assertEqual(guard.get_size_multiplier(), 0.25)
        guard.update_bankroll(580.0)
        self.assertEqual(guard.state.tier, DrawdownTier.RED)
        self.assertTrue(guard.is_halted())

    def test_recovery_after_drawdown(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(820.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(860.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(910.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_growth_then_drawdown(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(1200.0)
        guard.update_bankroll(1500.0)
        self.assertEqual(guard.state.high_water_mark, 1500.0)
        guard.update_bankroll(1275.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)
        guard.update_bankroll(1125.0)
        self.assertEqual(guard.state.tier, DrawdownTier.ORANGE)

    def test_m4_integration_sizing(self):
        """Simulate how conviction sizing interacts with drawdown multipliers."""
        guard = self._make_guard(1000.0)
        base_size_pct = 0.45

        effective = 1000.0 * base_size_pct * guard.get_size_multiplier()
        self.assertEqual(effective, 450.0)

        guard.update_bankroll(840.0)
        effective = 840.0 * base_size_pct * guard.get_size_multiplier()
        self.assertAlmostEqual(effective, 189.0)

        guard.update_bankroll(740.0)
        effective = 740.0 * base_size_pct * guard.get_size_multiplier()
        self.assertAlmostEqual(effective, 83.25)

    def test_restart_resilience_full_cycle(self):
        guard = self._make_guard(1000.0)
        guard.update_bankroll(1300.0)
        guard.update_bankroll(700.0)
        self.assertTrue(guard.is_halted())
        guard2 = self._make_guard(0.0, acknowledge=False)
        self.assertTrue(guard2.is_halted())
        guard3 = self._make_guard(0.0, acknowledge=True)
        self.assertFalse(guard3.is_halted())
        self.assertEqual(guard3.state.tier, DrawdownTier.ORANGE)
        self.assertEqual(guard3.state.high_water_mark, 1300.0)
        guard3.update_bankroll(1200.0)
        self.assertEqual(guard3.state.tier, DrawdownTier.GREEN)


# ============================================================================
# 14. CUSTOM THRESHOLDS (3 tests)
# ============================================================================

class TestCustomThresholds(TestBase):

    def test_tighter_thresholds(self):
        guard = self._make_guard(
            1000.0,
            yellow_threshold=0.05, orange_threshold=0.10, red_threshold=0.20,
            yellow_recovery=0.03, orange_recovery=0.05,
        )
        guard.update_bankroll(940.0)
        self.assertEqual(guard.state.tier, DrawdownTier.YELLOW)

    def test_looser_thresholds(self):
        guard = self._make_guard(
            1000.0,
            yellow_threshold=0.25, orange_threshold=0.40, red_threshold=0.60,
            yellow_recovery=0.15, orange_recovery=0.25,
        )
        guard.update_bankroll(840.0)
        self.assertEqual(guard.state.tier, DrawdownTier.GREEN)

    def test_custom_size_multipliers(self):
        guard = self._make_guard(1000.0, yellow_size_mult=0.75, orange_size_mult=0.10)
        guard.update_bankroll(840.0)
        self.assertEqual(guard.get_size_multiplier(), 0.75)
        guard.update_bankroll(740.0)
        self.assertEqual(guard.get_size_multiplier(), 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
