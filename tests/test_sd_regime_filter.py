"""
Tests for SimpleDirect Module 1: Regime Detection Filter
Adapted from GRIDPHANTOM-TARB test_regime_filter.py

Test plan:
  - Unit tests: Pure functions (_compute_dcr, _compute_nr, _classify_regime, _update_baseline_vol)
  - Candle adapter: Feed synthetic candle series and verify correct classification
  - Override logic: CHOPPY never overridden, UNKNOWN overridden at high confidence
  - Baseline volatility: EMA updates from candle ranges
  - Config validation
  - Snapshot & stats
  - Integration: should_allow_entry() with various candle patterns
"""

import math
import random
import sys
import os
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.regime_filter import (
    RegimeFilter,
    RegimeConfig,
    RegimeState,
    RegimeSnapshot,
    _compute_dcr,
    _compute_nr,
    _classify_regime,
    _update_baseline_vol,
)


# ─── Synthetic Candle Generator ────────────────────────────────────────────────

@dataclass
class FakeCandle:
    """Minimal candle object matching SimpleDirect's Candle interface."""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: float = 0.0


def gen_trending_up_candles(start: float = 100.0, step: float = 0.5, n: int = 30, noise: float = 0.1) -> list[FakeCandle]:
    """Strong uptrend candles: each close > previous close. DCR ~ 1.0, large NR."""
    candles = []
    price = start
    for i in range(n):
        o = price
        c = price + step
        h = max(o, c) + abs(noise)
        l = min(o, c) - abs(noise)
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def gen_trending_down_candles(start: float = 100.0, step: float = 0.5, n: int = 30, noise: float = 0.1) -> list[FakeCandle]:
    """Strong downtrend candles. DCR ~ 1.0, large NR."""
    candles = []
    price = start
    for i in range(n):
        o = price
        c = price - step
        h = max(o, c) + abs(noise)
        l = min(o, c) - abs(noise)
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def gen_choppy_candles(start: float = 100.0, amplitude: float = 0.3, period: int = 6, n: int = 30) -> list[FakeCandle]:
    """Oscillating sine-wave candles: choppy, DCR ~ 0.50, bounded NR."""
    candles = []
    for i in range(n):
        c = start + amplitude * math.sin(2 * math.pi * i / period)
        prev_c = start + amplitude * math.sin(2 * math.pi * (i - 1) / period) if i > 0 else c
        o = prev_c
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
    return candles


def gen_random_walk_candles(start: float = 100.0, step: float = 0.01, n: int = 30, seed: int = 42) -> list[FakeCandle]:
    """Random walk candles: DCR ~ 0.50, tiny NR."""
    rng = random.Random(seed)
    candles = []
    price = start
    for _ in range(n):
        delta = rng.choice([-1, 1]) * step
        o = price
        c = price + delta
        h = max(o, c) + 0.005
        l = min(o, c) - 0.005
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def gen_flat_candles(price: float = 100.0, n: int = 30) -> list[FakeCandle]:
    """Completely flat candles: no directional info."""
    return [FakeCandle(open=price, high=price + 0.01, low=price - 0.01, close=price) for _ in range(n)]


def gen_noisy_trend_candles(start: float = 100.0, trend: float = 0.3, noise: float = 0.05, n: int = 30, seed: int = 42) -> list[FakeCandle]:
    """Mostly trending with small noise. DCR > 0.65."""
    rng = random.Random(seed)
    candles = []
    price = start
    for _ in range(n):
        delta = trend + rng.uniform(-noise, noise)
        o = price
        c = price + delta
        h = max(o, c) + abs(rng.uniform(0, noise))
        l = min(o, c) - abs(rng.uniform(0, noise))
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


def gen_unknown_regime_candles(n: int = 30) -> list[FakeCandle]:
    """Generate candles that land in the UNKNOWN zone: moderate DCR, NR between thresholds."""
    candles = []
    price = 100.0
    for i in range(n):
        if i % 5 == 0:
            delta = -0.3
        else:
            delta = 0.5
        o = price
        c = price + delta
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        candles.append(FakeCandle(open=o, high=h, low=l, close=c))
        price = c
    return candles


# ─── Pure Function Tests (identical logic to TARB) ────────────────────────────

class TestComputeDCR:

    def test_perfect_uptrend(self):
        ticks = [100 + i * 0.5 for i in range(30)]
        assert _compute_dcr(ticks) == 1.0

    def test_perfect_downtrend(self):
        ticks = [100 - i * 0.5 for i in range(30)]
        assert _compute_dcr(ticks) == 1.0

    def test_random_walk_near_half(self):
        rng = random.Random(42)
        ticks = [100.0]
        for _ in range(99):
            ticks.append(ticks[-1] + rng.choice([-1, 1]) * 0.01)
        dcr = _compute_dcr(ticks)
        assert 0.45 <= dcr <= 0.60, f"Random walk DCR should be near 0.50, got {dcr}"

    def test_sine_wave_near_half(self):
        ticks = [100 + 0.5 * math.sin(2 * math.pi * i / 10) for i in range(100)]
        dcr = _compute_dcr(ticks)
        assert 0.45 <= dcr <= 0.60, f"Sine wave DCR should be near 0.50, got {dcr}"

    def test_flat_returns_half(self):
        assert _compute_dcr([100.0] * 30) == 0.5

    def test_single_tick(self):
        assert _compute_dcr([100.0]) == 0.5

    def test_two_ticks_up(self):
        assert _compute_dcr([100.0, 101.0]) == 1.0

    def test_two_ticks_down(self):
        assert _compute_dcr([101.0, 100.0]) == 1.0

    def test_empty_list(self):
        assert _compute_dcr([]) == 0.5


class TestComputeNR:

    def test_large_range_high_nr(self):
        ticks = [100 + i * 1.0 for i in range(30)]
        nr = _compute_nr(ticks, baseline_vol=5.0)
        assert nr > 1.5

    def test_small_range_low_nr(self):
        rng = random.Random(42)
        ticks = [100 + rng.choice([-1, 1]) * 0.01 for _ in range(30)]
        nr = _compute_nr(ticks, baseline_vol=5.0)
        assert nr < 0.8

    def test_zero_baseline_returns_zero(self):
        assert _compute_nr([100, 101, 102], baseline_vol=0.0) == 0.0

    def test_negative_baseline_returns_zero(self):
        assert _compute_nr([100, 101, 102], baseline_vol=-1.0) == 0.0

    def test_single_tick_returns_zero(self):
        assert _compute_nr([100.0], baseline_vol=5.0) == 0.0


class TestClassifyRegime:

    def test_trending(self):
        result = _classify_regime(dcr=0.70, nr=2.0, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.TRENDING

    def test_choppy_low_dcr(self):
        result = _classify_regime(dcr=0.50, nr=2.0, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.CHOPPY

    def test_choppy_low_nr(self):
        result = _classify_regime(dcr=0.70, nr=0.5, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.CHOPPY

    def test_choppy_both_low(self):
        result = _classify_regime(dcr=0.50, nr=0.5, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.CHOPPY

    def test_unknown_middle_ground(self):
        result = _classify_regime(dcr=0.60, nr=1.2, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.UNKNOWN

    def test_unknown_high_dcr_mid_nr(self):
        result = _classify_regime(dcr=0.70, nr=1.0, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.UNKNOWN

    def test_boundary_trending_exact(self):
        result = _classify_regime(dcr=0.65, nr=1.5, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.TRENDING

    def test_boundary_choppy_exact(self):
        result = _classify_regime(dcr=0.55, nr=0.8, dcr_trend=0.65, dcr_chop=0.55, nr_trend=1.5, nr_chop=0.8)
        assert result == RegimeState.CHOPPY


class TestBaselineVolEMA:

    def test_first_update(self):
        result = _update_baseline_vol(new_range=10.0, current_ema=5.0, alpha=0.1)
        expected = 0.1 * 10.0 + 0.9 * 5.0
        assert abs(result - expected) < 1e-10

    def test_high_alpha_reacts_fast(self):
        result = _update_baseline_vol(new_range=20.0, current_ema=10.0, alpha=0.25)
        expected = 0.25 * 20.0 + 0.75 * 10.0
        assert abs(result - expected) < 1e-10

    def test_stable_input_converges(self):
        ema = 5.0
        for _ in range(100):
            ema = _update_baseline_vol(10.0, ema, alpha=0.1)
        assert abs(ema - 10.0) < 0.01


# ─── Integration Tests: RegimeFilter with Candles ─────────────────────────────

class TestRegimeFilterCandles:

    def _make_filter(self, **kwargs) -> RegimeFilter:
        config = RegimeConfig(**kwargs)
        return RegimeFilter(config)

    def test_trending_uptrend_detected(self):
        rf = self._make_filter()
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(start=100, step=1.0, n=30)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.TRENDING

    def test_trending_downtrend_detected(self):
        rf = self._make_filter()
        rf.set_baseline_vol(5.0)
        candles = gen_trending_down_candles(start=100, step=1.0, n=30)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.TRENDING

    def test_choppy_sine_detected(self):
        rf = self._make_filter()
        rf.set_baseline_vol(5.0)
        candles = gen_choppy_candles(start=100, amplitude=0.3, period=6, n=30)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.CHOPPY

    def test_choppy_random_walk_detected(self):
        rf = self._make_filter()
        rf.set_baseline_vol(5.0)
        candles = gen_random_walk_candles(start=100, step=0.01, n=30)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.CHOPPY

    def test_flat_candles_choppy(self):
        rf = self._make_filter()
        rf.set_baseline_vol(5.0)
        candles = gen_flat_candles(price=100.0, n=30)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.CHOPPY

    def test_insufficient_candles_unknown(self):
        rf = self._make_filter(min_candles=10)
        candles = gen_trending_up_candles(n=5)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.UNKNOWN

    def test_noisy_trend_detected(self):
        rf = self._make_filter()
        rf.set_baseline_vol(2.0)
        candles = gen_noisy_trend_candles(start=100, trend=0.3, noise=0.05, n=30)
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.TRENDING

    def test_lookback_limits_candle_window(self):
        """Only the last lookback_candles are used, not the whole history."""
        rf = self._make_filter(lookback_candles=15)
        rf.set_baseline_vol(5.0)
        # 70 choppy candles followed by 15 trending candles
        choppy = gen_choppy_candles(n=70)
        trending = gen_trending_up_candles(start=choppy[-1].close, step=1.0, n=15)
        all_candles = choppy + trending
        regime = rf.classify_from_candles(all_candles)
        assert regime == RegimeState.TRENDING


# ─── Override Logic Tests ──────────────────────────────────────────────────────

class TestOverrideLogic:

    def test_choppy_never_overridden(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_choppy_candles(start=100, amplitude=0.3, period=6, n=30)
        allowed, reason = rf.should_allow_entry(candles, confidence=0.99)
        assert not allowed
        assert "CHOPPY" in reason
        assert "no override" in reason

    def test_trending_always_allowed(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(start=100, step=1.0, n=30)
        allowed, reason = rf.should_allow_entry(candles, confidence=0.60)
        assert allowed
        assert "TRENDING" in reason

    def test_trending_allowed_any_confidence(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(start=100, step=1.0, n=30)
        # Even very low confidence is allowed in trending
        allowed, reason = rf.should_allow_entry(candles, confidence=0.10)
        assert allowed

    def test_unknown_blocked_low_confidence(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_unknown_regime_candles(n=30)
        regime = rf.classify_from_candles(candles)
        if regime == RegimeState.UNKNOWN:
            allowed, reason = rf.should_allow_entry(candles, confidence=0.80)
            assert not allowed
            assert "UNKNOWN" in reason

    def test_unknown_allowed_high_confidence(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_unknown_regime_candles(n=30)
        regime = rf.classify_from_candles(candles)
        if regime == RegimeState.UNKNOWN:
            allowed, reason = rf.should_allow_entry(candles, confidence=0.96)
            assert allowed
            assert "override" in reason.lower()

    def test_unknown_blocked_at_boundary(self):
        """Confidence exactly at threshold should pass (>=)."""
        rf = RegimeFilter(RegimeConfig(regime_override_conviction=0.95))
        rf.set_baseline_vol(5.0)
        candles = gen_unknown_regime_candles(n=30)
        regime = rf.classify_from_candles(candles)
        if regime == RegimeState.UNKNOWN:
            allowed, _ = rf.should_allow_entry(candles, confidence=0.95)
            assert allowed
            # Just below threshold
            allowed2, _ = rf.should_allow_entry(candles, confidence=0.949)
            assert not allowed2

    def test_disabled_filter_always_allows(self):
        rf = RegimeFilter(RegimeConfig(enabled=False))
        candles = gen_choppy_candles(n=30)
        allowed, reason = rf.should_allow_entry(candles, confidence=0.50)
        assert allowed
        assert "disabled" in reason


# ─── Baseline Volatility from Candles ──────────────────────────────────────────

class TestBaselineFromCandles:

    def test_baseline_initializes_after_updates(self):
        rf = RegimeFilter(RegimeConfig())
        for _ in range(5):
            candles = gen_trending_up_candles(n=30, step=0.5)
            rf.update_baseline_from_candles(candles)
        assert rf.baseline_initialized
        assert rf.baseline_vol > 0

    def test_baseline_not_initialized_with_one_update(self):
        rf = RegimeFilter(RegimeConfig())
        candles = gen_trending_up_candles(n=30)
        rf.update_baseline_from_candles(candles)
        # Needs >= 3 windows
        assert not rf.baseline_initialized

    def test_baseline_adapts_higher(self):
        rf = RegimeFilter(RegimeConfig())
        # Low vol first
        for _ in range(5):
            candles = gen_trending_up_candles(n=30, step=0.1, noise=0.02)
            rf.update_baseline_from_candles(candles)
        low_vol = rf.baseline_vol

        # High vol
        for _ in range(20):
            candles = gen_trending_up_candles(n=30, step=5.0, noise=1.0)
            rf.update_baseline_from_candles(candles)
        high_vol = rf.baseline_vol
        assert high_vol > low_vol

    def test_baseline_with_too_few_candles(self):
        rf = RegimeFilter(RegimeConfig())
        candles = [FakeCandle(open=100, high=101, low=99, close=100)]
        rf.update_baseline_from_candles(candles)
        assert not rf.baseline_initialized

    def test_set_baseline_vol_manually(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(3.5)
        assert rf.baseline_initialized
        assert rf.baseline_vol == 3.5


# ─── Config Validation Tests ──────────────────────────────────────────────────

class TestConfigValidation:

    def test_default_config_valid(self):
        config = RegimeConfig()
        config.validate()

    def test_invalid_dcr_thresholds_reversed(self):
        config = RegimeConfig(dcr_trend_threshold=0.50, dcr_chop_threshold=0.55)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_nr_trend_too_low(self):
        config = RegimeConfig(nr_trend_threshold=0.5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_lookback_too_small(self):
        config = RegimeConfig(lookback_candles=5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_lookback_too_large(self):
        config = RegimeConfig(lookback_candles=100)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_alpha_too_low(self):
        config = RegimeConfig(baseline_ema_alpha=0.01)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_override_conviction_too_low(self):
        config = RegimeConfig(regime_override_conviction=0.80)
        with pytest.raises(AssertionError):
            config.validate()

    def test_nr_chop_must_be_less_than_trend(self):
        config = RegimeConfig(nr_chop_threshold=1.5, nr_trend_threshold=1.5)
        with pytest.raises(AssertionError):
            config.validate()


# ─── Snapshot & Stats Tests ────────────────────────────────────────────────────

class TestSnapshotAndStats:

    def test_snapshot_populated_after_classify(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(start=100, step=1.0, n=30)
        rf.classify_from_candles(candles)
        snap = rf.get_snapshot()
        assert snap is not None
        assert snap.state == RegimeState.TRENDING
        assert snap.dcr > 0
        assert snap.nr > 0
        assert snap.candle_count == 30

    def test_snapshot_to_dict(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(n=30)
        rf.classify_from_candles(candles)
        d = rf.get_snapshot().to_dict()
        assert "state" in d
        assert "dcr" in d
        assert "nr" in d
        assert d["state"] == "TRENDING"

    def test_stats_count(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(n=30)
        rf.classify_from_candles(candles)
        rf.classify_from_candles(candles)
        stats = rf.get_stats()
        assert stats["total_classifications"] == 2
        assert stats["trending"] == 2

    def test_stats_mixed(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        rf.classify_from_candles(gen_trending_up_candles(n=30))
        rf.classify_from_candles(gen_choppy_candles(n=30))
        stats = rf.get_stats()
        assert stats["total_classifications"] == 2
        assert stats["trending"] == 1
        assert stats["choppy"] >= 1

    def test_snapshot_none_before_classify(self):
        rf = RegimeFilter(RegimeConfig())
        assert rf.get_snapshot() is None

    def test_choppy_blocks_counted(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        rf.classify_from_candles(gen_choppy_candles(n=30))
        rf.classify_from_candles(gen_choppy_candles(n=30))
        stats = rf.get_stats()
        assert stats["choppy_blocks"] >= 2


# ─── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_candles(self):
        rf = RegimeFilter(RegimeConfig())
        regime = rf.classify_from_candles([])
        assert regime == RegimeState.UNKNOWN

    def test_single_candle(self):
        rf = RegimeFilter(RegimeConfig())
        candles = [FakeCandle(open=100, high=101, low=99, close=100.5)]
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.UNKNOWN

    def test_candles_without_high_low_fallback(self):
        """Candles that lack high/low should still work using close-based NR."""
        @dataclass
        class CloseOnlyCandle:
            close: float

        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = [CloseOnlyCandle(close=100 + i * 1.0) for i in range(30)]
        regime = rf.classify_from_candles(candles)
        assert regime == RegimeState.TRENDING

    def test_regime_state_persists(self):
        rf = RegimeFilter(RegimeConfig())
        rf.set_baseline_vol(5.0)
        candles = gen_trending_up_candles(n=30)
        rf.classify_from_candles(candles)
        assert rf.last_regime == RegimeState.TRENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
