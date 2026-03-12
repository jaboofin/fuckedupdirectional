"""
SimpleDirect Module 1: Regime Detection Filter
Adapted from GRIDPHANTOM-TARB core/regime_filter.py

Real-time regime classifier that categorizes the current market window into
TRENDING, CHOPPY, or UNKNOWN. The bot only enters positions during
TRENDING regimes and sits out during CHOPPY periods.

Uses two metrics:
  - Directional Consistency Ratio (DCR): % of consecutive candle closes moving same direction
  - Normalized Range (NR): price range / baseline volatility

TARB Adaptation Notes:
  - TARB feeds individual RTDS ticks continuously; SimpleDirect uses candle OHLC data
  - This version extracts synthetic tick-equivalent metrics from candle close prices
  - Baseline volatility is computed from candle high-low ranges instead of tick buffer ranges
  - Classification logic (_compute_dcr, _compute_nr, _classify_regime) is IDENTICAL to TARB
  - Integration: Inserts into bot.py _trading_cycle() BEFORE strategy.analyze()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config.settings import RegimeConfig

logger = logging.getLogger("regime_filter")


class RegimeState(Enum):
    TRENDING = "TRENDING"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeSnapshot:
    """Diagnostic snapshot of a regime classification."""
    state: RegimeState
    dcr: float
    nr: float
    baseline_vol: float
    candle_count: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "dcr": round(self.dcr, 4),
            "nr": round(self.nr, 4),
            "baseline_vol": round(self.baseline_vol, 6),
            "candle_count": self.candle_count,
            "timestamp": self.timestamp,
        }


class RegimeFilter:
    """
    Candle-based regime detection filter for SimpleDirect.

    Unlike TARB's version which ingests raw RTDS ticks, this works with
    candle OHLC data that SimpleDirect already pulls each cycle.

    Usage:
        rf = RegimeFilter(config)
        regime = rf.classify_from_candles(candles)
        allowed, reason = rf.should_allow_entry(candles, confidence)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.config.validate()
        self.baseline_vol: float = 0.0
        self.baseline_initialized: bool = False
        self._window_ranges: deque = deque(maxlen=self.config.baseline_window)
        self.last_regime: RegimeState = RegimeState.UNKNOWN
        self.last_snapshot: Optional[RegimeSnapshot] = None
        self._classify_count: int = 0
        self._prev_logged_state: Optional[RegimeState] = None
        self._stats = {
            "total_classifications": 0,
            "trending": 0,
            "choppy": 0,
            "unknown": 0,
            "choppy_blocks": 0,
        }

    def update_baseline_from_candles(self, candles) -> None:
        """
        Update baseline volatility from candle high-low ranges.

        Called at the start of each cycle to keep baseline current.
        Uses the average high-low range of recent candles as the window range,
        then feeds it into the EMA baseline — mirrors TARB's on_window_close().

        Args:
            candles: List of candle objects with .high and .low attributes
        """
        if len(candles) < 2:
            return

        # Use the last N candles' high-low ranges to compute an average window range
        lookback = min(self.config.lookback_candles, len(candles))
        recent = candles[-lookback:]
        ranges = [c.high - c.low for c in recent if hasattr(c, 'high') and hasattr(c, 'low')]

        if not ranges:
            return

        avg_range = sum(ranges) / len(ranges)
        self._window_ranges.append(avg_range)

        if not self.baseline_initialized and len(self._window_ranges) >= 3:
            self.baseline_vol = sum(self._window_ranges) / len(self._window_ranges)
            self.baseline_initialized = True
            logger.info(
                f"[REGIME] Baseline INITIALIZED: vol={self.baseline_vol:.6f} "
                f"from {len(self._window_ranges)} windows"
            )
        elif self.baseline_initialized:
            self.baseline_vol = _update_baseline_vol(
                avg_range, self.baseline_vol, self.config.baseline_ema_alpha
            )

    def classify_from_candles(self, candles) -> RegimeState:
        """
        Classify the current regime from candle data.

        Extracts close prices from the last N candles and computes DCR + NR,
        then applies the same classification logic as TARB.

        Args:
            candles: List of candle objects with .close, .high, .low attributes

        Returns:
            RegimeState.TRENDING, CHOPPY, or UNKNOWN
        """
        lookback = min(self.config.lookback_candles, len(candles))
        recent = candles[-lookback:]

        # Extract close prices as synthetic "ticks"
        closes = [c.close for c in recent if hasattr(c, 'close')]

        if len(closes) < self.config.min_candles:
            self.last_regime = RegimeState.UNKNOWN
            self.last_snapshot = RegimeSnapshot(
                state=RegimeState.UNKNOWN,
                dcr=0.0,
                nr=0.0,
                baseline_vol=self.baseline_vol,
                candle_count=len(closes),
            )
            self._stats["total_classifications"] += 1
            self._stats["unknown"] += 1
            return RegimeState.UNKNOWN

        dcr = _compute_dcr(closes)

        # For NR, use the full high-low range of recent candles (not just closes)
        # This gives a better picture of actual price movement
        if hasattr(recent[0], 'high') and hasattr(recent[0], 'low'):
            tick_range = max(c.high for c in recent) - min(c.low for c in recent)
            nr = tick_range / self.baseline_vol if self.baseline_vol > 0 else 0.0
        else:
            nr = _compute_nr(closes, self.baseline_vol)

        state = _classify_regime(
            dcr=dcr,
            nr=nr,
            dcr_trend=self.config.dcr_trend_threshold,
            dcr_chop=self.config.dcr_chop_threshold,
            nr_trend=self.config.nr_trend_threshold,
            nr_chop=self.config.nr_chop_threshold,
        )

        self.last_regime = state
        self.last_snapshot = RegimeSnapshot(
            state=state,
            dcr=dcr,
            nr=nr,
            baseline_vol=self.baseline_vol,
            candle_count=len(closes),
        )

        self._stats["total_classifications"] += 1
        self._stats[state.value.lower()] += 1
        if state == RegimeState.CHOPPY:
            self._stats["choppy_blocks"] += 1

        # Log regime changes and periodic diagnostics
        self._classify_count += 1
        if state != self._prev_logged_state or self._classify_count % 50 == 0:
            logger.info(
                f"[REGIME] state={state.value} dcr={dcr:.3f} nr={nr:.3f} "
                f"baseline_vol={self.baseline_vol:.6f} baseline_init={self.baseline_initialized} "
                f"candles={len(closes)}"
            )
            self._prev_logged_state = state
        else:
            logger.debug(
                f"[REGIME] state={state.value} dcr={dcr:.3f} nr={nr:.3f} candles={len(closes)}"
            )

        return state

    def should_allow_entry(self, candles, confidence: float) -> tuple[bool, str]:
        """
        High-level entry check combining regime classification with conviction override.

        Args:
            candles: List of candle objects
            confidence: The strategy confidence score (0.0 - 1.0)

        Returns:
            (allowed: bool, reason: str)
        """
        if not self.config.enabled:
            return True, "regime_filter=disabled"

        regime = self.classify_from_candles(candles)

        if regime == RegimeState.TRENDING:
            return True, "regime=TRENDING"

        if regime == RegimeState.CHOPPY:
            return False, "regime=CHOPPY (no override)"

        # UNKNOWN: allow only if confidence exceeds override threshold
        if confidence >= self.config.regime_override_conviction:
            logger.info(
                f"[REGIME] UNKNOWN but confidence={confidence:.3f} >= "
                f"{self.config.regime_override_conviction} -> override allowed"
            )
            return True, f"regime=UNKNOWN, confidence_override={confidence:.3f}"

        return False, f"regime=UNKNOWN, confidence={confidence:.3f} < override threshold"

    def get_snapshot(self) -> Optional[RegimeSnapshot]:
        """Get the last classification snapshot."""
        return self.last_snapshot

    def get_stats(self) -> dict:
        """Return classification statistics."""
        return dict(self._stats)

    def set_baseline_vol(self, vol: float) -> None:
        """
        Manually set the baseline volatility.
        Useful for initialization from saved state or testing.
        """
        self.baseline_vol = vol
        self.baseline_initialized = True


# ─── Pure functions (identical to TARB — testable independently) ──────────────

def _compute_dcr(ticks: List[float]) -> float:
    """
    Compute Directional Consistency Ratio.

    Measures what percentage of consecutive price ticks move in the
    dominant direction. Returns 0.5 (random) to 1.0 (perfect trend).

    In SimpleDirect context, "ticks" are candle close prices.
    """
    if len(ticks) < 2:
        return 0.5

    deltas = [t2 - t1 for t1, t2 in zip(ticks[:-1], ticks[1:])]
    if not deltas:
        return 0.5

    # Filter out zero deltas (no price change)
    non_zero = [d for d in deltas if d != 0]
    if not non_zero:
        return 0.5

    positive_ratio = sum(1 for d in non_zero if d > 0) / len(non_zero)
    dcr = max(positive_ratio, 1 - positive_ratio)

    return dcr


def _compute_nr(ticks: List[float], baseline_vol: float) -> float:
    """
    Compute Normalized Range.

    Price range over lookback window divided by baseline volatility.
    Higher NR = bigger move relative to normal.
    """
    if len(ticks) < 2 or baseline_vol <= 0:
        return 0.0

    price_range = max(ticks) - min(ticks)
    return price_range / baseline_vol


def _classify_regime(
    dcr: float,
    nr: float,
    dcr_trend: float,
    dcr_chop: float,
    nr_trend: float,
    nr_chop: float,
) -> RegimeState:
    """
    Core classification logic (IDENTICAL to TARB).

    TRENDING: dcr >= dcr_trend AND nr >= nr_trend
    CHOPPY:   dcr <= dcr_chop OR nr <= nr_chop
    UNKNOWN:  everything else
    """
    if dcr >= dcr_trend and nr >= nr_trend:
        return RegimeState.TRENDING

    if dcr <= dcr_chop or nr <= nr_chop:
        return RegimeState.CHOPPY

    return RegimeState.UNKNOWN


def _update_baseline_vol(new_range: float, current_ema: float, alpha: float = 0.1) -> float:
    """EMA baseline volatility update."""
    return alpha * new_range + (1 - alpha) * current_ema
