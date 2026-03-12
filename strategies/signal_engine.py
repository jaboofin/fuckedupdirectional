"""
╔══════════════════════════════════════════════════════════════════╗
║  STRATEGY ENGINE — BTC 15-MIN POLYMARKET PREDICTOR               ║
║                                                                    ║
║  Predicts: will BTC close ABOVE or BELOW the window open price?  ║
║                                                                    ║
║  Combines:                                                         ║
║    1. Price vs Open — where is BTC now vs the window open?        ║
║    2. Momentum — short-term directional pressure                  ║
║    3. RSI — overbought/oversold                                   ║
║    4. MACD — trend strength + crossover                           ║
║    5. EMA Cross — fast/slow trend shift                           ║
║                                                                    ║
║  The open-price anchor is critical: Polymarket resolves against   ║
║  Chainlink BTC/USD at window start vs window end.                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from config.settings import MarketDirection, StrategyConfig
from oracles.price_feed import Candle

logger = logging.getLogger("strategy")


@dataclass
class Signal:
    name: str
    direction: MarketDirection
    strength: float  # 0.0 to 1.0
    raw_value: float
    description: str


@dataclass
class StrategyDecision:
    direction: MarketDirection
    confidence: float
    signals: list[Signal]
    current_price: float
    open_price: Optional[float]   # Window anchor
    drift_pct: Optional[float]    # Current vs open
    volatility_pct: float
    should_trade: bool
    reason: str
    position_size_pct: float

    def summary(self) -> str:
        sigs = " | ".join(f"{s.name}={s.direction.value}({s.strength:.2f})" for s in self.signals)
        drift = f" drift={self.drift_pct:+.3f}%" if self.drift_pct is not None else ""
        return (
            f"[{self.direction.value.upper()}] conf={self.confidence:.2f}{drift} "
            f"trade={self.should_trade} | {sigs}"
        )


class StrategyEngine:
    """
    Multi-signal strategy anchored to the window opening price.

    The key insight: Polymarket 15-min BTC markets resolve as
    UP if chainlink_close >= chainlink_open, else DOWN.

    So the question isn't "will BTC go up?" — it's
    "will BTC be above WHERE IT WAS when this window opened?"

    If BTC already drifted +0.2% above the open in the first minute,
    that changes the probability significantly.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._trade_history: list[StrategyDecision] = []

    # ── Technical Indicators ─────────────────────────────────────

    @staticmethod
    def _ema(data: list[float], period: int) -> list[float]:
        if len(data) < period:
            return [sum(data) / len(data)] * len(data)
        multiplier = 2 / (period + 1)
        ema_values = [sum(data[:period]) / period]
        for price in data[period:]:
            ema_values.append(price * multiplier + ema_values[-1] * (1 - multiplier))
        return ema_values

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    @staticmethod
    def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0
        ema_fast = StrategyEngine._ema(closes, fast)
        ema_slow = StrategyEngine._ema(closes, slow)
        min_len = min(len(ema_fast), len(ema_slow))
        macd_line = [ema_fast[-(min_len - i)] - ema_slow[-(min_len - i)] for i in range(min_len)]
        if len(macd_line) < signal:
            return macd_line[-1] if macd_line else 0.0, 0.0, 0.0
        signal_line = StrategyEngine._ema(macd_line, signal)
        return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]

    def _volatility(self, candles: list[Candle]) -> float:
        if len(candles) < 2:
            return 0.0
        returns = [((candles[i].close - candles[i-1].close) / candles[i-1].close) * 100 for i in range(1, len(candles))]
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))

    # ── Signal Generators ────────────────────────────────────────

    def _signal_price_vs_open(self, current_price: float, open_price: float,
                                dead_zone_pct: float = 0.04) -> Signal:
        """
        THE KEY SIGNAL: Where is BTC now relative to the window open?

        If BTC is already 0.15% above the open, it's more likely to
        close above (UP). If it's 0.3% below, DOWN is more likely.

        This directly maps to what Polymarket is resolving on.

        Dead zone: < dead_zone_pct drift is noise — price hasn't committed.
        Per-asset: BTC=0.04%, ETH=0.05%, SOL=0.03% (from AssetConfig).
        """
        drift_pct = ((current_price - open_price) / open_price) * 100

        if drift_pct > dead_zone_pct:
            direction = MarketDirection.UP
        elif drift_pct < -dead_zone_pct:
            direction = MarketDirection.DOWN
        else:
            direction = MarketDirection.HOLD

        # Strength scales with drift magnitude
        # 0.04% drift = weak, 0.08% = moderate, 0.15%+ = strong
        strength = min(1.0, abs(drift_pct) / 0.15)

        return Signal(
            "price_vs_open", direction, strength, drift_pct,
            f"Price vs window open: {drift_pct:+.4f}%"
        )

    def _signal_momentum(self, candles: list[Candle], lookback: Optional[int] = None) -> Signal:
        _lookback = min(lookback or self.config.momentum_lookback, len(candles) - 1)
        if _lookback < 1:
            return Signal("momentum", MarketDirection.HOLD, 0.0, 0.0, "No data")
        current = candles[-1].close
        past = candles[-(_lookback + 1)].close
        pct = ((current - past) / past) * 100
        strength = min(1.0, abs(pct) / 0.5)
        if pct > 0.02:
            d = MarketDirection.UP
        elif pct < -0.02:
            d = MarketDirection.DOWN
        else:
            d = MarketDirection.HOLD
            strength = 0.0
        return Signal("momentum", d, strength, pct, f"{_lookback}-candle: {pct:+.3f}%")

    def _signal_rsi(self, candles: list[Candle], period: Optional[int] = None) -> Signal:
        closes = [c.close for c in candles]
        _period = period or self.config.rsi_period
        rsi = self._rsi(closes, _period)
        if rsi > self.config.rsi_overbought:
            d, strength = MarketDirection.DOWN, min(1.0, (rsi - self.config.rsi_overbought) / 15)
        elif rsi < self.config.rsi_oversold:
            d, strength = MarketDirection.UP, min(1.0, (self.config.rsi_oversold - rsi) / 15)
        else:
            center = 50.0
            if rsi > center:
                d = MarketDirection.UP
                strength = (rsi - center) / (self.config.rsi_overbought - center) * 0.3
            else:
                d = MarketDirection.DOWN
                strength = (center - rsi) / (center - self.config.rsi_oversold) * 0.3
        return Signal("rsi", d, strength, rsi, f"RSI={rsi:.1f}")

    def _signal_macd(self, candles: list[Candle], fast: Optional[int] = None,
                     slow: Optional[int] = None, signal_period: Optional[int] = None) -> Signal:
        closes = [c.close for c in candles]
        _fast = fast or self.config.macd_fast
        _slow = slow or self.config.macd_slow
        _signal = signal_period or self.config.macd_signal
        macd_line, signal_line, histogram = self._macd(closes, _fast, _slow, _signal)
        d = MarketDirection.UP if histogram > 0 else MarketDirection.DOWN if histogram < 0 else MarketDirection.HOLD
        normalized = abs(histogram) / (closes[-1] if closes else 1) * 10000
        strength = min(1.0, normalized / 10)
        if len(closes) > 2:
            prev = self._macd(closes[:-1], _fast, _slow, _signal)
            if prev[2] * histogram < 0:
                strength = min(1.0, strength * 1.5)
        return Signal("macd", d, strength, histogram, f"MACD hist={histogram:.2f}")

    def _signal_ema_cross(self, candles: list[Candle], fast: Optional[int] = None,
                          slow: Optional[int] = None) -> Signal:
        closes = [c.close for c in candles]
        ema_fast = self._ema(closes, fast or self.config.ema_fast)
        ema_slow = self._ema(closes, slow or self.config.ema_slow)
        if not ema_fast or not ema_slow:
            return Signal("ema_cross", MarketDirection.HOLD, 0.0, 0.0, "No data")
        diff = ema_fast[-1] - ema_slow[-1]
        d = MarketDirection.UP if diff > 0 else MarketDirection.DOWN if diff < 0 else MarketDirection.HOLD
        spread_pct = abs(diff) / closes[-1] * 100
        strength = min(1.0, spread_pct / 0.15)
        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            prev_diff = ema_fast[-2] - ema_slow[-2]
            if prev_diff * diff < 0:
                strength = min(1.0, strength * 2.0)
        return Signal("ema_cross", d, strength, diff, f"EMA diff={diff:.2f}")

    # ── Master Decision ──────────────────────────────────────────

    def analyze(self, candles: list[Candle], current_price: float,
                open_price: Optional[float] = None,
                fee_pct: Optional[float] = None,
                timeframe_overrides: Optional[dict] = None) -> StrategyDecision:
        """
        Run all signals and produce a weighted decision.

        Args:
            candles: Historical candles (oldest first)
            current_price: Latest BTC price (from Chainlink ideally)
            open_price: The Chainlink price at the start of this window.
                        If provided, price_vs_open becomes the highest-weighted signal.
            fee_pct: Actual taker fee % from Polymarket API for this market.
                     If None, falls back to conservative 1.56% estimate.
            timeframe_overrides: Optional dict to override indicator params for
                                 different timeframes or assets. Supported keys:
                                   rsi_period, macd_fast, macd_slow, macd_signal,
                                   ema_fast, ema_slow, momentum_lookback,
                                   weight_price_vs_open, dead_zone_pct,
                                   confidence_threshold, max_volatility_pct
        """
        drift_pct = None

        # ── Resolve indicator params (allow timeframe/asset overrides) ──
        ovr = timeframe_overrides or {}
        _rsi_period = ovr.get("rsi_period", self.config.rsi_period)
        _macd_fast = ovr.get("macd_fast", self.config.macd_fast)
        _macd_slow = ovr.get("macd_slow", self.config.macd_slow)
        _macd_signal = ovr.get("macd_signal", self.config.macd_signal)
        _ema_fast = ovr.get("ema_fast", self.config.ema_fast)
        _ema_slow = ovr.get("ema_slow", self.config.ema_slow)
        _momentum_lookback = ovr.get("momentum_lookback", self.config.momentum_lookback)
        _pvo_weight = ovr.get("weight_price_vs_open", 0.70)  # default 70% for 15m
        _dead_zone_pct = ovr.get("dead_zone_pct", 0.04)      # per-asset: BTC=0.04, ETH=0.05, SOL=0.03
        _confidence_threshold = ovr.get("confidence_threshold", self.config.confidence_threshold)
        _max_volatility_pct = ovr.get("max_volatility_pct", self.config.max_volatility_pct)
        _indicator_weight = ovr.get("indicator_agreement_weight", 0.10)  # per-indicator confidence adjustment

        if len(candles) < 30:
            return StrategyDecision(
                MarketDirection.HOLD, 0.0, [], current_price, open_price,
                None, 0.0, False, "Insufficient data (<30 candles)", 0.0,
            )

        volatility = self._volatility(candles[-20:])
        if volatility < self.config.min_volatility_pct:
            return StrategyDecision(
                MarketDirection.HOLD, 0.0, [], current_price, open_price,
                None, volatility, False, f"Volatility too low ({volatility:.3f}%)", 0.0,
            )
        if volatility > _max_volatility_pct:
            return StrategyDecision(
                MarketDirection.HOLD, 0.0, [], current_price, open_price,
                None, volatility, False, f"Volatility too high ({volatility:.3f}%)", 0.0,
            )

        # ── Build signals ──
        signals = []
        weights = {}

        if open_price and open_price > 0:
            # Window anchor available — price_vs_open is the DOMINANT signal
            # Weight is 0.70 for 15m, 0.50 for 5m (via timeframe_overrides)
            # Indicators split the remainder
            pvo = self._signal_price_vs_open(current_price, open_price, dead_zone_pct=_dead_zone_pct)
            signals.append(pvo)
            drift_pct = pvo.raw_value

            indicator_share = 1.0 - _pvo_weight
            weights["price_vs_open"] = _pvo_weight
            weights["momentum"] = self.config.weight_momentum * indicator_share
            weights["rsi"] = self.config.weight_rsi * indicator_share
            weights["macd"] = self.config.weight_macd * indicator_share
            weights["ema_cross"] = self.config.weight_ema_cross * indicator_share
        else:
            # No anchor — use original weights
            weights["momentum"] = self.config.weight_momentum
            weights["rsi"] = self.config.weight_rsi
            weights["macd"] = self.config.weight_macd
            weights["ema_cross"] = self.config.weight_ema_cross

        signals.extend([
            self._signal_momentum(candles, lookback=_momentum_lookback),
            self._signal_rsi(candles, period=_rsi_period),
            self._signal_macd(candles, fast=_macd_fast, slow=_macd_slow, signal_period=_macd_signal),
            self._signal_ema_cross(candles, fast=_ema_fast, slow=_ema_slow),
        ])

        # ── Chop filter: indicators split 2v2 = no trend ──
        if open_price and open_price > 0:
            indicator_dirs = [s.direction for s in signals if s.name != "price_vs_open" and s.direction != MarketDirection.HOLD]
            if len(indicator_dirs) >= 4:
                up_count = sum(1 for d in indicator_dirs if d == MarketDirection.UP)
                down_count = sum(1 for d in indicator_dirs if d == MarketDirection.DOWN)
                pvo_drift = abs(((current_price - open_price) / open_price) * 100)
                if up_count == 2 and down_count == 2 and pvo_drift < 0.12:
                    logger.info(f"Chop filter: indicators split 2v2, drift {pvo_drift:.4f}% < 0.12% — holding")
                    return StrategyDecision(
                        MarketDirection.HOLD, 0.0, signals, current_price, open_price,
                        drift_pct, volatility, False,
                        f"Chop detected: indicators split 2v2, drift only {pvo_drift:.4f}%", 0.0,
                    )

        # ── Agreement-weighted confidence ──
        # Old approach: ratio-based (winning_score/total) with 0.92 cap.
        # Problem: PvO at 70% weight dominates so completely that confidence
        # clusters at 0.85-0.92 regardless of indicator agreement. A trade
        # with 4/4 indicators agreeing gets the SAME confidence as 1/4.
        #
        # New approach: drift magnitude sets the base, indicators adjust it.
        # This produces a natural spread (0.55-0.92) where agreement matters.
        #   Strong drift + 4/4 agree → ~0.91 (trade, size up)
        #   Strong drift + 2/4 agree → ~0.73 (trade, smaller)
        #   Strong drift + 0/4 agree → ~0.59 (skip — indicators fight drift)
        #   Medium drift + 4/4 agree → ~0.78 (trade — indicators confirm)
        #   Weak drift + anything   → ~0.55-0.72 (skip)

        # Step 1: Determine direction from weighted scores (same as before)
        up_score = 0.0
        down_score = 0.0
        for sig in signals:
            w = weights.get(sig.name, 0.0)
            if sig.direction == MarketDirection.UP:
                up_score += sig.strength * w
            elif sig.direction == MarketDirection.DOWN:
                down_score += sig.strength * w

        total = up_score + down_score
        if total == 0:
            direction = MarketDirection.HOLD
            confidence = 0.0
        elif up_score > down_score:
            direction = MarketDirection.UP
        else:
            direction = MarketDirection.DOWN

        if direction == MarketDirection.HOLD:
            return StrategyDecision(
                MarketDirection.HOLD, 0.0, signals, current_price, open_price,
                drift_pct, volatility, False, "No directional signal", 0.0,
            )

        # Step 2: Drift base — how far price moved from anchor
        abs_drift = abs(drift_pct) if drift_pct is not None else 0.0
        if abs_drift < _dead_zone_pct:
            drift_base = 0.50  # Deadzone — no directional edge from drift
        else:
            drift_base = 0.50 + min(0.28, (abs_drift - _dead_zone_pct) * 1.35)

        # Step 3: Indicator agreement adjustment
        # Each indicator that agrees with the drift direction adds confidence;
        # each that opposes subtracts. Scaled by signal strength.
        # ±_indicator_weight per indicator × 4 = ±0.40 total swing (at 0.10)
        indicator_signals = [s for s in signals if s.name != "price_vs_open"]
        indicator_adj = 0.0
        agree_count = 0
        disagree_count = 0
        for sig in indicator_signals:
            if sig.direction == MarketDirection.HOLD:
                continue
            if sig.direction == direction:
                indicator_adj += sig.strength * _indicator_weight
                agree_count += 1
            else:
                indicator_adj -= sig.strength * _indicator_weight
                disagree_count += 1

        confidence = max(0.50, min(0.92, drift_base + indicator_adj))

        logger.info(
            f"Confidence: drift_base={drift_base:.3f} + indicator_adj={indicator_adj:+.3f} "
            f"= {confidence:.3f} (agree={agree_count}, oppose={disagree_count})"
        )

        # ── Fee-adjusted edge check ──
        # Uses real fee from Polymarket API if available, else conservative fallback.
        # Fee is highest at 50% odds (~1.56%) and drops toward 0 at extremes.
        est_fee_pct = fee_pct if fee_pct is not None else 1.56  # Dynamic from API or fallback
        raw_edge = abs(confidence - 0.5) * 2 * 100  # Edge as %
        if raw_edge < est_fee_pct and direction != MarketDirection.HOLD:
            logger.info(f"Edge {raw_edge:.1f}% < fee {est_fee_pct:.2f}% — skipping")
            return StrategyDecision(
                direction, confidence, signals, current_price, open_price,
                drift_pct, volatility, False,
                f"Edge ({raw_edge:.1f}%) below fee threshold ({est_fee_pct:.2f}%)", 0.0,
            )

        should_trade = direction != MarketDirection.HOLD and confidence >= _confidence_threshold

        if should_trade:
            kelly = max(0, confidence - (1 - confidence))
            position_size_pct = min(kelly * 100 * 0.25, 10.0)
        else:
            position_size_pct = 0.0

        reason = (
            f"drift_base={drift_base:.3f} + ind_adj={indicator_adj:+.3f} → "
            f"{direction.value} @ {confidence:.3f} "
            f"(agree={agree_count}, oppose={disagree_count})"
        )
        if drift_pct is not None:
            reason += f" drift={drift_pct:+.4f}%"

        decision = StrategyDecision(
            direction=direction, confidence=confidence, signals=signals,
            current_price=current_price, open_price=open_price,
            drift_pct=drift_pct, volatility_pct=volatility,
            should_trade=should_trade, reason=reason,
            position_size_pct=position_size_pct,
        )

        self._trade_history.append(decision)
        logger.info(f"Strategy: {decision.summary()}")
        return decision

    def get_history(self) -> list[StrategyDecision]:
        return self._trade_history.copy()
