"""
SimpleDirect Module 4: Confidence-Scaled Position Sizing
Adapted from GRIDPHANTOM-TARB core/position_sizer.py

Replaces flat Kelly sizing with tiered confidence multipliers and
bankroll-aware caps. Integrates with Module 3 drawdown multiplier.

Design: Option C -- bankroll-tiered caps + lower base % for small bankrolls.

Confidence Tiers:
    0.60-0.69  Marginal     0.5x
    0.70-0.79  Standard     0.75x
    0.80-0.89  High         1.0x (base)
    0.90-0.95  Very High    1.3x
    0.95-1.00  Extreme      1.6x

Bankroll Guards:
    < $200   -> max multiplier capped at 1.0x, base_pct = small_bankroll_pct
    $200-500 -> max multiplier capped at 1.3x, base_pct transitions toward full
    > $500   -> full multipliers unlocked, base_pct = full rate

TARB Adaptation Notes:
    - "conviction" renamed to "confidence" to match SimpleDirect terminology
    - Core sizing logic is IDENTICAL to TARB
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logger = logging.getLogger("position_sizer")


@dataclass
class SizerConfig:
    """Configuration for confidence-scaled position sizing."""

    # ── Confidence Tiers ──
    # Each tier: (min_confidence, max_confidence, multiplier, label)
    confidence_tiers: List[Tuple[float, float, float, str]] = field(default_factory=lambda: [
        (0.60, 0.70, 0.50, "Marginal"),
        (0.70, 0.80, 0.75, "Standard"),
        (0.80, 0.90, 1.00, "High"),
        (0.90, 0.95, 1.30, "Very High"),
        (0.95, 1.01, 1.60, "Extreme"),   # 1.01 upper to catch 1.0 exactly
    ])

    # ── Base Sizing ──
    per_market_budget_pct: float = 0.15       # Full base % of bankroll per position (TARB live)
    small_bankroll_budget_pct: float = 0.08   # Reduced base % for small bankrolls (TARB live)

    # ── Bankroll Tier Thresholds ──
    small_bankroll_cap: float = 200.0         # Below this: cap mult at 1.0x, use small base %
    mid_bankroll_cap: float = 500.0           # Below this: cap mult at 1.3x, interpolate base %

    # ── Hard Limits ──
    min_order_usd: float = 1.0                # Polymarket minimum
    max_order_usd: float = 50.0               # Hard cap per position

    def validate(self):
        """Validate configuration parameters."""
        if not (0 < self.small_bankroll_budget_pct <= self.per_market_budget_pct <= 1.0):
            raise ValueError(
                f"Budget pcts must be: 0 < small({self.small_bankroll_budget_pct}) "
                f"<= full({self.per_market_budget_pct}) <= 1.0"
            )
        if not (0 < self.small_bankroll_cap < self.mid_bankroll_cap):
            raise ValueError(
                f"Bankroll caps must be: 0 < small({self.small_bankroll_cap}) "
                f"< mid({self.mid_bankroll_cap})"
            )
        if self.min_order_usd < 0:
            raise ValueError(f"min_order_usd must be >= 0, got {self.min_order_usd}")
        if self.max_order_usd < self.min_order_usd:
            raise ValueError(
                f"max_order_usd ({self.max_order_usd}) must be >= "
                f"min_order_usd ({self.min_order_usd})"
            )
        for i, (lo, hi, mult, label) in enumerate(self.confidence_tiers):
            if lo >= hi:
                raise ValueError(f"Tier {label}: min ({lo}) must be < max ({hi})")
            if mult <= 0:
                raise ValueError(f"Tier {label}: multiplier must be > 0, got {mult}")


class PositionSizer:
    """
    Confidence-scaled position sizer with bankroll-aware caps.

    Usage:
        sizer = PositionSizer(config)
        size = sizer.compute_size(
            confidence=0.82,
            bankroll=150.0,
            drawdown_mult=1.0,  # from Module 3
        )
    """

    def __init__(self, config: Optional[SizerConfig] = None):
        self.config = config or SizerConfig()
        self.config.validate()

    def compute_size(
        self,
        confidence: float,
        bankroll: float,
        drawdown_mult: float = 1.0,
    ) -> float:
        """
        Compute position size in USD.

        Args:
            confidence: Strategy confidence score (0.0-1.0)
            bankroll: Current bankroll in USD
            drawdown_mult: Multiplier from M3 drawdown guard (1.0/0.5/0.25/0.0)

        Returns:
            Position size in USD, clamped to [min_order, max_order].
            Returns 0.0 if inputs are invalid or drawdown_mult is 0.
        """
        if drawdown_mult <= 0:
            return 0.0
        if bankroll <= 0 or confidence < 0:
            return 0.0

        conv_mult = self._get_confidence_multiplier(confidence)
        if conv_mult == 0.0:
            return 0.0

        capped_mult = self._apply_bankroll_cap(conv_mult, bankroll)
        base_pct = self._get_base_pct(bankroll)

        raw_size = bankroll * base_pct * capped_mult * drawdown_mult

        if raw_size < self.config.min_order_usd:
            return 0.0
        size = min(raw_size, self.config.max_order_usd)

        return round(size, 2)

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Look up confidence tier multiplier."""
        for lo, hi, mult, label in self.config.confidence_tiers:
            if lo <= confidence < hi:
                return mult
        if confidence < self.config.confidence_tiers[0][0]:
            return 0.0
        return self.config.confidence_tiers[-1][2]

    def _apply_bankroll_cap(self, conf_mult: float, bankroll: float) -> float:
        """Cap confidence multiplier based on bankroll size."""
        if bankroll < self.config.small_bankroll_cap:
            return min(conf_mult, 1.0)
        elif bankroll < self.config.mid_bankroll_cap:
            return min(conf_mult, 1.3)
        else:
            return conf_mult

    def _get_base_pct(self, bankroll: float) -> float:
        """
        Get base budget percentage based on bankroll size.
        Interpolates linearly between small and full rates in the mid zone.
        """
        if bankroll <= self.config.small_bankroll_cap:
            return self.config.small_bankroll_budget_pct
        elif bankroll >= self.config.mid_bankroll_cap:
            return self.config.per_market_budget_pct
        else:
            progress = (bankroll - self.config.small_bankroll_cap) / (
                self.config.mid_bankroll_cap - self.config.small_bankroll_cap
            )
            return (
                self.config.small_bankroll_budget_pct
                + progress * (self.config.per_market_budget_pct - self.config.small_bankroll_budget_pct)
            )

    def get_sizing_info(
        self,
        confidence: float,
        bankroll: float,
        drawdown_mult: float = 1.0,
    ) -> dict:
        """Get detailed sizing breakdown for logging/dashboard."""
        conv_mult = self._get_confidence_multiplier(confidence)
        capped_mult = self._apply_bankroll_cap(conv_mult, bankroll) if conv_mult > 0 else 0.0
        base_pct = self._get_base_pct(bankroll)
        size = self.compute_size(confidence, bankroll, drawdown_mult)

        tier_label = "Below minimum"
        for lo, hi, mult, label in self.config.confidence_tiers:
            if lo <= confidence < hi:
                tier_label = label
                break

        if bankroll < self.config.small_bankroll_cap:
            bankroll_tier = "Small"
        elif bankroll < self.config.mid_bankroll_cap:
            bankroll_tier = "Mid"
        else:
            bankroll_tier = "Full"

        return {
            "confidence": confidence,
            "confidence_tier": tier_label,
            "confidence_mult": conv_mult,
            "capped_mult": capped_mult,
            "was_capped": capped_mult < conv_mult,
            "bankroll": bankroll,
            "bankroll_tier": bankroll_tier,
            "base_pct": round(base_pct, 4),
            "drawdown_mult": drawdown_mult,
            "final_size": size,
        }
