"""
SimpleDirect Module 3: Cumulative Drawdown Kill Switch
Adapted from GRIDPHANTOM-TARB core/drawdown_guard.py

Three-tier escalation (YELLOW/ORANGE/RED) based on drawdown from
persisted high-water mark (HWM). Prevents slow multi-day capital bleed
that the daily 25% loss limit cannot catch.

Tiers:
    GREEN  - Normal operation (no drawdown action)
    YELLOW - 15% from HWM -> size_mult=0.50, log warning
    ORANGE - 25% from HWM -> size_mult=0.25, min_confidence=0.90, log urgent
    RED    - 40% from HWM -> HALT all trading, write HALTED file

HWM persists to data/hwm_state.json across restarts.
RED tier requires --acknowledge-drawdown CLI flag (manual override) to resume.

TARB Adaptation Notes:
    - Logger name changed from gridphantom.drawdown_guard to drawdown_guard
    - Log messages use text tags [RED] [ORANGE] [YELLOW] [GREEN] (no emoji, Windows safe)
    - References "confidence" instead of "conviction" to match SimpleDirect terminology
    - Core logic is IDENTICAL to TARB
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("drawdown_guard")


class DrawdownTier(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


@dataclass
class DrawdownConfig:
    """Configuration for drawdown guard thresholds and behavior."""
    yellow_threshold: float = 0.15       # 15% drawdown from HWM
    orange_threshold: float = 0.25       # 25% drawdown from HWM
    red_threshold: float = 0.40          # 40% drawdown from HWM

    yellow_size_mult: float = 0.50       # Size multiplier in YELLOW
    orange_size_mult: float = 0.25       # Size multiplier in ORANGE
    orange_min_confidence: float = 0.90  # Min confidence floor in ORANGE

    yellow_recovery: float = 0.10        # Recover from YELLOW when within 10% of HWM
    orange_recovery: float = 0.15        # Recover from ORANGE when within 15% of HWM

    hwm_state_path: str = "data/hwm_state.json"
    halted_file_path: str = "data/HALTED"

    # Enable/disable
    enabled: bool = True

    def validate(self):
        """Validate configuration parameters."""
        if not (0 < self.yellow_threshold < self.orange_threshold < self.red_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must be ordered: 0 < yellow({self.yellow_threshold}) "
                f"< orange({self.orange_threshold}) < red({self.red_threshold}) <= 1.0"
            )
        if not (0 < self.yellow_size_mult <= 1.0):
            raise ValueError(f"yellow_size_mult must be in (0, 1.0], got {self.yellow_size_mult}")
        if not (0 < self.orange_size_mult <= self.yellow_size_mult):
            raise ValueError(
                f"orange_size_mult ({self.orange_size_mult}) must be in "
                f"(0, yellow_size_mult={self.yellow_size_mult}]"
            )
        if not (0 < self.orange_min_confidence <= 1.0):
            raise ValueError(f"orange_min_confidence must be in (0, 1.0], got {self.orange_min_confidence}")
        if not (0 < self.yellow_recovery < self.yellow_threshold):
            raise ValueError(
                f"yellow_recovery ({self.yellow_recovery}) must be less than "
                f"yellow_threshold ({self.yellow_threshold})"
            )
        if not (0 < self.orange_recovery < self.orange_threshold):
            raise ValueError(
                f"orange_recovery ({self.orange_recovery}) must be less than "
                f"orange_threshold ({self.orange_threshold})"
            )


@dataclass
class DrawdownState:
    """Persisted state for drawdown tracking."""
    high_water_mark: float = 0.0
    current_bankroll: float = 0.0
    tier: DrawdownTier = DrawdownTier.GREEN
    last_updated: float = 0.0
    tier_entry_time: float = 0.0
    halted: bool = False

    def drawdown_pct(self) -> float:
        """Current drawdown as a percentage of HWM. Returns 0.0 if HWM is 0."""
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.current_bankroll) / self.high_water_mark

    def to_dict(self) -> dict:
        return {
            "high_water_mark": self.high_water_mark,
            "current_bankroll": self.current_bankroll,
            "tier": self.tier.value,
            "last_updated": self.last_updated,
            "tier_entry_time": self.tier_entry_time,
            "halted": self.halted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DrawdownState":
        return cls(
            high_water_mark=float(d.get("high_water_mark", 0.0)),
            current_bankroll=float(d.get("current_bankroll", 0.0)),
            tier=DrawdownTier(d.get("tier", "GREEN")),
            last_updated=float(d.get("last_updated", 0.0)),
            tier_entry_time=float(d.get("tier_entry_time", 0.0)),
            halted=bool(d.get("halted", False)),
        )


class DrawdownGuard:
    """
    Cumulative drawdown kill switch with three-tier escalation.

    Usage:
        guard = DrawdownGuard(config, initial_bankroll=500.0)
        # On each trade resolution:
        guard.update_bankroll(new_bankroll)
        # Before placing a trade:
        if guard.is_halted():
            # Don't trade
        mult = guard.get_size_multiplier()
        min_conf = guard.get_min_confidence()
    """

    def __init__(
        self,
        config: Optional[DrawdownConfig] = None,
        initial_bankroll: float = 0.0,
        acknowledge_drawdown: bool = False,
        time_fn=None,
    ):
        self.config = config or DrawdownConfig()
        self.config.validate()
        self._time_fn = time_fn or time.time
        self._acknowledge_drawdown = acknowledge_drawdown

        # Try to load persisted state
        self.state = self._load_state()

        if self.state is None:
            # Fresh start
            self.state = DrawdownState(
                high_water_mark=initial_bankroll,
                current_bankroll=initial_bankroll,
                tier=DrawdownTier.GREEN,
                last_updated=self._time_fn(),
                tier_entry_time=self._time_fn(),
                halted=False,
            )
            self._save_state()
            logger.info(
                f"DrawdownGuard initialized fresh. HWM=${initial_bankroll:.2f}, "
                f"tier={self.state.tier.value}"
            )
        else:
            # Restored from persistence
            if self.state.halted and not self._acknowledge_drawdown:
                logger.critical(
                    "[RED] DRAWDOWN GUARD: Trading is HALTED (RED tier). "
                    "Use --acknowledge-drawdown flag to resume."
                )
            elif self.state.halted and self._acknowledge_drawdown:
                logger.warning(
                    "[ORANGE] DRAWDOWN GUARD: RED tier acknowledged. "
                    "Resetting to ORANGE with current bankroll."
                )
                self.state.halted = False
                self.state.tier = DrawdownTier.ORANGE
                self.state.tier_entry_time = self._time_fn()
                self._remove_halted_file()
                self._save_state()

            # If initial_bankroll is provided and higher than stored, update
            if initial_bankroll > 0 and initial_bankroll > self.state.current_bankroll:
                self.state.current_bankroll = initial_bankroll
                if initial_bankroll > self.state.high_water_mark:
                    self.state.high_water_mark = initial_bankroll
                self._save_state()

            logger.info(
                f"DrawdownGuard restored. HWM=${self.state.high_water_mark:.2f}, "
                f"bankroll=${self.state.current_bankroll:.2f}, "
                f"tier={self.state.tier.value}, halted={self.state.halted}"
            )

    def update_bankroll(self, new_bankroll: float) -> DrawdownTier:
        """
        Update bankroll after a trade resolves. Recalculates tier.
        Updates HWM if new_bankroll is a new high.
        Returns the current tier after update.
        """
        old_tier = self.state.tier
        self.state.current_bankroll = new_bankroll
        self.state.last_updated = self._time_fn()

        # Update HWM on new highs
        if new_bankroll > self.state.high_water_mark:
            self.state.high_water_mark = new_bankroll
            logger.info(f"New high-water mark: ${new_bankroll:.2f}")

        # Calculate current drawdown
        dd_pct = self.state.drawdown_pct()

        # Determine tier (check from most severe first)
        new_tier = self._classify_tier(dd_pct, old_tier)

        if new_tier != old_tier:
            self._handle_tier_transition(old_tier, new_tier, dd_pct)

        self.state.tier = new_tier
        self._save_state()

        return new_tier

    def _classify_tier(self, dd_pct: float, current_tier: DrawdownTier) -> DrawdownTier:
        """
        Determine tier based on drawdown percentage.
        Uses hysteresis: recovery thresholds are tighter than entry thresholds
        to prevent rapid tier oscillation.
        """
        # RED recovery is manual only
        if current_tier == DrawdownTier.RED:
            return DrawdownTier.RED

        # Escalation (always immediate, check from most severe)
        if dd_pct >= self.config.red_threshold:
            return DrawdownTier.RED
        if dd_pct >= self.config.orange_threshold:
            return DrawdownTier.ORANGE
        if dd_pct >= self.config.yellow_threshold:
            if current_tier == DrawdownTier.ORANGE:
                if dd_pct <= self.config.orange_recovery:
                    return DrawdownTier.GREEN
                return DrawdownTier.ORANGE
            return DrawdownTier.YELLOW

        # Below yellow threshold — check recovery hysteresis
        if current_tier == DrawdownTier.ORANGE:
            if dd_pct <= self.config.orange_recovery:
                return DrawdownTier.GREEN
            return DrawdownTier.ORANGE
        elif current_tier == DrawdownTier.YELLOW:
            if dd_pct <= self.config.yellow_recovery:
                return DrawdownTier.GREEN
            return DrawdownTier.YELLOW

        return DrawdownTier.GREEN

    def _handle_tier_transition(self, old_tier: DrawdownTier, new_tier: DrawdownTier, dd_pct: float):
        """Handle logging and side effects of tier changes."""
        dd_pct_display = dd_pct * 100

        if new_tier == DrawdownTier.RED:
            self.state.halted = True
            self.state.tier_entry_time = self._time_fn()
            self._write_halted_file()
            logger.critical(
                f"[RED] DRAWDOWN RED -- TRADING HALTED. "
                f"Drawdown: {dd_pct_display:.1f}% from HWM ${self.state.high_water_mark:.2f}. "
                f"Bankroll: ${self.state.current_bankroll:.2f}. "
                f"Manual --acknowledge-drawdown required to resume."
            )
        elif new_tier == DrawdownTier.ORANGE:
            self.state.tier_entry_time = self._time_fn()
            logger.warning(
                f"[ORANGE] DRAWDOWN ORANGE -- Size->{self.config.orange_size_mult}x, "
                f"min confidence->{self.config.orange_min_confidence}. "
                f"Drawdown: {dd_pct_display:.1f}% from HWM ${self.state.high_water_mark:.2f}. "
                f"Bankroll: ${self.state.current_bankroll:.2f}"
            )
        elif new_tier == DrawdownTier.YELLOW:
            self.state.tier_entry_time = self._time_fn()
            logger.warning(
                f"[YELLOW] DRAWDOWN YELLOW -- Size->{self.config.yellow_size_mult}x. "
                f"Drawdown: {dd_pct_display:.1f}% from HWM ${self.state.high_water_mark:.2f}. "
                f"Bankroll: ${self.state.current_bankroll:.2f}"
            )
        elif new_tier == DrawdownTier.GREEN:
            logger.info(
                f"[GREEN] DRAWDOWN RECOVERED to GREEN. "
                f"Previous tier: {old_tier.value}. "
                f"Bankroll: ${self.state.current_bankroll:.2f}, "
                f"HWM: ${self.state.high_water_mark:.2f}"
            )

    def is_halted(self) -> bool:
        """Check if trading is halted (RED tier)."""
        return self.state.halted

    def get_size_multiplier(self) -> float:
        """Get position size multiplier based on current tier."""
        if self.state.tier == DrawdownTier.RED:
            return 0.0
        elif self.state.tier == DrawdownTier.ORANGE:
            return self.config.orange_size_mult
        elif self.state.tier == DrawdownTier.YELLOW:
            return self.config.yellow_size_mult
        return 1.0  # GREEN

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold based on current tier."""
        if self.state.tier == DrawdownTier.ORANGE:
            return self.config.orange_min_confidence
        return 0.0

    def get_status(self) -> dict:
        """Get current status summary for dashboard/logging."""
        return {
            "tier": self.state.tier.value,
            "halted": self.state.halted,
            "high_water_mark": self.state.high_water_mark,
            "current_bankroll": self.state.current_bankroll,
            "drawdown_pct": round(self.state.drawdown_pct() * 100, 2),
            "size_multiplier": self.get_size_multiplier(),
            "min_confidence": self.get_min_confidence(),
            "last_updated": self.state.last_updated,
        }

    # --- Persistence ---

    def _load_state(self) -> Optional[DrawdownState]:
        """Load persisted HWM state from disk."""
        path = Path(self.config.hwm_state_path)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            state = DrawdownState.from_dict(data)
            logger.info(f"Loaded HWM state from {path}")
            return state
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load HWM state from {path}: {e}. Starting fresh.")
            return None

    def _save_state(self):
        """Persist HWM state to disk."""
        path = Path(self.config.hwm_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save HWM state to {path}: {e}")

    def _write_halted_file(self):
        """Write HALTED sentinel file for RED tier."""
        path = Path(self.config.halted_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                f.write(
                    f"TRADING HALTED -- RED TIER\n"
                    f"Time: {self._time_fn()}\n"
                    f"HWM: ${self.state.high_water_mark:.2f}\n"
                    f"Bankroll: ${self.state.current_bankroll:.2f}\n"
                    f"Drawdown: {self.state.drawdown_pct() * 100:.1f}%\n"
                    f"Use --acknowledge-drawdown to resume.\n"
                )
            logger.info(f"HALTED file written to {path}")
        except OSError as e:
            logger.error(f"Failed to write HALTED file to {path}: {e}")

    def _remove_halted_file(self):
        """Remove HALTED sentinel file."""
        path = Path(self.config.halted_file_path)
        try:
            if path.exists():
                path.unlink()
                logger.info(f"HALTED file removed from {path}")
        except OSError as e:
            logger.error(f"Failed to remove HALTED file from {path}: {e}")
