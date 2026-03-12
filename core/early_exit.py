"""
╔══════════════════════════════════════════════════════════════════╗
║  EARLY EXIT SYSTEM — Take Profit + Stop Loss + Trailing Stop     ║
║                                                                    ║
║  Monitors open positions and triggers sell orders when price       ║
║  thresholds are hit, instead of always holding to resolution.      ║
║                                                                    ║
║  Three exit types evaluated in priority order:                     ║
║    1. Take Profit  — lock in gains when price >= TP target         ║
║    2. Stop Loss    — cap losses when price <= SL target            ║
║    3. Trailing Stop — sell when price drops X from peak            ║
║                                                                    ║
║  If nothing triggers, position resolves normally (safety net).     ║
║                                                                    ║
║  Phase 1: Pure logic — no network calls. Fully testable.           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config.settings import EarlyExitConfig

logger = logging.getLogger("early_exit")


# ── Exit Signal Types ────────────────────────────────────────────

class ExitType(Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"


@dataclass
class ExitSignal:
    """Returned by check_position() when an exit should fire."""
    trade_id: str
    exit_type: ExitType
    trigger_price: float      # The threshold that was breached
    current_bid: float        # Actual CLOB bid at time of check
    entry_price: float
    gross_pnl_per_share: float
    estimated_fees: float     # Round-trip fee estimate
    net_pnl_per_share: float
    hold_duration_secs: float
    peak_price: Optional[float]  # For trailing stop — highest observed bid
    reason: str


# ── Conviction Tiers ─────────────────────────────────────────────

# Conviction → (tp_multiplier, trail_multiplier)
# SL does NOT scale with conviction per spec
_CONVICTION_TIERS = [
    (0.60, 0.70, 0.7, 0.8),   # Marginal:  0.60-0.69
    (0.70, 0.80, 1.0, 1.0),   # Standard:  0.70-0.79
    (0.80, 0.90, 1.2, 1.1),   # High:      0.80-0.89
    (0.90, 0.95, 1.5, 1.3),   # Very High: 0.90-0.94
    (0.95, 1.01, 2.0, 1.5),   # Extreme:   0.95-1.00
]


def _get_conviction_multipliers(conviction: float) -> tuple[float, float]:
    """Return (tp_multiplier, trail_multiplier) for a given conviction score."""
    for low, high, tp_mult, trail_mult in _CONVICTION_TIERS:
        if low <= conviction < high:
            return tp_mult, trail_mult
    # Fallback: if below 0.60 or above 1.0, use most conservative
    if conviction < 0.60:
        return 0.7, 0.8
    return 2.0, 1.5


# ── Fee Computation ──────────────────────────────────────────────

def compute_polymarket_fee(price: float) -> float:
    """
    Polymarket taker fee for a single trade at a given share price.
    Formula: fee = price * (1 - price) * 0.0625
    Parabolic — highest at 0.50 (~$0.0156), zero at extremes (0 or 1).
    """
    if price <= 0 or price >= 1:
        return 0.0
    return price * (1.0 - price) * 0.0625


def compute_round_trip_fee(entry_price: float, exit_price: float) -> float:
    """Total fees for a buy-then-sell round trip."""
    return compute_polymarket_fee(entry_price) + compute_polymarket_fee(exit_price)


# ── Tracked Position ─────────────────────────────────────────────

@dataclass
class TrackedPosition:
    """
    Tracks a single open position's exit thresholds.
    Computed at registration time from entry price + conviction + config.
    """
    trade_id: str
    token_id: str
    side: str                   # "BUY" — we always buy, early exit sells
    entry_price: float
    conviction: float
    size_usd: float
    registered_at: float        # timestamp
    market_close_ts: float      # when the market window resolves

    # ── Computed thresholds ──
    tp_price: Optional[float] = None       # Sell if bid >= this
    sl_price: Optional[float] = None       # Sell if bid <= this
    trail_activation_price: float = 0.0    # Trail activates when bid >= this
    trail_offset: float = 0.0              # Trail distance from peak
    trail_active: bool = False             # Has trail activated?
    trail_peak: float = 0.0               # Highest observed bid since activation
    trail_trigger_price: float = 0.0       # Sell if bid <= peak - offset

    # ── State ──
    exited: bool = False
    exit_type: Optional[ExitType] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[float] = None

    @property
    def hold_duration_secs(self) -> float:
        return time.time() - self.registered_at

    @property
    def time_remaining_secs(self) -> float:
        return max(0, self.market_close_ts - time.time())

    @property
    def shares_estimate(self) -> float:
        """Approximate number of shares (size / entry_price)."""
        if self.entry_price <= 0:
            return 0.0
        return self.size_usd / self.entry_price


# ── Early Exit Manager ───────────────────────────────────────────

class EarlyExitManager:
    """
    Manages all tracked positions and evaluates exit conditions.

    Pure logic — does not make any network calls itself.
    The bot's exit loop feeds it current CLOB bids and acts on returned signals.
    """

    def __init__(self, config: EarlyExitConfig):
        self.config = config
        self._positions: dict[str, TrackedPosition] = {}
        self._exit_count = {"take_profit": 0, "stop_loss": 0, "trailing_stop": 0}
        self._total_exit_pnl = 0.0

    # ── Registration ─────────────────────────────────────────────

    def register_position(
        self,
        trade_id: str,
        token_id: str,
        entry_price: float,
        conviction: float,
        size_usd: float,
        market_close_ts: float,
        side: str = "BUY",
        # ── Per-asset overrides (Phase E4) ──
        asset_tp_offset: Optional[float] = None,
        asset_sl_offset: Optional[float] = None,
        asset_trail_offset: Optional[float] = None,
    ) -> TrackedPosition:
        """
        Compute exit thresholds and begin tracking a new position.

        Called immediately after a successful fill.
        Per-asset overrides (from AssetConfig) take precedence over global config.
        """
        if trade_id in self._positions:
            logger.warning(f"Position {trade_id} already tracked — skipping re-register")
            return self._positions[trade_id]

        # Resolve offsets: per-asset override → global config
        tp_offset = asset_tp_offset if asset_tp_offset is not None else self.config.tp_offset
        sl_offset = asset_sl_offset if asset_sl_offset is not None else self.config.sl_offset
        trail_offset_base = asset_trail_offset if asset_trail_offset is not None else self.config.trail_offset

        pos = TrackedPosition(
            trade_id=trade_id,
            token_id=token_id,
            side=side,
            entry_price=entry_price,
            conviction=conviction,
            size_usd=size_usd,
            registered_at=time.time(),
            market_close_ts=market_close_ts,
        )

        # ── Compute Take Profit ──
        if self.config.tp_enabled:
            tp_mult, _ = _get_conviction_multipliers(conviction)
            if not self.config.tp_conviction_scale:
                tp_mult = 1.0

            raw_tp_offset = tp_offset * tp_mult
            raw_tp_price = entry_price + raw_tp_offset

            # Validate TP nets positive after round-trip fees
            rt_fee = compute_round_trip_fee(entry_price, raw_tp_price)
            net_profit = raw_tp_offset - rt_fee

            if net_profit >= self.config.tp_min_net_profit:
                pos.tp_price = round(raw_tp_price, 4)
            else:
                # Push TP up until it meets min_net_profit
                # Solve: (tp - entry) - fee(entry) - fee(tp) >= min_net_profit
                # Iterate since fee(tp) depends on tp
                adjusted_tp = raw_tp_price
                for _ in range(20):
                    needed = self.config.tp_min_net_profit + compute_round_trip_fee(entry_price, adjusted_tp)
                    adjusted_tp = entry_price + needed
                    # Check convergence
                    rt = compute_round_trip_fee(entry_price, adjusted_tp)
                    if (adjusted_tp - entry_price) - rt >= self.config.tp_min_net_profit:
                        break

                if adjusted_tp < 1.0:
                    pos.tp_price = round(adjusted_tp, 4)
                else:
                    # TP would be above $1.00 — disable TP for this position
                    pos.tp_price = None
                    logger.debug(f"TP disabled for {trade_id}: adjusted TP >= $1.00")

        # ── Compute Stop Loss ──
        if self.config.sl_enabled:
            raw_sl = entry_price - sl_offset
            pos.sl_price = round(max(raw_sl, self.config.sl_min_price), 4)

        # ── Compute Trailing Stop params ──
        if self.config.trail_enabled:
            _, trail_mult = _get_conviction_multipliers(conviction)
            if not self.config.tp_conviction_scale:
                trail_mult = 1.0

            pos.trail_activation_price = round(entry_price + self.config.trail_activation, 4)
            pos.trail_offset = round(trail_offset_base * trail_mult, 4)
            pos.trail_active = False
            pos.trail_peak = 0.0
            pos.trail_trigger_price = 0.0

        self._positions[trade_id] = pos

        logger.info(
            f"[EARLY_EXIT] Registered {trade_id}: entry=${entry_price:.2f} "
            f"conv={conviction:.2f} | "
            f"TP={'$' + f'{pos.tp_price:.2f}' if pos.tp_price else 'off'} "
            f"SL={'$' + f'{pos.sl_price:.2f}' if pos.sl_price else 'off'} "
            f"Trail={f'activate>${pos.trail_activation_price:.2f}, offset={pos.trail_offset:.2f}' if self.config.trail_enabled else 'off'}"
        )

        return pos

    # ── Exit Evaluation ──────────────────────────────────────────

    def check_position(self, trade_id: str, current_bid: float) -> Optional[ExitSignal]:
        """
        Evaluate a single position against the current CLOB best bid.

        Returns an ExitSignal if an exit should fire, None otherwise.

        Priority order per spec:
          1. Take Profit
          2. Stop Loss
          3. Trailing Stop
        """
        pos = self._positions.get(trade_id)
        if not pos or pos.exited:
            return None

        if current_bid <= 0:
            return None

        now = time.time()

        # ── Min hold enforcement ──
        if (now - pos.registered_at) < self.config.min_hold_secs:
            return None

        time_remaining = max(0, pos.market_close_ts - now)

        # ── 1. Take Profit ──
        if pos.tp_price is not None and current_bid >= pos.tp_price:
            gross = current_bid - pos.entry_price
            fees = compute_round_trip_fee(pos.entry_price, current_bid)
            net = gross - fees
            return ExitSignal(
                trade_id=trade_id,
                exit_type=ExitType.TAKE_PROFIT,
                trigger_price=pos.tp_price,
                current_bid=current_bid,
                entry_price=pos.entry_price,
                gross_pnl_per_share=round(gross, 4),
                estimated_fees=round(fees, 4),
                net_pnl_per_share=round(net, 4),
                hold_duration_secs=round(now - pos.registered_at, 1),
                peak_price=pos.trail_peak if pos.trail_active else None,
                reason=f"TP hit: bid ${current_bid:.4f} >= target ${pos.tp_price:.4f}",
            )

        # ── 2. Stop Loss (with optional time decay) ──
        if pos.sl_price is not None:
            effective_sl = pos.sl_price

            # Time decay: tighten SL as market close approaches
            if self.config.sl_time_decay and time_remaining > 0:
                if time_remaining <= 60:
                    # Final 60s: linear tighten to sl_offset * 0.5
                    decay_factor = 0.5 + 0.5 * (time_remaining / 60.0)
                    decayed_offset = self.config.sl_offset * decay_factor
                    effective_sl = max(
                        pos.entry_price - decayed_offset,
                        self.config.sl_min_price,
                    )
                    effective_sl = round(effective_sl, 4)

            if current_bid <= effective_sl:
                gross = current_bid - pos.entry_price  # Negative
                fees = compute_round_trip_fee(pos.entry_price, current_bid)
                net = gross - fees
                return ExitSignal(
                    trade_id=trade_id,
                    exit_type=ExitType.STOP_LOSS,
                    trigger_price=effective_sl,
                    current_bid=current_bid,
                    entry_price=pos.entry_price,
                    gross_pnl_per_share=round(gross, 4),
                    estimated_fees=round(fees, 4),
                    net_pnl_per_share=round(net, 4),
                    hold_duration_secs=round(now - pos.registered_at, 1),
                    peak_price=pos.trail_peak if pos.trail_active else None,
                    reason=f"SL hit: bid ${current_bid:.4f} <= target ${effective_sl:.4f}"
                           + (f" (time-decayed from ${pos.sl_price:.4f})" if effective_sl != pos.sl_price else ""),
                )

        # ── 3. Trailing Stop ──
        if self.config.trail_enabled and pos.trail_offset > 0:
            # Activate trail once price moves far enough above entry
            if not pos.trail_active:
                if current_bid >= pos.trail_activation_price:
                    pos.trail_active = True
                    pos.trail_peak = current_bid
                    pos.trail_trigger_price = round(current_bid - pos.trail_offset, 4)
                    logger.debug(
                        f"[EARLY_EXIT] Trail activated for {trade_id}: "
                        f"peak=${current_bid:.4f}, trigger=${pos.trail_trigger_price:.4f}"
                    )

            if pos.trail_active:
                # Update peak if new high
                if current_bid > pos.trail_peak:
                    pos.trail_peak = current_bid

                    # Compute trail trigger with optional near-close tightening
                    effective_trail_offset = pos.trail_offset
                    if self.config.trail_tighten_near_close and time_remaining <= 60 and time_remaining > 0:
                        effective_trail_offset = pos.trail_offset * 0.6

                    pos.trail_trigger_price = round(pos.trail_peak - effective_trail_offset, 4)

                # Also apply near-close tightening on non-peak ticks
                elif self.config.trail_tighten_near_close and time_remaining <= 60 and time_remaining > 0:
                    effective_trail_offset = pos.trail_offset * 0.6
                    tightened_trigger = round(pos.trail_peak - effective_trail_offset, 4)
                    if tightened_trigger > pos.trail_trigger_price:
                        pos.trail_trigger_price = tightened_trigger

                # Check trail trigger
                if current_bid <= pos.trail_trigger_price:
                    gross = current_bid - pos.entry_price
                    fees = compute_round_trip_fee(pos.entry_price, current_bid)
                    net = gross - fees
                    return ExitSignal(
                        trade_id=trade_id,
                        exit_type=ExitType.TRAILING_STOP,
                        trigger_price=pos.trail_trigger_price,
                        current_bid=current_bid,
                        entry_price=pos.entry_price,
                        gross_pnl_per_share=round(gross, 4),
                        estimated_fees=round(fees, 4),
                        net_pnl_per_share=round(net, 4),
                        hold_duration_secs=round(now - pos.registered_at, 1),
                        peak_price=pos.trail_peak,
                        reason=f"Trail hit: bid ${current_bid:.4f} <= trigger ${pos.trail_trigger_price:.4f} "
                               f"(peak=${pos.trail_peak:.4f}, offset=${pos.trail_offset:.4f})",
                    )

        return None

    # ── Position Management ──────────────────────────────────────

    def mark_exited(self, trade_id: str, exit_type: ExitType, exit_price: float):
        """Mark a position as exited after successful sell execution."""
        pos = self._positions.get(trade_id)
        if not pos:
            return
        pos.exited = True
        pos.exit_type = exit_type
        pos.exit_price = exit_price
        pos.exit_timestamp = time.time()

        # Stats
        self._exit_count[exit_type.value] += 1
        net = (exit_price - pos.entry_price) - compute_round_trip_fee(pos.entry_price, exit_price)
        self._total_exit_pnl += net * pos.shares_estimate

        logger.info(
            f"[EARLY_EXIT] {exit_type.value.upper()} executed for {trade_id}: "
            f"entry=${pos.entry_price:.4f} -> exit=${exit_price:.4f} | "
            f"net/share=${net:+.4f} | held {pos.hold_duration_secs:.0f}s"
        )

    def remove_position(self, trade_id: str):
        """Remove a position from tracking (after resolution or exit)."""
        removed = self._positions.pop(trade_id, None)
        if removed:
            logger.debug(f"[EARLY_EXIT] Removed {trade_id} from tracking")

    def get_tracked(self) -> dict[str, TrackedPosition]:
        """Return all actively tracked (non-exited) positions."""
        return {tid: pos for tid, pos in self._positions.items() if not pos.exited}

    def get_all_tracked(self) -> dict[str, TrackedPosition]:
        """Return ALL tracked positions including exited ones."""
        return dict(self._positions)

    def get_status(self) -> dict:
        """Summary stats for logging/dashboard."""
        active = self.get_tracked()
        return {
            "active_positions": len(active),
            "total_tracked": len(self._positions),
            "exits": dict(self._exit_count),
            "total_exit_pnl": round(self._total_exit_pnl, 4),
            "positions": {
                tid: {
                    "entry": pos.entry_price,
                    "tp": pos.tp_price,
                    "sl": pos.sl_price,
                    "trail_active": pos.trail_active,
                    "trail_peak": pos.trail_peak if pos.trail_active else None,
                    "trail_trigger": pos.trail_trigger_price if pos.trail_active else None,
                    "hold_secs": round(pos.hold_duration_secs, 0),
                    "time_remaining": round(pos.time_remaining_secs, 0),
                }
                for tid, pos in active.items()
            },
        }
