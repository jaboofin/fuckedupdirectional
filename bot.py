"""
╔══════════════════════════════════════════════════════════════════════════╗
║  GRIDPHANTOMDEV — Autonomous Polymarket Prediction Bot                   ║
║                                                                          ║
║  Clock-synced entries · Multi-asset (BTC/ETH/SOL)                       ║
║  Live trading via py-clob-client SDK                                     ║
║  5m + 15m + 1hr UP/DOWN binary markets                                   ║
║                                                                          ║
║  v1.3 — Multi-asset trading (Phase C)                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import signal
import sys
import time
import logging
import json
import datetime
import re
from pathlib import Path

from config.settings import BotConfig, MarketDirection
from oracles.price_feed import OracleEngine
from strategies.signal_engine import StrategyEngine
from core.polymarket_client import PolymarketClient
from core.risk_manager import RiskManager
from core.trade_logger import TradeLogger
from core.early_exit import EarlyExitManager, ExitType
from core.regime_filter import RegimeFilter, RegimeState
from core.drawdown_guard import DrawdownGuard, DrawdownConfig as DDConfig, DrawdownTier
from core.position_sizer import PositionSizer, SizerConfig
from core.dashboard_server import DashboardServer, build_dashboard_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bot")


class BTCPredictionBot:
    def __init__(self, config: BotConfig, dashboard: bool = False):
        self.config = config
        self.running = False
        self.trade_logger = TradeLogger(config.logging)

        # ── Multi-asset oracle instances (Phase A) ──
        self.oracles: dict[str, OracleEngine] = {}
        for asset_cfg in config.enabled_assets():
            self.oracles[asset_cfg.symbol] = OracleEngine(config, asset_config=asset_cfg)

        # Backward compat alias
        self.oracle = self.oracles.get("BTC") or OracleEngine(config)

        # ── Per-asset strategy instances (Phase C) ──
        self.strategies: dict[str, StrategyEngine] = {}
        for asset_cfg in config.enabled_assets():
            self.strategies[asset_cfg.symbol] = StrategyEngine(config.strategy)
        self.strategy = self.strategies.get("BTC") or StrategyEngine(config.strategy)

        # ── Per-asset risk managers (Phase C) ──
        # Each asset gets its own budget slice. Drawdown guard stays global.
        self.risk_managers: dict[str, RiskManager] = {}
        for asset_cfg in config.enabled_assets():
            asset_capital = round(config.bankroll * (asset_cfg.budget_pct / 100.0), 2)
            self.risk_managers[asset_cfg.symbol] = RiskManager(config.risk, capital=asset_capital)
            logger.info(f"[{asset_cfg.symbol}] Risk manager: ${asset_capital:.2f} ({asset_cfg.budget_pct:.0f}% of ${config.bankroll:.2f})")
        self.risk_manager = self.risk_managers.get("BTC") or RiskManager(config.risk, capital=config.bankroll)

        # ── Per-asset regime filters (Phase C) ──
        self.regime_filters: dict[str, RegimeFilter] = {}
        if config.regime.enabled:
            for asset_cfg in config.enabled_assets():
                self.regime_filters[asset_cfg.symbol] = RegimeFilter(config.regime)
            logger.info(f"[REGIME] Per-asset regime filters initialized: {list(self.regime_filters.keys())}")
        self.regime_filter = self.regime_filters.get("BTC")

        self.polymarket = PolymarketClient(config)
        self.dashboard = DashboardServer() if dashboard else None
        self._cycle_count = 0
        self._start_time = 0
        self._traded_this_window = False
        self._last_consensus = None
        self._last_anchor = None
        self._last_decision = None
        self._last_live_bankroll_sync = 0.0
        self._last_live_bankroll_value = None
        self._directional_interval_mins = int(config.polymarket.market_interval_minutes or 15)
        self._last_interval_refresh = 0.0
        self._last_regime = RegimeState.UNKNOWN

        # ── Per-asset decision/regime tracking for dashboard ──
        self._last_decisions: dict = {}   # symbol → StrategyDecision
        self._last_regimes: dict = {}     # symbol → RegimeState
        self._last_signals: dict = {}     # symbol → {signal_name: {direction, strength, ...}}

        # ── Unified trade-to-asset routing (Phase C) ──
        # Maps trade_id → (asset_symbol, timeframe) for PnL routing
        self._trade_asset_map: dict[str, tuple[str, str]] = {}
        # Maps trade_id → direction ("up"/"down") for correlation guard
        self._trade_direction_map: dict[str, str] = {}
        # Lock to prevent multiple loops from double-counting resolutions
        self._resolution_lock = asyncio.Lock()

        # ── Legacy 5m/1hr state (kept for backward compat with existing loops) ──
        self._5m_traded_this_window = False
        self._5m_cycle_count = 0
        self._5m_last_anchor_price = None
        self._5m_trade_ids: set = set()

        self._1h_traded_this_window = False
        self._1h_cycle_count = 0
        self._1h_last_anchor_price = None
        self._1h_trade_ids: set = set()

        # Wire 5m budget limits into BTC risk manager (legacy compat)
        if hasattr(config, 'active_5m') and "BTC" in self.risk_managers:
            rm = self.risk_managers["BTC"]
            rm.fivem_budget_pct = config.active_5m.budget_pct
            rm.fivem_max_daily_trades = config.active_5m.max_daily_trades
            rm.fivem_max_trade_usd = config.active_5m.max_trade_size_usd
            rm.fivem_max_daily_loss_pct = config.active_5m.max_daily_loss_pct
            rm.fivem_max_consecutive_losses = config.active_5m.max_consecutive_losses
            rm.fivem_cooldown_mins = config.active_5m.loss_streak_cooldown_mins

        # ── Early Exit (TP/SL/Trail) ──
        if hasattr(config, 'early_exit') and config.early_exit.enabled:
            self.early_exit = EarlyExitManager(config.early_exit)
            logger.info(
                f"Early exit system initialized "
                f"({'DRY RUN' if config.early_exit.dry_run else 'LIVE'})"
            )
        else:
            self.early_exit = None

        # ── Drawdown Guard (TARB M3) — GLOBAL across all assets ──
        self._acknowledge_drawdown = getattr(config, '_acknowledge_drawdown', False)
        if config.drawdown.enabled:
            self.drawdown_guard = DrawdownGuard(
                config=DDConfig(
                    yellow_threshold=config.drawdown.yellow_threshold,
                    orange_threshold=config.drawdown.orange_threshold,
                    red_threshold=config.drawdown.red_threshold,
                    yellow_size_mult=config.drawdown.yellow_size_mult,
                    orange_size_mult=config.drawdown.orange_size_mult,
                    orange_min_confidence=config.drawdown.orange_min_confidence,
                    yellow_recovery=config.drawdown.yellow_recovery,
                    orange_recovery=config.drawdown.orange_recovery,
                    hwm_state_path=config.drawdown.hwm_state_path,
                    halted_file_path=config.drawdown.halted_file_path,
                ),
                initial_bankroll=config.bankroll,
                acknowledge_drawdown=self._acknowledge_drawdown,
            )
            if self.drawdown_guard.is_halted():
                logger.critical(
                    "[RED] DRAWDOWN GUARD: Trading is HALTED. "
                    "Use --acknowledge-drawdown to resume."
                )
            else:
                status = self.drawdown_guard.get_status()
                logger.info(
                    f"[DRAWDOWN] Guard initialized -- tier={status['tier']}, "
                    f"HWM=${status['high_water_mark']:.2f}, "
                    f"drawdown={status['drawdown_pct']:.1f}%"
                )
        else:
            self.drawdown_guard = None

        # ── Position Sizer (TARB M4) ──
        if config.sizer.enabled:
            self.position_sizer = PositionSizer(SizerConfig(
                per_market_budget_pct=config.sizer.per_market_budget_pct,
                small_bankroll_budget_pct=config.sizer.small_bankroll_budget_pct,
                small_bankroll_cap=config.sizer.small_bankroll_cap,
                mid_bankroll_cap=config.sizer.mid_bankroll_cap,
                min_order_usd=config.sizer.min_order_usd,
                max_order_usd=config.sizer.max_order_usd,
            ))
            logger.info("[SIZER] Position sizer initialized (confidence-scaled + bankroll-tiered)")
        else:
            self.position_sizer = None

    # ── Trading Cycle ───────────────────────────────────────────

    def _total_capital(self) -> float:
        """Sum capital across all per-asset risk managers."""
        return sum(rm.capital for rm in self.risk_managers.values())

    async def _broadcast_log(self, tag: str, msg: str, cls: str = "info"):
        """Send a log event to the dashboard for the live log panel."""
        if self.dashboard and self.dashboard.is_running:
            try:
                await self.dashboard.broadcast({
                    "type": "bot_log", "tag": tag, "msg": msg, "cls": cls,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    def _route_pnl(self, trade_id: str, pnl: float):
        """Route P&L to the correct per-asset risk manager based on trade-asset map."""
        mapping = self._trade_asset_map.get(trade_id)
        if mapping:
            asset_symbol, tf = mapping
            rm = self.risk_managers.get(asset_symbol)
            if rm:
                if tf == "5m":
                    rm.record_5m_trade(0, pnl=pnl)
                else:
                    rm.record_trade(pnl)
                self._trade_asset_map.pop(trade_id, None)
                self._trade_direction_map.pop(trade_id, None)
                return
        # Legacy fallback: check old trade ID sets
        if trade_id in self._1h_trade_ids:
            self.risk_manager.record_trade(pnl)
            self._1h_trade_ids.discard(trade_id)
        elif trade_id in self._5m_trade_ids:
            self.risk_manager.record_5m_trade(0, pnl=pnl)
            self._5m_trade_ids.discard(trade_id)
        else:
            self.risk_manager.record_trade(pnl)
        self._trade_direction_map.pop(trade_id, None)

    def _correlation_guard_multiplier(self, asset_symbol: str, direction: str) -> float:
        """
        Correlation Guard (Phase E1): reduce position size when multiple assets
        have open positions in the same direction.

        BTC, ETH, SOL are correlated — when BTC dumps, all three dump.
        If all 3 assets have positions in the same direction, a correlated
        move causes triple-sized drawdowns.

        Rules:
          - All 3 assets open in same direction → 0.50x new position size
          - 2 of 3 assets open in same direction → 0.75x
          - Otherwise → 1.0x (no reduction)

        Only counts positions from OTHER assets (not the one being sized).
        """
        # Collect the direction of each asset's open positions (excluding current asset)
        asset_directions: dict[str, str] = {}
        for trade_id, (trade_asset, _tf) in self._trade_asset_map.items():
            if trade_asset == asset_symbol:
                continue  # Don't count the asset we're sizing for
            trade_dir = self._trade_direction_map.get(trade_id)
            if trade_dir:
                # Use most recent direction per asset (last wins if multiple)
                asset_directions[trade_asset] = trade_dir

        if not asset_directions:
            return 1.0  # No other assets have open positions

        # Count how many other assets share the proposed direction
        same_dir_count = sum(1 for d in asset_directions.values() if d == direction)
        other_asset_count = len(asset_directions)

        if same_dir_count >= 2:
            # All open assets + this new one = 3+ in same direction
            logger.info(
                f"[CORR GUARD] {asset_symbol} {direction.upper()}: "
                f"{same_dir_count} other assets also {direction} → 0.50x size"
            )
            return 0.50
        elif same_dir_count == 1 and other_asset_count >= 1:
            # 2 assets in same direction (this + 1 other)
            logger.info(
                f"[CORR GUARD] {asset_symbol} {direction.upper()}: "
                f"1 other asset also {direction} → 0.75x size"
            )
            return 0.75
        else:
            return 1.0

    # ── Global Position Limit (Phase E3) ──

    MAX_DEPLOYED_PCT = 40.0  # Never more than 40% of total bankroll in open positions

    def _check_global_position_limit(self, proposed_size: float) -> tuple[bool, float]:
        """
        Global Position Limit: cap total capital deployed across all assets.
        Prevents over-exposure when many signals fire simultaneously
        (e.g. at :00 when 9 loops could all fire).

        Returns (allowed, max_allowed_size):
          - allowed=True, max_allowed_size=proposed_size if under limit
          - allowed=True, max_allowed_size=reduced if partially available
          - allowed=False, 0 if fully exhausted
        """
        total_bankroll = self._total_capital()
        if total_bankroll <= 0:
            return False, 0

        max_deployed = total_bankroll * (self.MAX_DEPLOYED_PCT / 100.0)

        # Sum all currently open (unresolved) positions
        current_deployed = sum(
            r.size_usd for r in self.polymarket._trade_records
            if r.outcome is None
        )

        remaining = max_deployed - current_deployed
        if remaining <= 0:
            logger.info(
                f"[POS LIMIT] BLOCKED — ${current_deployed:.2f} deployed "
                f"(max ${max_deployed:.2f} = {self.MAX_DEPLOYED_PCT:.0f}% of ${total_bankroll:.2f})"
            )
            return False, 0

        if proposed_size > remaining:
            logger.info(
                f"[POS LIMIT] Capped ${proposed_size:.2f} → ${remaining:.2f} "
                f"(deployed ${current_deployed:.2f} / max ${max_deployed:.2f})"
            )
            return True, round(remaining, 2)

        return True, proposed_size

    async def _sync_live_bankroll_if_enabled(self, force: bool = False):
        if not self.config.polymarket.sync_live_bankroll:
            return

        now = time.time()
        poll_secs = max(5, int(self.config.polymarket.live_bankroll_poll_secs))
        if not force and (now - self._last_live_bankroll_sync) < poll_secs:
            return

        live_balance = await self.polymarket.get_available_balance_usd()
        self._last_live_bankroll_sync = now

        if live_balance is None:
            return

        self._last_live_bankroll_value = round(float(live_balance), 2)
        self.risk_manager.capital = self._last_live_bankroll_value
        logger.info(f"Synced live bankroll: ${self._last_live_bankroll_value:.2f}")

    # ── Unified Multi-Asset Trading Cycle (Phase C) ──────────────

    async def _asset_trading_cycle(self, asset_symbol: str, timeframe: str,
                                     window_minutes: int, strategy_delay: int,
                                     indicator_overrides: dict, max_trade_usd: float):
        """
        Generic trading cycle for any asset + timeframe combination.
        Spawned per (asset, timeframe) pair by the unified trading loops.
        """
        tag = f"[{asset_symbol}/{timeframe}]"
        oracle = self.oracles.get(asset_symbol)
        strategy = self.strategies.get(asset_symbol)
        risk_mgr = self.risk_managers.get(asset_symbol)
        regime_flt = self.regime_filters.get(asset_symbol)
        asset_cfg = self.config.get_asset(asset_symbol)

        if not oracle or not strategy or not risk_mgr or not asset_cfg:
            logger.error(f"{tag} Missing oracle/strategy/risk for {asset_symbol}")
            return

        try:
            # 1. Capture window anchor
            anchor = await oracle.capture_window_open(window_minutes=window_minutes)
            open_price = anchor.open_price if anchor else None

            # 2. Strategy delay
            if strategy_delay > 0 and open_price:
                logger.info(f"📌 {tag} Anchor: ${open_price:,.2f} — waiting {strategy_delay}s...")
                await asyncio.sleep(strategy_delay)

            # 3. Fresh price + candles
            consensus = await oracle.get_price()
            if not consensus or not consensus.price:
                logger.warning(f"{tag} No oracle price — skipping")
                return

            candle_interval = timeframe if timeframe != "1h" else "1h"
            candles = await oracle.get_candles(candle_interval, limit=100)
            if len(candles) < 30:
                logger.warning(f"{tag} Only {len(candles)} candles — skipping")
                return

            # 3b. Regime filter
            if regime_flt:
                regime_flt.update_baseline_from_candles(candles)
                regime = regime_flt.classify_from_candles(candles)
                self._last_regime = regime
                self._last_regimes[asset_symbol] = regime
                if regime == RegimeState.CHOPPY:
                    logger.info(f"{tag} HOLD — regime=CHOPPY")
                    await self._broadcast_log("REGIME", f"{asset_symbol} CHOPPY — skipping", "warn")
                    return

            # Fee estimate
            market_fee_pct = None
            try:
                if self.polymarket._active_markets:
                    first_market = next(iter(self.polymarket._active_markets.values()), None)
                    if first_market:
                        token_id = first_market.token_id_up
                        fetched = await self.polymarket.get_fee_pct_for_price(token_id, first_market.price_up)
                        if fetched is not None:
                            market_fee_pct = fetched
            except Exception:
                pass

            # 4. Strategy — merge per-asset overrides with timeframe overrides
            full_overrides = {
                "dead_zone_pct": asset_cfg.dead_zone_pct,
                "confidence_threshold": asset_cfg.confidence_threshold,
                "max_volatility_pct": asset_cfg.max_volatility_pct,
                **indicator_overrides,
            }

            decision = strategy.analyze(candles, consensus.price, open_price=open_price,
                                         fee_pct=market_fee_pct, timeframe_overrides=full_overrides)

            # Store per-asset decision + signals for dashboard
            self._last_decisions[asset_symbol] = decision
            self._last_signals[asset_symbol] = {}
            for sig in decision.signals:
                self._last_signals[asset_symbol][sig.name] = {
                    "direction": sig.direction.value,
                    "strength": round(sig.strength, 3),
                    "raw_value": round(sig.raw_value, 4),
                }

            # Update top-level dashboard state (BTC priority)
            if asset_symbol == "BTC":
                self._last_consensus = consensus
                self._last_anchor = anchor
                self._last_decision = decision
            elif self._last_decision is None:
                self._last_consensus = consensus
                self._last_anchor = anchor
                self._last_decision = decision

            if not decision.should_trade:
                logger.info(f"{tag} HOLD — {decision.reason}")
                await self._broadcast_log("HOLD", f"{asset_symbol}/{timeframe} {decision.direction.value.upper()} conf={decision.confidence:.3f}", "hold")
                return

            # Regime UNKNOWN override
            if regime_flt and self._last_regime == RegimeState.UNKNOWN:
                if decision.confidence < self.config.regime.regime_override_conviction:
                    logger.info(f"{tag} HOLD — regime=UNKNOWN, confidence {decision.confidence:.3f} < override")
                    return

            # Drawdown Guard (global)
            if self.drawdown_guard:
                if self.drawdown_guard.is_halted():
                    logger.info(f"{tag} HALTED — drawdown RED tier")
                    return
                min_conf = self.drawdown_guard.get_min_confidence()
                if min_conf > 0 and decision.confidence < min_conf:
                    logger.info(f"{tag} BLOCKED — confidence {decision.confidence:.2f} < drawdown floor {min_conf}")
                    return

            # Risk check (per-asset)
            can_trade, reason = risk_mgr.can_trade()
            if not can_trade:
                logger.info(f"{tag} BLOCKED — {reason}")
                return

            # Global Position Limit (Phase E3) — check before discovery to save API calls
            allowed, _ = self._check_global_position_limit(1.0)  # Quick check with min size
            if not allowed:
                logger.info(f"{tag} BLOCKED — global position limit reached")
                return

            # 5. Discover + filter to current window AND correct timeframe
            markets = await self.polymarket.discover_markets(asset_config=asset_cfg)
            tradeable = [m for m in markets if m.is_tradeable and m.liquidity >= self.config.polymarket.min_liquidity_usd]
            # Filter to this loop's specific timeframe (e.g. only 1h markets for the 1h loop)
            tradeable = [m for m in tradeable if f"-{timeframe}-" in m.slug]
            tradeable = self.polymarket.filter_current_window(tradeable, window_minutes)
            if not tradeable:
                logger.info(f"{tag} No markets for current {timeframe} window")
                return

            market = max(tradeable, key=lambda m: m.liquidity)

            # 6. Size + execute
            if self.position_sizer:
                dd_mult = self.drawdown_guard.get_size_multiplier() if self.drawdown_guard else 1.0
                size = self.position_sizer.compute_size(
                    confidence=decision.confidence,
                    bankroll=risk_mgr.capital,
                    drawdown_mult=dd_mult,
                )
            else:
                size = risk_mgr.calculate_position_size(decision.confidence)
                if self.drawdown_guard:
                    size = round(size * self.drawdown_guard.get_size_multiplier(), 2)
            size = min(size, max_trade_usd)
            if size <= 0:
                return

            direction = decision.direction.value

            # ── Global Position Limit (Phase E3) — cap to remaining room ──
            pos_allowed, pos_max = self._check_global_position_limit(size)
            if not pos_allowed:
                return
            size = pos_max

            # ── Correlation Guard (Phase E1) ──
            corr_mult = self._correlation_guard_multiplier(asset_symbol, direction)
            if corr_mult < 1.0:
                size = round(size * corr_mult, 2)
                if size <= 0:
                    return

            trade = await self.polymarket.place_order(
                market=market, direction=direction, size_usd=size,
                oracle_price=consensus.price, confidence=decision.confidence,
            )

            if trade:
                # Register in trade-asset map for PnL routing + direction for correlation guard
                self._trade_asset_map[trade.trade_id] = (asset_symbol, timeframe)
                self._trade_direction_map[trade.trade_id] = direction

                self.trade_logger.log_trade({
                    "type": f"directional_{asset_symbol.lower()}_{timeframe}",
                    "asset": asset_symbol,
                    "timeframe": timeframe,
                    "trade_id": trade.trade_id, "direction": trade.direction,
                    "size_usd": trade.size_usd, "confidence": trade.confidence,
                    "oracle_price": trade.oracle_price_at_entry,
                    "order_id": trade.order_id,
                    "market": market.question[:80],
                })

                # Register with early exit system (per-asset TP/SL/trail offsets)
                if self.early_exit:
                    try:
                        token_id = market.token_id_up if direction == "up" else market.token_id_down
                        close_ts = self._parse_market_close_ts(market)
                        self.early_exit.register_position(
                            trade_id=trade.trade_id,
                            token_id=token_id,
                            entry_price=trade.entry_price,
                            conviction=decision.confidence,
                            size_usd=trade.size_usd,
                            market_close_ts=close_ts,
                            # Per-asset early exit overrides (Phase E4)
                            asset_tp_offset=asset_cfg.tp_offset,
                            asset_sl_offset=asset_cfg.sl_offset,
                            asset_trail_offset=asset_cfg.trail_offset,
                        )
                    except Exception as e:
                        logger.warning(f"{tag} Early exit registration failed: {e}")

                logger.info(
                    f"{'🕐' if timeframe == '1h' else '⏱️' if timeframe == '5m' else '⏰'} {tag} "
                    f"{direction.upper()} | ${size:.2f} @ conf={decision.confidence:.2f} | "
                    f"${consensus.price:,.2f}"
                )

                # Broadcast trade opened to dashboard
                if self.dashboard and self.dashboard.is_running:
                    try:
                        await self.dashboard.broadcast({
                            "type": "trade_notification",
                            "action": "opened",
                            "asset": asset_symbol,
                            "engine": timeframe,
                            "direction": direction,
                            "size_usd": trade.size_usd,
                            "entry_price": trade.entry_price,
                            "confidence": decision.confidence,
                            "trade_id": trade.trade_id,
                            "timestamp": time.time(),
                        })
                    except Exception:
                        pass

                await self._refresh_dashboard()
                await self._broadcast_log("TRADE", f"{asset_symbol}/{timeframe} {direction.upper()} ${trade.size_usd:.2f} @ {trade.entry_price:.4f}", "trade")

            # 7. Resolution check (locked to prevent double-counting across loops)
            async with self._resolution_lock:
                resolved = await self.polymarket.check_resolutions()
                for r in resolved:
                    # Look up asset for this trade
                    r_asset = "BTC"
                    r_mapping = self._trade_asset_map.get(r.trade_id)
                    if r_mapping:
                        r_asset = r_mapping[0]
                    self._route_pnl(r.trade_id, r.pnl)
                    self.trade_logger.log_resolution({"trade_id": r.trade_id, "outcome": r.outcome, "pnl": r.pnl})
                    if self.early_exit:
                        self.early_exit.remove_position(r.trade_id)
                    # Broadcast resolution to dashboard
                    if self.dashboard and self.dashboard.is_running:
                        try:
                            await self.dashboard.broadcast({
                                "type": "trade_notification",
                                "action": "resolved",
                                "asset": r_asset,
                                "trade_id": r.trade_id,
                                "outcome": r.outcome,
                                "pnl": r.pnl,
                                "timestamp": time.time(),
                            })
                        except Exception:
                            pass
                if self.drawdown_guard and resolved:
                    self.drawdown_guard.update_bankroll(self._total_capital())

        except Exception as e:
            logger.error(f"{tag} Cycle error: {e}", exc_info=True)
        finally:
            await self._refresh_dashboard()

    async def _trading_cycle(self):
        self._cycle_count += 1

        try:
            # ─────────────────────────────────────────────────────
            # PHASE 1: Capture window anchor (at the boundary)
            # ─────────────────────────────────────────────────────
            anchor = await self.oracle.capture_window_open(window_minutes=self._directional_interval_mins)
            open_price = anchor.open_price if anchor else None
            self._last_anchor = anchor

            # ─────────────────────────────────────────────────────
            # PHASE 2: Wait for BTC to drift (strategy_delay_secs)
            # ─────────────────────────────────────────────────────
            delay = getattr(self.config, 'strategy_delay_secs', 0)
            if delay > 0 and open_price:
                logger.info(f"📌 Anchor: ${open_price:,.2f} — waiting {delay}s for price drift...")
                await asyncio.sleep(delay)

            # ─────────────────────────────────────────────────────
            # PHASE 3: Fresh price + strategy (after the delay)
            # ─────────────────────────────────────────────────────
            consensus = await self.oracle.get_price()
            self._last_consensus = consensus
            self.trade_logger.log_oracle({
                "price": consensus.price, "chainlink": consensus.chainlink_price,
                "sources": consensus.sources, "spread_pct": consensus.spread_pct,
                "window_open": open_price,
            })

            # 3. Candles
            candles = await self.oracle.get_candles(f"{self._directional_interval_mins}m", limit=100)
            if len(candles) < 30:
                logger.warning(f"Only {len(candles)} candles — skipping")
                return

            # 3b. Regime filter — block entry during CHOPPY markets
            if self.regime_filter:
                self.regime_filter.update_baseline_from_candles(candles)
                regime = self.regime_filter.classify_from_candles(candles)
                self._last_regime = regime
                if regime == RegimeState.CHOPPY:
                    logger.info(f"Cycle {self._cycle_count}: HOLD — regime=CHOPPY")
                    return

            # 4. Strategy (anchored to window open price)
            # ── Fetch dynamic fee estimate for edge filtering ──
            market_fee_pct = None
            try:
                market_fee_pct = self.polymarket._fee_fallback_pct * 4.0 * 0.5 * 0.5
                if self.polymarket._active_markets:
                    first_market = next(iter(self.polymarket._active_markets.values()), None)
                    if first_market:
                        token_id = first_market.token_id_up
                        fetched = await self.polymarket.get_fee_pct_for_price(token_id, first_market.price_up)
                        if fetched is not None:
                            market_fee_pct = fetched
            except Exception:
                pass  # Fall back to None → signal_engine uses 1.56% default

            decision = self.strategy.analyze(candles, consensus.price, open_price=open_price, fee_pct=market_fee_pct)
            self._last_decision = decision
            self.trade_logger.log_strategy({
                "direction": decision.direction.value,
                "confidence": decision.confidence,
                "should_trade": decision.should_trade,
                "drift_pct": decision.drift_pct,
                "open_price": open_price,
                "fee_pct": market_fee_pct,
                "signals": {s.name: {"dir": s.direction.value, "str": round(s.strength, 3)} for s in decision.signals},
                "btc_price": consensus.price,
            })

            if not decision.should_trade:
                logger.info(f"Cycle {self._cycle_count}: HOLD — {decision.reason}")
                return

            # 4b. Regime UNKNOWN override — need high conviction to trade
            if self.regime_filter and self._last_regime == RegimeState.UNKNOWN:
                if decision.confidence < self.config.regime.regime_override_conviction:
                    logger.info(
                        f"Cycle {self._cycle_count}: HOLD — regime=UNKNOWN, "
                        f"confidence {decision.confidence:.3f} < override {self.config.regime.regime_override_conviction}"
                    )
                    return
                else:
                    logger.info(
                        f"Cycle {self._cycle_count}: regime=UNKNOWN but confidence "
                        f"{decision.confidence:.3f} >= override — proceeding"
                    )

            # 5. Live bankroll sync + risk
            await self._sync_live_bankroll_if_enabled()

            # 5.1 Drawdown Guard (M3) — halt check + confidence floor
            if self.drawdown_guard:
                if self.drawdown_guard.is_halted():
                    logger.info(f"Cycle {self._cycle_count}: HALTED — drawdown RED tier")
                    return
                min_conf = self.drawdown_guard.get_min_confidence()
                if min_conf > 0 and decision.confidence < min_conf:
                    logger.info(
                        f"Cycle {self._cycle_count}: BLOCKED — confidence {decision.confidence:.2f} "
                        f"< drawdown floor {min_conf}"
                    )
                    return

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.info(f"Cycle {self._cycle_count}: BLOCKED — {reason}")
                return

            # 6. Markets — discover then filter to CURRENT window only
            markets = await self.polymarket.discover_markets()
            tradeable = [m for m in markets if m.is_tradeable and m.liquidity >= self.config.polymarket.min_liquidity_usd]

            if not markets:
                logger.info(f"Cycle {self._cycle_count}: No directional markets discovered")
                return
            if not tradeable:
                logger.info(
                    f"Cycle {self._cycle_count}: {len(markets)} markets discovered but none met liquidity "
                    f"threshold ${self.config.polymarket.min_liquidity_usd:.2f}"
                )
                return

            # Filter to current window — prevents trading future windows
            tradeable = self.polymarket.filter_current_window(tradeable, self._directional_interval_mins)
            if not tradeable:
                logger.info(f"Cycle {self._cycle_count}: No markets for current {self._directional_interval_mins}m window")
                return

            market = max(tradeable, key=lambda m: m.liquidity)

            # 7. Execute directional trade
            direction = decision.direction.value
            # Position sizing: M4 sizer if available, else flat Kelly fallback
            if self.position_sizer:
                dd_mult = self.drawdown_guard.get_size_multiplier() if self.drawdown_guard else 1.0
                size = self.position_sizer.compute_size(
                    confidence=decision.confidence,
                    bankroll=self.risk_manager.capital,
                    drawdown_mult=dd_mult,
                )
            else:
                size = self.risk_manager.calculate_position_size(decision.confidence)
                if self.drawdown_guard:
                    size = round(size * self.drawdown_guard.get_size_multiplier(), 2)
            if size <= 0:
                return

            trade = await self.polymarket.place_order(
                market=market, direction=direction, size_usd=size,
                oracle_price=consensus.price, confidence=decision.confidence,
            )

            if trade:
                self.trade_logger.log_trade({
                    "type": "directional_15m",
                    "trade_id": trade.trade_id, "direction": trade.direction,
                    "size_usd": trade.size_usd, "confidence": trade.confidence,
                    "oracle_price": trade.oracle_price_at_entry,
                    "order_id": trade.order_id,
                    "market": market.question[:80],
                })

                # ── Register with early exit system ──
                if self.early_exit:
                    try:
                        token_id = market.token_id_up if direction == "up" else market.token_id_down
                        close_ts = self._parse_market_close_ts(market)
                        self.early_exit.register_position(
                            trade_id=trade.trade_id,
                            token_id=token_id,
                            entry_price=trade.entry_price,
                            conviction=decision.confidence,
                            size_usd=trade.size_usd,
                            market_close_ts=close_ts,
                        )
                    except Exception as e:
                        logger.warning(f"Early exit registration failed: {e}")

            # 8. Resolutions — route 5m/1hr PnL to separate trackers
            resolved = await self.polymarket.check_resolutions()
            for r in resolved:
                is_1h = r.trade_id in self._1h_trade_ids
                is_5m = r.trade_id in self._5m_trade_ids
                if is_1h:
                    self.risk_manager.record_trade(r.pnl)
                    self._1h_trade_ids.discard(r.trade_id)
                elif is_5m:
                    self.risk_manager.record_5m_trade(0, pnl=r.pnl)
                    self._5m_trade_ids.discard(r.trade_id)
                else:
                    self.risk_manager.record_trade(r.pnl)
                self.trade_logger.log_resolution({"trade_id": r.trade_id, "outcome": r.outcome, "pnl": r.pnl})
                # Clean up early exit tracking (resolved = no longer needs monitoring)
                if self.early_exit:
                    self.early_exit.remove_position(r.trade_id)

            # 8.5 Update Drawdown Guard with current bankroll
            if self.drawdown_guard and resolved:
                self.drawdown_guard.update_bankroll(self._total_capital())

            # 9. Status
            stats = self.polymarket.get_stats()
            self.trade_logger.save_performance({
                "cycle": self._cycle_count, "btc_price": consensus.price,
                **stats, **self.risk_manager.get_status(),
            })

            logger.info(
                f"Cycle {self._cycle_count} | BTC=${consensus.price:,.2f} | "
                f"{direction.upper()} conf={decision.confidence:.2f} | "
                f"W/R={stats.get('win_rate', 0):.0f}%"
            )

        except Exception as e:
            logger.error(f"Cycle {self._cycle_count} error: {e}", exc_info=True)

        finally:
            await self._refresh_dashboard()

    # ── 5-Minute Parallel Loop ───────────────────────────────────

    def _next_5m_boundary(self) -> float:
        """Next 5-minute boundary timestamp."""
        now = time.time()
        dt = datetime.datetime.fromtimestamp(now)
        next_min = ((dt.minute // 5) + 1) * 5
        if next_min >= 60:
            b = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        else:
            b = dt.replace(minute=next_min, second=0, microsecond=0)
        return b.timestamp()

    def _is_in_5m_entry_window(self) -> bool:
        """Check if we're in the entry window for a 5m boundary."""
        cfg = self.config.active_5m
        secs_until = self._next_5m_boundary() - cfg.entry_lead_secs - time.time()
        return -cfg.entry_window_secs <= secs_until <= 0

    async def _5m_trading_cycle(self):
        """Execute a single 5m directional trade — mirrors _trading_cycle but with 5m params."""
        self._5m_cycle_count += 1
        cfg = self.config.active_5m

        try:
            # 1. Capture 5m anchor
            anchor = await self.oracle.capture_window_open(window_minutes=5)
            open_price = anchor.open_price if anchor else None
            self._5m_last_anchor_price = open_price

            # 2. Shorter strategy delay for 5m
            delay = cfg.strategy_delay_secs
            if delay > 0 and open_price:
                logger.info(f"📌 [5m] Anchor: ${open_price:,.2f} — waiting {delay}s...")
                await asyncio.sleep(delay)

            # 3. Fresh price + strategy
            consensus = await self.oracle.get_price()
            if not consensus or not consensus.price:
                logger.warning("[5m] No oracle price — skipping")
                return

            candles = await self.oracle.get_candles("5m", limit=100)
            if len(candles) < 30:
                logger.warning(f"[5m] Only {len(candles)} candles — skipping")
                return

            # Regime filter — block CHOPPY
            if self.regime_filter:
                self.regime_filter.update_baseline_from_candles(candles)
                regime = self.regime_filter.classify_from_candles(candles)
                self._last_regime = regime
                if regime == RegimeState.CHOPPY:
                    logger.info(f"[5m] Cycle {self._5m_cycle_count}: HOLD — regime=CHOPPY")
                    return

            # Fee estimate
            market_fee_pct = None
            try:
                if self.polymarket._active_markets:
                    first_market = next(iter(self.polymarket._active_markets.values()), None)
                    if first_market:
                        token_id = first_market.token_id_up
                        fetched = await self.polymarket.get_fee_pct_for_price(token_id, first_market.price_up)
                        if fetched is not None:
                            market_fee_pct = fetched
            except Exception:
                pass

            # Build 5m-tuned indicator overrides from Active5mConfig
            _5m_overrides = {
                "rsi_period": cfg.rsi_period,
                "macd_fast": cfg.macd_fast,
                "macd_slow": cfg.macd_slow,
                "macd_signal": cfg.macd_signal,
                "ema_fast": cfg.ema_fast,
                "ema_slow": cfg.ema_slow,
                "momentum_lookback": cfg.momentum_lookback,
                "weight_price_vs_open": cfg.weight_price_vs_open,
            }

            decision = self.strategy.analyze(candles, consensus.price, open_price=open_price,
                                             fee_pct=market_fee_pct, timeframe_overrides=_5m_overrides)

            if not decision.should_trade:
                logger.info(f"[5m] Cycle {self._5m_cycle_count}: HOLD — {decision.reason}")
                return

            # Regime UNKNOWN override for 5m
            if self.regime_filter and self._last_regime == RegimeState.UNKNOWN:
                if decision.confidence < self.config.regime.regime_override_conviction:
                    logger.info(
                        f"[5m] Cycle {self._5m_cycle_count}: HOLD — regime=UNKNOWN, "
                        f"confidence {decision.confidence:.3f} < override"
                    )
                    return

            # 4.0 Drawdown Guard (M3) — same check as 15m
            if self.drawdown_guard:
                if self.drawdown_guard.is_halted():
                    logger.info(f"[5m] Cycle {self._5m_cycle_count}: HALTED — drawdown RED tier")
                    return
                min_conf = self.drawdown_guard.get_min_confidence()
                if min_conf > 0 and decision.confidence < min_conf:
                    logger.info(
                        f"[5m] Cycle {self._5m_cycle_count}: BLOCKED — confidence "
                        f"{decision.confidence:.2f} < drawdown floor {min_conf}"
                    )
                    return

            # 4. Risk check (separate 5m budget)
            can_trade, reason = self.risk_manager.can_trade_5m()
            if not can_trade:
                logger.info(f"[5m] Cycle {self._5m_cycle_count}: BLOCKED — {reason}")
                return

            # 5. Discover + filter to current 5m window
            markets = await self.polymarket.discover_markets()
            tradeable = [m for m in markets if m.is_tradeable and m.liquidity >= self.config.polymarket.min_liquidity_usd]
            tradeable = self.polymarket.filter_current_window(tradeable, 5)
            if not tradeable:
                logger.info(f"[5m] Cycle {self._5m_cycle_count}: No markets for current 5m window")
                return

            market = max(tradeable, key=lambda m: m.liquidity)

            # 6. Size + execute
            if self.position_sizer:
                dd_mult = self.drawdown_guard.get_size_multiplier() if self.drawdown_guard else 1.0
                size = self.position_sizer.compute_size(
                    confidence=decision.confidence,
                    bankroll=self.risk_manager.capital,
                    drawdown_mult=dd_mult,
                )
            else:
                size = self.risk_manager.calculate_5m_size(decision.confidence)
                if self.drawdown_guard:
                    size = round(size * self.drawdown_guard.get_size_multiplier(), 2)
            if size <= 0:
                return

            direction = decision.direction.value
            trade = await self.polymarket.place_order(
                market=market, direction=direction, size_usd=size,
                oracle_price=consensus.price, confidence=decision.confidence,
            )

            if trade:
                self._5m_trade_ids.add(trade.trade_id)
                self.risk_manager.record_5m_trade(size)
                self.trade_logger.log_trade({
                    "type": "directional_5m",
                    "trade_id": trade.trade_id, "direction": trade.direction,
                    "size_usd": trade.size_usd, "confidence": trade.confidence,
                    "oracle_price": trade.oracle_price_at_entry,
                    "order_id": trade.order_id,
                    "market": market.question[:80],
                })

                # ── Register with early exit system ──
                if self.early_exit:
                    try:
                        token_id = market.token_id_up if direction == "up" else market.token_id_down
                        close_ts = self._parse_market_close_ts(market)
                        self.early_exit.register_position(
                            trade_id=trade.trade_id,
                            token_id=token_id,
                            entry_price=trade.entry_price,
                            conviction=decision.confidence,
                            size_usd=trade.size_usd,
                            market_close_ts=close_ts,
                        )
                    except Exception as e:
                        logger.warning(f"[5m] Early exit registration failed: {e}")

                logger.info(
                    f"⏱️ [5m] {direction.upper()} | ${size:.2f} @ conf={decision.confidence:.2f} | "
                    f"BTC=${consensus.price:,.2f}"
                )
                await self._refresh_dashboard()

        except Exception as e:
            logger.error(f"[5m] Cycle {self._5m_cycle_count} error: {e}", exc_info=True)

    async def _5m_loop(self):
        """Independent 5m trading loop — runs as async task alongside 15m loop."""
        logger.info("⏱️ [5m] Parallel trading loop started")

        while self.running:
            try:
                # Check resolutions every tick — 5m trades resolve fast
                try:
                    resolved = await self.polymarket.check_resolutions()
                    for r in resolved:
                        is_5m = r.trade_id in self._5m_trade_ids
                        if is_5m:
                            self.risk_manager.record_5m_trade(0, pnl=r.pnl)
                            self._5m_trade_ids.discard(r.trade_id)
                        else:
                            self.risk_manager.record_trade(r.pnl)
                        self.trade_logger.log_resolution({"trade_id": r.trade_id, "outcome": r.outcome, "pnl": r.pnl})
                        # Clean up early exit tracking
                        if self.early_exit:
                            self.early_exit.remove_position(r.trade_id)
                    if resolved:
                        if self.drawdown_guard:
                            self.drawdown_guard.update_bankroll(self._total_capital())
                        await self._refresh_dashboard()
                except Exception:
                    pass

                if self._is_in_5m_entry_window():
                    if not self._5m_traded_this_window:
                        boundary = datetime.datetime.fromtimestamp(self._next_5m_boundary())
                        logger.info(f"⏱️ [5m] ENTRY — targeting {boundary.strftime('%H:%M')}")
                        await self._5m_trading_cycle()
                        self._5m_traded_this_window = True
                else:
                    # Reset when approaching next 5m entry window
                    cfg = self.config.active_5m
                    secs_until = self._next_5m_boundary() - cfg.entry_lead_secs - time.time()
                    if secs_until > 0 and secs_until < cfg.entry_lead_secs:
                        self._5m_traded_this_window = False

            except Exception as e:
                logger.error(f"[5m] Loop error: {e}", exc_info=True)

            await asyncio.sleep(self.config.sleep_poll_secs)

        logger.info("⏱️ [5m] Parallel trading loop stopped")

    # ── 1-Hour Parallel Loop ─────────────────────────────────────

    def _next_1h_boundary(self) -> float:
        """Next 1-hour boundary timestamp (:00)."""
        now = time.time()
        dt = datetime.datetime.fromtimestamp(now)
        b = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        return b.timestamp()

    def _is_in_1h_entry_window(self) -> bool:
        """Check if we're in the entry window for a 1hr boundary."""
        cfg = self.config.active_1h
        secs_until = self._next_1h_boundary() - cfg.entry_lead_secs - time.time()
        return -cfg.entry_window_secs <= secs_until <= 0

    async def _1h_trading_cycle(self):
        """Execute a single 1hr directional trade — mirrors _5m_trading_cycle with 1hr params."""
        self._1h_cycle_count += 1
        cfg = self.config.active_1h

        try:
            # 1. Capture 1hr anchor
            anchor = await self.oracle.capture_window_open(window_minutes=60)
            open_price = anchor.open_price if anchor else None
            self._1h_last_anchor_price = open_price

            # 2. Strategy delay for 1hr (longer drift window)
            delay = cfg.strategy_delay_secs
            if delay > 0 and open_price:
                logger.info(f"📌 [1hr] Anchor: ${open_price:,.2f} — waiting {delay}s...")
                await asyncio.sleep(delay)

            # 3. Fresh price + strategy
            consensus = await self.oracle.get_price()
            if not consensus or not consensus.price:
                logger.warning("[1hr] No oracle price — skipping")
                return

            candles = await self.oracle.get_candles("1h", limit=100)
            if len(candles) < 30:
                logger.warning(f"[1hr] Only {len(candles)} candles — skipping")
                return

            # Regime filter — block CHOPPY
            if self.regime_filter:
                self.regime_filter.update_baseline_from_candles(candles)
                regime = self.regime_filter.classify_from_candles(candles)
                self._last_regime = regime
                if regime == RegimeState.CHOPPY:
                    logger.info(f"[1hr] Cycle {self._1h_cycle_count}: HOLD — regime=CHOPPY")
                    return

            # Fee estimate
            market_fee_pct = None
            try:
                if self.polymarket._active_markets:
                    first_market = next(iter(self.polymarket._active_markets.values()), None)
                    if first_market:
                        token_id = first_market.token_id_up
                        fetched = await self.polymarket.get_fee_pct_for_price(token_id, first_market.price_up)
                        if fetched is not None:
                            market_fee_pct = fetched
            except Exception:
                pass

            # Build 1hr-tuned indicator overrides from Active1hConfig
            _1h_overrides = {
                "rsi_period": cfg.rsi_period,
                "macd_fast": cfg.macd_fast,
                "macd_slow": cfg.macd_slow,
                "macd_signal": cfg.macd_signal,
                "ema_fast": cfg.ema_fast,
                "ema_slow": cfg.ema_slow,
                "momentum_lookback": cfg.momentum_lookback,
                "weight_price_vs_open": cfg.weight_price_vs_open,
            }

            decision = self.strategy.analyze(candles, consensus.price, open_price=open_price,
                                              fee_pct=market_fee_pct, timeframe_overrides=_1h_overrides)

            if not decision.should_trade:
                logger.info(f"[1hr] Cycle {self._1h_cycle_count}: HOLD — {decision.reason}")
                return

            # Regime UNKNOWN override for 1hr
            if self.regime_filter and self._last_regime == RegimeState.UNKNOWN:
                if decision.confidence < self.config.regime.regime_override_conviction:
                    logger.info(
                        f"[1hr] Cycle {self._1h_cycle_count}: HOLD — regime=UNKNOWN, "
                        f"confidence {decision.confidence:.3f} < override"
                    )
                    return

            # Drawdown Guard (M3)
            if self.drawdown_guard:
                if self.drawdown_guard.is_halted():
                    logger.info(f"[1hr] Cycle {self._1h_cycle_count}: HALTED — drawdown RED tier")
                    return
                min_conf = self.drawdown_guard.get_min_confidence()
                if min_conf > 0 and decision.confidence < min_conf:
                    logger.info(
                        f"[1hr] Cycle {self._1h_cycle_count}: BLOCKED — confidence "
                        f"{decision.confidence:.2f} < drawdown floor {min_conf}"
                    )
                    return

            # Risk check (uses 15m risk budget for now — Phase C adds per-asset budgets)
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.info(f"[1hr] Cycle {self._1h_cycle_count}: BLOCKED — {reason}")
                return

            # Discover + filter to current 1hr window
            markets = await self.polymarket.discover_markets()
            tradeable = [m for m in markets if m.is_tradeable and m.liquidity >= self.config.polymarket.min_liquidity_usd]
            tradeable = self.polymarket.filter_current_window(tradeable, 60)
            if not tradeable:
                logger.info(f"[1hr] Cycle {self._1h_cycle_count}: No markets for current 1hr window")
                return

            market = max(tradeable, key=lambda m: m.liquidity)

            # Size + execute (capped by Active1hConfig.max_trade_size_usd)
            if self.position_sizer:
                dd_mult = self.drawdown_guard.get_size_multiplier() if self.drawdown_guard else 1.0
                size = self.position_sizer.compute_size(
                    confidence=decision.confidence,
                    bankroll=self.risk_manager.capital,
                    drawdown_mult=dd_mult,
                )
            else:
                size = self.risk_manager.calculate_position_size(decision.confidence)
                if self.drawdown_guard:
                    size = round(size * self.drawdown_guard.get_size_multiplier(), 2)
            # Cap to 1hr max trade size
            size = min(size, cfg.max_trade_size_usd)
            if size <= 0:
                return

            direction = decision.direction.value
            trade = await self.polymarket.place_order(
                market=market, direction=direction, size_usd=size,
                oracle_price=consensus.price, confidence=decision.confidence,
            )

            if trade:
                self._1h_trade_ids.add(trade.trade_id)
                self.trade_logger.log_trade({
                    "type": "directional_1h",
                    "trade_id": trade.trade_id, "direction": trade.direction,
                    "size_usd": trade.size_usd, "confidence": trade.confidence,
                    "oracle_price": trade.oracle_price_at_entry,
                    "order_id": trade.order_id,
                    "market": market.question[:80],
                })

                # Register with early exit system
                if self.early_exit:
                    try:
                        token_id = market.token_id_up if direction == "up" else market.token_id_down
                        close_ts = self._parse_market_close_ts(market)
                        self.early_exit.register_position(
                            trade_id=trade.trade_id,
                            token_id=token_id,
                            entry_price=trade.entry_price,
                            conviction=decision.confidence,
                            size_usd=trade.size_usd,
                            market_close_ts=close_ts,
                        )
                    except Exception as e:
                        logger.warning(f"[1hr] Early exit registration failed: {e}")

                logger.info(
                    f"🕐 [1hr] {direction.upper()} | ${size:.2f} @ conf={decision.confidence:.2f} | "
                    f"BTC=${consensus.price:,.2f}"
                )
                await self._refresh_dashboard()

        except Exception as e:
            logger.error(f"[1hr] Cycle {self._1h_cycle_count} error: {e}", exc_info=True)

    async def _1h_loop(self):
        """Independent 1hr trading loop — runs as async task alongside 15m/5m loops."""
        logger.info("🕐 [1hr] Parallel trading loop started")

        while self.running:
            try:
                # Check resolutions every tick
                try:
                    resolved = await self.polymarket.check_resolutions()
                    for r in resolved:
                        is_1h = r.trade_id in self._1h_trade_ids
                        is_5m = r.trade_id in self._5m_trade_ids
                        if is_1h:
                            self.risk_manager.record_trade(r.pnl)
                            self._1h_trade_ids.discard(r.trade_id)
                        elif is_5m:
                            self.risk_manager.record_5m_trade(0, pnl=r.pnl)
                            self._5m_trade_ids.discard(r.trade_id)
                        else:
                            self.risk_manager.record_trade(r.pnl)
                        self.trade_logger.log_resolution({"trade_id": r.trade_id, "outcome": r.outcome, "pnl": r.pnl})
                        if self.early_exit:
                            self.early_exit.remove_position(r.trade_id)
                    if resolved:
                        if self.drawdown_guard:
                            self.drawdown_guard.update_bankroll(self._total_capital())
                        await self._refresh_dashboard()
                except Exception:
                    pass

                if self._is_in_1h_entry_window():
                    if not self._1h_traded_this_window:
                        boundary = datetime.datetime.fromtimestamp(self._next_1h_boundary())
                        logger.info(f"🕐 [1hr] ENTRY — targeting {boundary.strftime('%H:%M')}")
                        await self._1h_trading_cycle()
                        self._1h_traded_this_window = True
                else:
                    # Reset when approaching next 1hr entry window
                    cfg = self.config.active_1h
                    secs_until = self._next_1h_boundary() - cfg.entry_lead_secs - time.time()
                    if secs_until > 0 and secs_until < cfg.entry_lead_secs:
                        self._1h_traded_this_window = False

            except Exception as e:
                logger.error(f"[1hr] Loop error: {e}", exc_info=True)

            await asyncio.sleep(self.config.sleep_poll_secs)

        logger.info("🕐 [1hr] Parallel trading loop stopped")

    # ── Unified Multi-Asset Loop (Phase C) ───────────────────────

    async def _multi_asset_loop(self, asset_symbol: str, timeframe: str):
        """
        Generic trading loop for any (asset, timeframe) combination.
        Used for ETH and SOL loops. BTC uses the legacy loops for now.
        """
        tag = f"[{asset_symbol}/{timeframe}]"
        asset_cfg = self.config.get_asset(asset_symbol)
        oracle = self.oracles.get(asset_symbol)
        strategy = self.strategies.get(asset_symbol)
        regime_flt = self.regime_filters.get(asset_symbol)
        if not asset_cfg:
            logger.error(f"{tag} Asset config not found")
            return

        # Determine timing params based on timeframe
        TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60}
        window_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)

        if timeframe == "5m":
            tf_cfg = self.config.active_5m
        elif timeframe == "1h":
            tf_cfg = self.config.active_1h
        else:
            tf_cfg = None  # Use global defaults for 15m

        strategy_delay = tf_cfg.strategy_delay_secs if tf_cfg else self.config.strategy_delay_secs
        entry_lead = tf_cfg.entry_lead_secs if tf_cfg else self.config.entry_lead_secs
        entry_window = tf_cfg.entry_window_secs if tf_cfg else self.config.entry_window_secs
        max_trade_usd = tf_cfg.max_trade_size_usd if tf_cfg else self.config.risk.max_trade_size_usd

        # Build indicator overrides from timeframe config
        indicator_overrides = {}
        if tf_cfg:
            indicator_overrides = {
                "rsi_period": tf_cfg.rsi_period,
                "macd_fast": tf_cfg.macd_fast,
                "macd_slow": tf_cfg.macd_slow,
                "macd_signal": tf_cfg.macd_signal,
                "ema_fast": tf_cfg.ema_fast,
                "ema_slow": tf_cfg.ema_slow,
                "momentum_lookback": tf_cfg.momentum_lookback,
                "weight_price_vs_open": tf_cfg.weight_price_vs_open,
            }

        traded_this_window = False
        logger.info(f"🔄 {tag} Multi-asset trading loop started")

        while self.running:
            try:
                # Resolution polling (locked to prevent double-counting across loops)
                try:
                    async with self._resolution_lock:
                        resolved = await self.polymarket.check_resolutions()
                        for r in resolved:
                            r_asset = "BTC"
                            r_mapping = self._trade_asset_map.get(r.trade_id)
                            if r_mapping:
                                r_asset = r_mapping[0]
                            self._route_pnl(r.trade_id, r.pnl)
                            self.trade_logger.log_resolution({"trade_id": r.trade_id, "outcome": r.outcome, "pnl": r.pnl})
                            if self.early_exit:
                                self.early_exit.remove_position(r.trade_id)
                            # Broadcast resolution to dashboard
                            if self.dashboard and self.dashboard.is_running:
                                try:
                                    await self.dashboard.broadcast({
                                        "type": "trade_notification",
                                        "action": "resolved",
                                        "asset": r_asset,
                                        "trade_id": r.trade_id,
                                        "outcome": r.outcome,
                                        "pnl": r.pnl,
                                        "timestamp": time.time(),
                                    })
                                except Exception:
                                    pass
                        if resolved:
                            if self.drawdown_guard:
                                self.drawdown_guard.update_bankroll(self._total_capital())
                            await self._refresh_dashboard()
                except Exception:
                    pass

                # Check if in entry window for this timeframe
                now = time.time()
                dt = datetime.datetime.fromtimestamp(now)

                # ── Between-window dashboard refresh (15m loops only) ──
                # Update regime + latest strategy analysis every tick so dashboard stays alive
                if timeframe == "15m" and oracle and strategy and regime_flt:
                    try:
                        candle_iv = "15m"
                        candles = await oracle.get_candles(candle_iv, limit=100)
                        if len(candles) >= 30:
                            regime_flt.update_baseline_from_candles(candles)
                            regime = regime_flt.classify_from_candles(candles)
                            self._last_regimes[asset_symbol] = regime
                            if asset_symbol == "BTC":
                                self._last_regime = regime

                            # Run strategy for dashboard display (not for trading)
                            consensus = await oracle.get_price()
                            if consensus and consensus.price:
                                anchor_inst = oracle.get_window_anchor()
                                op = anchor_inst.open_price if anchor_inst else None
                                full_ovr = {
                                    "dead_zone_pct": asset_cfg.dead_zone_pct if asset_cfg else 0.04,
                                    "confidence_threshold": asset_cfg.confidence_threshold if asset_cfg else 0.72,
                                    "max_volatility_pct": asset_cfg.max_volatility_pct if asset_cfg else 3.0,
                                    **indicator_overrides,
                                }
                                decision = strategy.analyze(candles, consensus.price, open_price=op, timeframe_overrides=full_ovr)
                                self._last_decisions[asset_symbol] = decision
                                self._last_signals[asset_symbol] = {}
                                for sig in decision.signals:
                                    self._last_signals[asset_symbol][sig.name] = {
                                        "direction": sig.direction.value,
                                        "strength": round(sig.strength, 3),
                                        "raw_value": round(sig.raw_value, 4),
                                    }
                                if asset_symbol == "BTC":
                                    self._last_consensus = consensus
                                    self._last_decision = decision
                                    self._last_anchor = anchor_inst
                    except Exception:
                        pass

                if window_minutes == 5:
                    next_min = ((dt.minute // 5) + 1) * 5
                    if next_min >= 60:
                        boundary = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
                    else:
                        boundary = dt.replace(minute=next_min, second=0, microsecond=0)
                elif window_minutes == 60:
                    boundary = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
                else:  # 15m
                    next_min = ((dt.minute // 15) + 1) * 15
                    if next_min >= 60:
                        boundary = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
                    else:
                        boundary = dt.replace(minute=next_min, second=0, microsecond=0)

                boundary_ts = boundary.timestamp()
                secs_until = boundary_ts - entry_lead - now
                in_entry_window = -entry_window <= secs_until <= 0

                if in_entry_window:
                    if not traded_this_window:
                        logger.info(f"🔄 {tag} ENTRY — targeting {boundary.strftime('%H:%M')}")
                        await self._asset_trading_cycle(
                            asset_symbol=asset_symbol,
                            timeframe=timeframe,
                            window_minutes=window_minutes,
                            strategy_delay=strategy_delay,
                            indicator_overrides=indicator_overrides,
                            max_trade_usd=max_trade_usd,
                        )
                        traded_this_window = True
                else:
                    # Reset when approaching next entry window
                    if secs_until > 0 and secs_until < entry_lead:
                        traded_this_window = False

            except Exception as e:
                logger.error(f"{tag} Loop error: {e}", exc_info=True)

            await asyncio.sleep(self.config.sleep_poll_secs)

        logger.info(f"🔄 {tag} Multi-asset trading loop stopped")

    # ── Early Exit Loop ──────────────────────────────────────────

    @staticmethod
    def _parse_market_close_ts(market) -> float:
        """Extract market close timestamp from end_date field."""
        end_date = getattr(market, "end_date", "")
        if not end_date:
            # Fallback: assume 15 minutes from now
            return time.time() + 900
        try:
            import datetime as _dt
            return _dt.datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
        except Exception:
            return time.time() + 900

    async def _early_exit_loop(self):
        """
        Independent poll loop for early exit system.
        Checks CLOB bid prices for all tracked positions every poll_interval_secs.
        Executes sells (or logs dry-run) when TP/SL/trail thresholds are hit.
        """
        cfg = self.config.early_exit
        logger.info(
            f"🚪 Early exit loop started "
            f"({'DRY RUN' if cfg.dry_run else 'LIVE'}, "
            f"poll={cfg.poll_interval_secs}s)"
        )

        while self.running:
            try:
                tracked = self.early_exit.get_tracked()
                if not tracked:
                    await asyncio.sleep(cfg.poll_interval_secs)
                    continue

                for trade_id, pos in tracked.items():
                    try:
                        # Fetch current CLOB best bid for this position's token
                        bid_info = self.polymarket.get_best_bid(pos.token_id)
                        if not bid_info or bid_info["price"] <= 0:
                            continue

                        current_bid = bid_info["price"]

                        # Store bid on position for dashboard broadcasting
                        pos._live_bid = current_bid

                        # Check exit conditions (TP → SL → Trail priority)
                        signal = self.early_exit.check_position(trade_id, current_bid)
                        if not signal:
                            continue

                        # ── Exit triggered ──
                        exit_label = signal.exit_type.value.upper()

                        if cfg.dry_run:
                            # Log what would happen but don't sell
                            logger.info(
                                f"🚪 [DRY RUN] {exit_label} would fire for {trade_id}: "
                                f"entry=${signal.entry_price:.4f} → bid=${signal.current_bid:.4f} | "
                                f"net/share=${signal.net_pnl_per_share:+.4f} | "
                                f"held {signal.hold_duration_secs:.0f}s | "
                                f"{signal.reason}"
                            )
                            continue

                        # ── Execute sell ──
                        logger.info(
                            f"🚪 {exit_label} triggered for {trade_id}: "
                            f"entry=${signal.entry_price:.4f} → bid=${signal.current_bid:.4f} | "
                            f"{signal.reason}"
                        )

                        shares = pos.shares_estimate
                        sell_result = await self.polymarket.sell_shares(
                            token_id=pos.token_id,
                            shares=shares,
                            min_price=0.0,  # SL sells at any price
                            reason=signal.exit_type.value,
                        )

                        if sell_result:
                            # ── Successful sell ──
                            exit_price = sell_result["exit_price"]
                            self.early_exit.mark_exited(trade_id, signal.exit_type, exit_price)

                            # Compute actual P&L and update risk manager
                            from core.early_exit import compute_round_trip_fee
                            gross = exit_price - pos.entry_price
                            fees = compute_round_trip_fee(pos.entry_price, exit_price)
                            net_per_share = gross - fees
                            net_pnl = net_per_share * shares

                            # Route P&L to correct per-asset risk budget
                            self._route_pnl(trade_id, net_pnl)

                            # Update drawdown guard (global)
                            if self.drawdown_guard:
                                self.drawdown_guard.update_bankroll(self._total_capital())

                            # Log the exit trade
                            self.trade_logger.log_trade({
                                "type": f"early_exit_{signal.exit_type.value}",
                                "trade_id": trade_id,
                                "entry_price": pos.entry_price,
                                "exit_price": exit_price,
                                "shares": round(shares, 2),
                                "gross_pnl": round(gross * shares, 4),
                                "fees": round(fees * shares, 4),
                                "net_pnl": round(net_pnl, 4),
                                "hold_secs": round(pos.hold_duration_secs, 0),
                                "conviction": pos.conviction,
                                "peak_price": pos.trail_peak if pos.trail_active else None,
                                "order_id": sell_result.get("order_id", ""),
                                "verified": sell_result.get("verified", False),
                            })

                            # Remove from polymarket trade records so resolution
                            # doesn't double-count this position
                            self._remove_trade_record(trade_id)

                            logger.info(
                                f"🚪 ✅ {exit_label} executed: {trade_id} | "
                                f"${pos.entry_price:.4f} → ${exit_price:.4f} | "
                                f"net=${net_pnl:+.2f} | held {pos.hold_duration_secs:.0f}s"
                            )

                            # Dashboard: broadcast exit event + refresh state
                            if self.dashboard and self.dashboard.is_running:
                                try:
                                    await self.dashboard.broadcast({
                                        "type": "early_exit_event",
                                        "exit_type": signal.exit_type.value,
                                        "trade_id": trade_id,
                                        "entry_price": pos.entry_price,
                                        "exit_price": exit_price,
                                        "net_pnl": round(net_pnl, 4),
                                        "hold_secs": round(pos.hold_duration_secs, 0),
                                        "timestamp": time.time(),
                                    })
                                except Exception:
                                    pass
                            await self._refresh_dashboard()
                        else:
                            # Sell failed — position falls through to resolution
                            logger.warning(
                                f"🚪 {exit_label} sell FAILED for {trade_id} — "
                                f"falling through to resolution"
                            )

                    except Exception as e:
                        logger.error(f"Early exit check error for {trade_id}: {e}", exc_info=True)

                # ── Broadcast live bids to dashboard every tick ──
                if self.dashboard and self.dashboard.is_running and self.dashboard.client_count > 0:
                    try:
                        bid_data = {}
                        for tid, pos in tracked.items():
                            bid = getattr(pos, '_live_bid', None)
                            if bid and bid > 0:
                                bid_data[tid] = {
                                    "bid": round(bid, 4),
                                    "entry": pos.entry_price,
                                    "tp": pos.tp_price,
                                    "sl": pos.sl_price,
                                    "trail_active": pos.trail_active,
                                    "trail_peak": round(pos.trail_peak, 4) if pos.trail_active else None,
                                    "trail_trigger": round(pos.trail_trigger_price, 4) if pos.trail_active else None,
                                    "time_remaining": round(pos.time_remaining_secs, 0),
                                    "conviction": pos.conviction,
                                    "size_usd": pos.size_usd,
                                }
                        if bid_data:
                            await self.dashboard.broadcast({
                                "type": "position_bids",
                                "timestamp": time.time(),
                                "positions": bid_data,
                            })
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Early exit loop error: {e}", exc_info=True)

            await asyncio.sleep(cfg.poll_interval_secs)

        logger.info("🚪 Early exit loop stopped")

    def _remove_trade_record(self, trade_id: str):
        """
        Remove a trade from polymarket's _trade_records after early exit sell.
        Prevents resolution from double-counting a position we already sold.
        """
        self.polymarket._trade_records = [
            r for r in self.polymarket._trade_records if r.trade_id != trade_id
        ]

    # ── Dashboard Live Updates ──────────────────────────────────

    async def _refresh_dashboard(self):
        """Re-broadcast full state so dashboard updates immediately."""
        if not self.dashboard or not self.dashboard.is_running:
            return
        try:
            state = build_dashboard_state(
                cycle=self._cycle_count,
                consensus=self._last_consensus,
                anchor=self._last_anchor,
                decision=self._last_decision,
                risk_manager=self.risk_manager,
                polymarket_client=self.polymarket,
                config=self.config,
                early_exit_manager=self.early_exit,
                regime_filter=self.regime_filter,
                last_regime=self._last_regime,
                drawdown_guard=self.drawdown_guard,
                position_sizer=self.position_sizer,
                # ── Multi-asset (Phase D) ──
                oracles=self.oracles,
                risk_managers=self.risk_managers,
                regime_filters=self.regime_filters,
                last_regimes=self._last_regimes,
                trade_asset_map=self._trade_asset_map,
                last_decisions=self._last_decisions,
                last_signals=self._last_signals,
            )
            await self.dashboard.broadcast(state)
        except Exception:
            pass

    async def _price_push_loop(self):
        """Push live prices for ALL enabled assets to dashboard every 2 seconds."""
        while self.running:
            try:
                if self.dashboard and self.dashboard.is_running and self.dashboard.client_count > 0:
                    # Build per-asset price dict
                    prices = {}
                    for symbol, oracle_inst in self.oracles.items():
                        cl_pp = getattr(oracle_inst, '_rtds_chainlink_latest', None)
                        bn_pp = getattr(oracle_inst, '_rtds_binance_latest', None)
                        cl_price = cl_pp.price if cl_pp and hasattr(cl_pp, 'price') and cl_pp.price else 0
                        bn_price = bn_pp.price if bn_pp and hasattr(bn_pp, 'price') and bn_pp.price else 0
                        price = cl_price or bn_price
                        if price:
                            prices[symbol] = {
                                "price": price,
                                "chainlink": cl_price,
                                "binance": bn_price,
                            }

                    # Fallback: if BTC has no RTDS price, use last consensus
                    if "BTC" not in prices and self._last_consensus:
                        p = getattr(self._last_consensus, 'price', 0)
                        if p:
                            prices["BTC"] = {"price": p, "chainlink": p, "binance": 0}

                    if prices:
                        # Backward compat: top-level price/chainlink/binance = BTC
                        btc = prices.get("BTC", {})
                        msg = {
                            "type": "price_tick",
                            "price": btc.get("price", 0),
                            "chainlink": btc.get("chainlink", 0),
                            "binance": btc.get("binance", 0),
                            "timestamp": time.time(),
                            # Multi-asset prices (Phase D)
                            "assets": prices,
                        }
                        await self.dashboard.broadcast(msg)
            except Exception:
                pass
            await asyncio.sleep(2)

    async def _state_push_loop(self):
        """Broadcast full dashboard state every 5 seconds so all stats stay live."""
        while self.running:
            try:
                if self.dashboard and self.dashboard.is_running and self.dashboard.client_count > 0:
                    await self._refresh_dashboard()
            except Exception:
                pass
            await asyncio.sleep(5)

    # ── Clock Sync ──────────────────────────────────────────────

    def _next_boundary(self) -> float:
        now = time.time()
        dt = datetime.datetime.fromtimestamp(now)
        interval = max(1, int(self._directional_interval_mins))
        next_min = ((dt.minute // interval) + 1) * interval
        if next_min >= 60:
            b = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        else:
            b = dt.replace(minute=next_min, second=0, microsecond=0)
        return b.timestamp()

    def _seconds_until_entry(self) -> float:
        return self._next_boundary() - self.config.entry_lead_secs - time.time()

    def _is_in_entry_window(self) -> bool:
        secs = self._seconds_until_entry()
        return -self.config.entry_window_secs <= secs <= 0

    def _format_next_entry(self) -> str:
        entry_ts = self._next_boundary() - self.config.entry_lead_secs
        entry_dt = datetime.datetime.fromtimestamp(entry_ts)
        boundary_dt = datetime.datetime.fromtimestamp(self._next_boundary())
        return f"{entry_dt.strftime('%H:%M:%S')} (→ {boundary_dt.strftime('%H:%M')})"

    def _infer_market_interval_minutes(self, market) -> int | None:
        slug = (getattr(market, "slug", "") or "").lower()
        m = re.search(r"btc-updown-(\d+)(m|h)-", slug)
        if m:
            qty = int(m.group(1))
            unit = m.group(2)
            return qty * (60 if unit == "h" else 1)

        text = f"{getattr(market, 'question', '')} {getattr(market, 'slug', '')}".lower()
        if any(k in text for k in ["5-min", "5 min", "5m", "5-minute"]):
            return 5
        if any(k in text for k in ["15-min", "15 min", "15m", "15-minute"]):
            return 15
        return None

    async def _refresh_directional_interval(self, force: bool = False):
        # When 5m parallel loop is active, lock main loop to 15m only
        if hasattr(self.config, 'active_5m') and self.config.active_5m.enabled:
            self._directional_interval_mins = 15
            return

        now = time.time()
        if not force and (now - self._last_interval_refresh) < 45:
            return
        self._last_interval_refresh = now

        markets = await self.polymarket.discover_markets()
        tradeable = [m for m in markets if m.is_tradeable and m.liquidity >= self.config.polymarket.min_liquidity_usd]
        if not tradeable:
            return

        intervals = {self._infer_market_interval_minutes(m) for m in tradeable}
        intervals.discard(None)
        if not intervals:
            return

        target = 15 if 15 in intervals else (5 if 5 in intervals else min(intervals))
        if target != self._directional_interval_mins:
            logger.info(
                f"Directional interval switched: {self._directional_interval_mins}m -> {target}m "
                f"(available: {sorted(intervals)})"
            )
            self._directional_interval_mins = target
            self._traded_this_window = False

    # ── Main Loop ───────────────────────────────────────────────

    async def run(self):
        print()
        print("=" * 60)
        print(f"  GRIDPHANTOMDEV — LIVE (v{self.config.version})")
        print(f"  Bankroll: ${self.config.bankroll:,.2f}")
        # ── Multi-asset summary ──
        enabled = self.config.enabled_assets()
        asset_str = ", ".join(f"{a.symbol} ({a.budget_pct:.0f}%)" for a in enabled)
        print(f"  Assets: {asset_str}")
        print(f"  Oracles: {', '.join(self.oracles.keys())}")
        a5 = getattr(self.config, 'active_5m', None)
        print(f"  5m Parallel: {'ON (budget=' + str(a5.budget_pct) + '%, max $' + str(a5.max_trade_size_usd) + '/trade, delay=' + str(a5.strategy_delay_secs) + 's)' if a5 and a5.enabled else 'off'}")
        a1h = getattr(self.config, 'active_1h', None)
        print(f"  1hr Parallel: {'ON (max $' + str(a1h.max_trade_size_usd) + '/trade, delay=' + str(a1h.strategy_delay_secs) + 's, lead=' + str(a1h.entry_lead_secs) + 's)' if a1h and a1h.enabled else 'off'}")
        ee = getattr(self.config, 'early_exit', None)
        if ee and ee.enabled:
            mode = "DRY RUN" if ee.dry_run else "LIVE"
            print(f"  Early Exit: {mode} (TP={'$' + str(ee.tp_offset) if ee.tp_enabled else 'off'}, "
                  f"SL={'$' + str(ee.sl_offset) if ee.sl_enabled else 'off'}, "
                  f"Trail={'$' + str(ee.trail_offset) if ee.trail_enabled else 'off'}, "
                  f"poll={ee.poll_interval_secs}s)")
        else:
            print(f"  Early Exit: off")
        rf = getattr(self.config, 'regime', None)
        print(f"  Regime Filter: {'ON (DCR>' + str(rf.dcr_trend_threshold) + ', NR>' + str(rf.nr_trend_threshold) + ')' if rf and rf.enabled else 'off'}")
        dd = self.drawdown_guard
        if dd:
            st = dd.get_status()
            print(f"  Drawdown Guard: ON (tier={st['tier']}, HWM=${st['high_water_mark']:.2f}, dd={st['drawdown_pct']:.1f}%)")
        else:
            print(f"  Drawdown Guard: off")
        print(f"  Position Sizer: {'ON (confidence-scaled)' if self.position_sizer else 'off (flat Kelly)'}")
        print(f"  Entry: {self.config.entry_lead_secs}s before each 15m boundary")
        print(f"  Strategy delay: {getattr(self.config, 'strategy_delay_secs', 0)}s after anchor capture")
        print(f"  Next: {self._format_next_entry()}")
        if self.dashboard:
            print(f"  Dashboard: http://localhost:8765")
        print("=" * 60)
        print()

        # Start dashboard server if enabled
        if self.dashboard:
            await self.dashboard.start()

        # Start persistent RTDS price streams — one per enabled asset
        rtds_tasks = []
        for symbol, oracle_inst in self.oracles.items():
            task = asyncio.create_task(oracle_inst.start_rtds_stream())
            rtds_tasks.append(task)
            logger.info(f"🔌 [{symbol}] RTDS persistent stream launched")

        # ── Multi-asset market discovery validation (Phase A) ──
        # Discover markets for ALL enabled assets and log what's available
        for asset_cfg in self.config.enabled_assets():
            try:
                discovered = await self.polymarket.discover_markets(asset_config=asset_cfg)
                if discovered:
                    tf_counts: dict[str, int] = {}
                    for m in discovered:
                        for tf in asset_cfg.timeframes:
                            if f"-{tf}-" in m.slug:
                                tf_counts[tf] = tf_counts.get(tf, 0) + 1
                    logger.info(
                        f"🔍 [{asset_cfg.symbol}] Market discovery: "
                        f"{len(discovered)} markets — {tf_counts}"
                    )
                else:
                    logger.warning(f"🔍 [{asset_cfg.symbol}] Market discovery: NO markets found")
            except Exception as e:
                logger.warning(f"🔍 [{asset_cfg.symbol}] Market discovery failed: {e}")

        # ── Launch ALL asset trading loops via unified _multi_asset_loop (Phase C fix) ──
        # BTC now uses the same path as ETH/SOL — no more legacy loops.
        # This fixes: duplicate resolutions, missing asset tags, missing correlation guard on BTC.
        multi_asset_tasks = []
        for asset_cfg in self.config.enabled_assets():
            for tf in asset_cfg.timeframes:
                # Only launch timeframes that are enabled
                if tf == "5m" and not (hasattr(self.config, 'active_5m') and self.config.active_5m.enabled):
                    continue
                if tf == "1h" and not (hasattr(self.config, 'active_1h') and self.config.active_1h.enabled):
                    continue
                task = asyncio.create_task(self._multi_asset_loop(asset_cfg.symbol, tf))
                multi_asset_tasks.append(task)
                logger.info(f"🔄 [{asset_cfg.symbol}/{tf}] Trading loop launched")

        # Start early exit monitoring loop
        early_exit_task = None
        if self.early_exit:
            early_exit_task = asyncio.create_task(self._early_exit_loop())
            logger.info("🚪 Early exit loop launched as independent task")

        # Start dashboard live price push
        price_push_task = None
        state_push_task = None
        if self.dashboard:
            price_push_task = asyncio.create_task(self._price_push_loop())
            state_push_task = asyncio.create_task(self._state_push_loop())
            logger.info("📊 Dashboard live price + state push launched")

        self.running = True
        self._start_time = time.time()

        # Bankroll is set strictly from --bankroll CLI flag.
        logger.info(f"💰 Bankroll: ${self._total_capital():.2f} (from config)")

        # Main loop — just keeps the bot alive. All trading happens in _multi_asset_loop tasks.
        while self.running:
            await asyncio.sleep(self.config.sleep_poll_secs)

    def stop(self):
        self.running = False
        logger.info("Shutdown initiated")

    async def shutdown(self):
        self.stop()
        # Close all oracle instances
        for symbol, oracle_inst in self.oracles.items():
            await oracle_inst.close()
        await self.polymarket.close()
        if self.dashboard:
            await self.dashboard.stop()
        stats = self.polymarket.get_stats()
        if self.early_exit:
            ee_status = self.early_exit.get_status()
            stats["early_exit"] = ee_status
            logger.info(
                f"Early exit stats: {ee_status['exits']} | "
                f"net P&L=${ee_status['total_exit_pnl']:+.2f}"
            )
        self.trade_logger.save_performance({
            "status": "shutdown", "cycles": self._cycle_count,
            "uptime_secs": time.time() - self._start_time, **stats,
        })
        logger.info(f"Stopped after {self._cycle_count} cycles")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTC-Directional — Polymarket Prediction Bot")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Starting bankroll in USD (default: 500)")
    parser.add_argument("--cycles", type=int, default=0, help="Max cycles, 0=unlimited (default: 0)")
    parser.add_argument("--5m", dest="fivem", action="store_true", help="Enable parallel 5m directional trading")
    parser.add_argument("--1h", dest="onehr", action="store_true", help="Enable parallel 1hr directional trading")
    parser.add_argument("--eth", action="store_true", help="Enable ETH directional trading")
    parser.add_argument("--sol", action="store_true", help="Enable SOL directional trading")
    parser.add_argument("--early-exit", action="store_true", help="Enable early exit system (TP/SL/trailing stop)")
    parser.add_argument("--early-exit-dry-run", action="store_true", help="Early exit dry run: log exits but don't sell")
    parser.add_argument("--dashboard", action="store_true", help="Start WebSocket dashboard on :8765")
    parser.add_argument("--acknowledge-drawdown", action="store_true", help="Acknowledge RED drawdown tier and resume trading")
    parser.add_argument("--no-drawdown-guard", action="store_true", help="Disable drawdown guard (not recommended)")
    parser.add_argument("--no-regime-filter", action="store_true", help="Disable regime filter")
    parser.add_argument("--no-sizer", action="store_true", help="Disable confidence-scaled position sizer (use flat Kelly)")
    parser.add_argument("--sync-live-bankroll", action="store_true", help="Sync risk bankroll from live Polymarket account balance")
    parser.add_argument("--live-bankroll-poll-secs", type=int, default=60, help="Live bankroll sync interval in seconds (default: 60)")
    parser.add_argument("--strategy-delay", type=int, default=None, help="Override strategy_delay_secs (seconds to wait after anchor capture)")
    args = parser.parse_args()

    config = BotConfig(bankroll=args.bankroll)
    config.polymarket.sync_live_bankroll = args.sync_live_bankroll
    config.polymarket.live_bankroll_poll_secs = args.live_bankroll_poll_secs
    config.active_5m.enabled = args.fivem
    config.active_1h.enabled = args.onehr
    # Enable ETH/SOL assets via CLI
    if args.eth:
        eth = config.get_asset("ETH")
        if eth:
            eth.enabled = True
            logger.info("ETH trading enabled via --eth flag")
    if args.sol:
        sol = config.get_asset("SOL")
        if sol:
            sol.enabled = True
            logger.info("SOL trading enabled via --sol flag")
    config.early_exit.enabled = args.early_exit or args.early_exit_dry_run
    config.early_exit.dry_run = args.early_exit_dry_run
    if args.no_drawdown_guard:
        config.drawdown.enabled = False
    if args.no_regime_filter:
        config.regime.enabled = False
    if args.no_sizer:
        config.sizer.enabled = False
    config._acknowledge_drawdown = args.acknowledge_drawdown

    # Apply strategy delay override if provided via CLI
    if args.strategy_delay is not None:
        config.strategy_delay_secs = max(0, args.strategy_delay)

    bot = BTCPredictionBot(config, dashboard=args.dashboard)

    def handle_signal(sig, frame):
        print("\n\nCtrl+C — shutting down...")
        bot.stop()
    signal.signal(signal.SIGINT, handle_signal)

    try:
        # --cycles mode is no longer separate — run() launches all unified loops.
        # Cycles limit is handled by the bot's internal cycle counter.
        await bot.run()
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
