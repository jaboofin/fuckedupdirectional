"""
Dashboard Server — HTTP + WebSocket via aiohttp.

  http://localhost:8765      → Dashboard HTML page
  ws://localhost:8765/ws     → Live state broadcast
  http://localhost:8765/state → JSON snapshot

  python bot.py --bankroll 500 --dashboard
  → Open http://localhost:8765 in your browser
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger("dashboard")

_DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "dashboard.html"


class DashboardServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients: set[web.WebSocketResponse] = set()
        self._state: dict = {}
        self._running = False
        self._runner: Optional[web.AppRunner] = None

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._handle_page)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/state", self._handle_state)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self._running = True
        logger.info(f"Dashboard: http://localhost:{self.port}")

    async def _handle_page(self, request):
        try:
            html = _DASHBOARD_HTML.read_text(encoding="utf-8")
            return web.Response(text=html, content_type="text/html")
        except FileNotFoundError:
            return web.Response(
                text="<h1>dashboard.html not found</h1><p>Expected at: {}</p>".format(_DASHBOARD_HTML),
                content_type="text/html", status=404,
            )

    async def _handle_state(self, request):
        return web.json_response(self._state)

    async def _handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        logger.info(f"Dashboard client connected ({len(self.clients)} total)")
        try:
            if self._state:
                await ws.send_json(self._state)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.PING:
                    await ws.pong(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            self.clients.discard(ws)
            logger.info(f"Dashboard client disconnected ({len(self.clients)} remaining)")
        return ws

    async def broadcast(self, state: dict):
        self._state = state
        dead = set()
        for ws in self.clients:
            try:
                await ws.send_json(state)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    async def stop(self):
        self._running = False
        for ws in list(self.clients):
            await ws.close()
        self.clients.clear()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Dashboard server stopped")

    @property
    def client_count(self):
        return len(self.clients)

    @property
    def is_running(self):
        return self._running


def build_dashboard_state(cycle, consensus, anchor, decision, risk_manager,
                          polymarket_client, config, early_exit_manager=None,
                          regime_filter=None, last_regime=None, drawdown_guard=None,
                          position_sizer=None,
                          # ── Multi-asset params (Phase D) ──
                          oracles=None, risk_managers=None,
                          regime_filters=None, last_regimes=None,
                          trade_asset_map=None,
                          last_decisions=None, last_signals=None):
    """
    Build the JSON state dict that gets broadcast to dashboard clients.

    Multi-asset: if oracles/risk_managers/regime_filters dicts are provided,
    builds per-asset state blocks under "assets" key. The top-level oracle/risk/etc
    fields remain BTC-focused for backward compat with existing dashboard.html.
    """
    stats = polymarket_client.get_stats()
    risk_status = risk_manager.get_status()
    open_trades = polymarket_client.get_trade_records()
    live_capital = round(risk_manager.capital, 2)
    _trade_map = trade_asset_map or {}

    # ── Total capital across all assets ──
    if risk_managers:
        total_capital = round(sum(rm.capital for rm in risk_managers.values()), 2)
    else:
        total_capital = live_capital

    # Signals
    signals = {}
    for s in (decision.signals if decision else []):
        signals[s.name] = {
            "direction": s.direction.value,
            "strength": round(s.strength, 3),
            "raw_value": round(s.raw_value, 4),
            "description": s.description,
        }

    # Positions — tag with asset and timeframe from trade_asset_map
    open_pos = []
    for t in open_trades:
        if t.outcome is None:
            mapping = _trade_map.get(t.trade_id)
            asset = mapping[0] if mapping else "BTC"
            tf = mapping[1] if mapping else "15m"
            open_pos.append({
                "id": t.trade_id, "direction": t.direction,
                "size_usd": t.size_usd, "entry_price": t.entry_price,
                "confidence": t.confidence, "timestamp": t.timestamp,
                "oracle_price": t.oracle_price_at_entry,
                "asset": asset, "timeframe": tf,
            })

    closed_pos = []
    for t in open_trades:
        if t.outcome is not None:
            mapping = _trade_map.get(t.trade_id)
            asset = mapping[0] if mapping else "BTC"
            closed_pos.append({
                "id": t.trade_id, "direction": t.direction,
                "size_usd": t.size_usd, "entry_price": t.entry_price,
                "confidence": t.confidence, "pnl": t.pnl,
                "outcome": t.outcome, "timestamp": t.timestamp,
                "asset": asset,
            })

    # Early exit positions
    ee_positions = {}
    ee_stats = None
    if early_exit_manager:
        ee_stats = early_exit_manager.get_status()
        for tid, pos in early_exit_manager.get_tracked().items():
            ee_positions[tid] = {
                "entry_price": pos.entry_price,
                "conviction": pos.conviction,
                "size_usd": pos.size_usd,
                "tp_price": pos.tp_price,
                "sl_price": pos.sl_price,
                "trail_active": pos.trail_active,
                "trail_peak": pos.trail_peak if pos.trail_active else None,
                "trail_trigger": pos.trail_trigger_price if pos.trail_active else None,
                "trail_activation_price": pos.trail_activation_price,
                "time_remaining_secs": round(pos.time_remaining_secs, 0),
                "hold_secs": round(pos.hold_duration_secs, 0),
                "token_id": pos.token_id[:20] + "..." if pos.token_id else "",
                "live_bid": round(getattr(pos, '_live_bid', 0), 4),
            }

    # ── Multi-asset per-asset state blocks (Phase D) ──
    assets_state = {}
    if oracles and risk_managers:
        _last_regimes = last_regimes or {}
        _regime_filters = regime_filters or {}
        for symbol, oracle_inst in oracles.items():
            rm = risk_managers.get(symbol)
            rf = _regime_filters.get(symbol)
            lr = _last_regimes.get(symbol)

            # Get latest price from oracle's RTDS buffer
            cl_pp = getattr(oracle_inst, '_rtds_chainlink_latest', None)
            bn_pp = getattr(oracle_inst, '_rtds_binance_latest', None)
            price = 0
            if cl_pp and hasattr(cl_pp, 'price') and cl_pp.price:
                price = cl_pp.price
            elif bn_pp and hasattr(bn_pp, 'price') and bn_pp.price:
                price = bn_pp.price

            # Get anchor
            anchor_inst = getattr(oracle_inst, '_window_anchor', None)

            rm_status = rm.get_status() if rm else {}

            # Per-asset strategy decision
            _decisions = last_decisions or {}
            _sigs = last_signals or {}
            asset_decision = _decisions.get(symbol)
            asset_signals = _sigs.get(symbol, {})

            assets_state[symbol] = {
                "price": round(price, 2) if price else 0,
                "anchor_price": round(anchor_inst.open_price, 2) if anchor_inst and anchor_inst.open_price else None,
                "capital": round(rm.capital, 2) if rm else 0,
                "daily_trades": rm_status.get("daily_trades", 0),
                "daily_pnl": round(rm_status.get("daily_pnl", 0), 2),
                "total_pnl": round(rm_status.get("total_pnl", 0), 2),
                "consecutive_losses": rm_status.get("consecutive_losses", 0),
                "can_trade": rm_status.get("can_trade", False),
                "regime": lr.value if lr else "UNKNOWN",
                "regime_enabled": rf is not None,
                # Per-asset strategy state
                "strategy": {
                    "direction": asset_decision.direction.value if asset_decision else "hold",
                    "confidence": round(asset_decision.confidence, 3) if asset_decision else 0,
                    "should_trade": asset_decision.should_trade if asset_decision else False,
                    "drift_pct": round(asset_decision.drift_pct, 4) if asset_decision and asset_decision.drift_pct else None,
                    "volatility_pct": round(asset_decision.volatility_pct, 3) if asset_decision else 0,
                } if asset_decision else None,
                "signals": asset_signals,
            }

    # ── Enabled assets list for frontend ──
    enabled_assets = []
    for a in config.enabled_assets() if hasattr(config, 'enabled_assets') else []:
        enabled_assets.append({
            "symbol": a.symbol,
            "budget_pct": a.budget_pct,
            "timeframes": a.timeframes,
        })

    state = {
        "type": "state",
        "timestamp": time.time(),
        "cycle": cycle,
        "oracle": {
            "price": consensus.price if consensus else 0,
            "chainlink": consensus.chainlink_price if consensus else None,
            "sources": consensus.sources if consensus else [],
            "spread_pct": consensus.spread_pct if consensus else 0,
        },
        "anchor": {
            "open_price": anchor.open_price if anchor else None,
            "source": anchor.source if anchor else None,
            "drift_pct": decision.drift_pct if decision else None,
        },
        "strategy": {
            "direction": decision.direction.value if decision else "hold",
            "confidence": decision.confidence if decision else 0,
            "should_trade": decision.should_trade if decision else False,
            "reason": decision.reason if decision else "",
            "drift_pct": decision.drift_pct if decision else None,
            "volatility_pct": decision.volatility_pct if decision else 0,
        },
        "signals": signals,
        "stats": {
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "win_rate": stats.get("win_rate", 0),
            "total_pnl": stats.get("total_pnl", 0),
            "total_wagered": stats.get("total_wagered", 0),
            "total_trades": stats.get("total_trades", 0),
            "completed": stats.get("completed", 0),
            "pending": stats.get("pending", 0),
        },
        "risk": {
            "capital": live_capital,
            "total_capital": total_capital,
            "daily_trades": risk_status.get("daily_trades", 0),
            "max_daily_trades": config.risk.max_daily_trades,
            "daily_pnl": risk_status.get("daily_pnl", 0),
            "daily_loss_pct": risk_status.get("daily_loss_pct", 0),
            "consecutive_losses": risk_status.get("consecutive_losses", 0),
            "cooldown_active": risk_status.get("in_cooldown", False),
            "total_pnl": risk_status.get("total_pnl", 0),
            "5m_trades": risk_status.get("5m_trades", 0),
            "5m_pnl": risk_status.get("5m_pnl", 0),
        },
        "positions": {"open": open_pos, "closed": closed_pos[-50:]},
        "early_exit": {
            "enabled": early_exit_manager is not None,
            "positions": ee_positions,
            "stats": ee_stats,
        },
        "regime": {
            "enabled": regime_filter is not None,
            "state": last_regime.value if last_regime else "UNKNOWN",
            "snapshot": regime_filter.get_snapshot().to_dict() if regime_filter and regime_filter.get_snapshot() else None,
            "stats": regime_filter.get_stats() if regime_filter else None,
        },
        "drawdown": {
            "enabled": drawdown_guard is not None,
            "tier": drawdown_guard.get_status()["tier"] if drawdown_guard else "GREEN",
            "drawdown_pct": drawdown_guard.get_status()["drawdown_pct"] if drawdown_guard else 0,
            "hwm": drawdown_guard.get_status()["high_water_mark"] if drawdown_guard else 0,
            "size_mult": drawdown_guard.get_size_multiplier() if drawdown_guard else 1.0,
            "halted": drawdown_guard.is_halted() if drawdown_guard else False,
        },
        "config": {
            "bankroll": total_capital,
            "fivem_enabled": getattr(getattr(config, 'active_5m', None), 'enabled', False),
            "onehr_enabled": getattr(getattr(config, 'active_1h', None), 'enabled', False),
            "early_exit_enabled": early_exit_manager is not None,
            "early_exit_dry_run": getattr(getattr(config, 'early_exit', None), 'dry_run', False),
            "regime_filter_enabled": regime_filter is not None,
            "drawdown_guard_enabled": drawdown_guard is not None,
            "position_sizer_enabled": position_sizer is not None,
            "enabled_assets": enabled_assets,
        },
        # ── Multi-asset state (Phase D) ──
        "assets": assets_state,
    }

    return state
