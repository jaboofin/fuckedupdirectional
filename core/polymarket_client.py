"""
╔══════════════════════════════════════════════════════════════════╗
║  POLYMARKET CLOB CLIENT — LIVE TRADING                           ║
║  py-clob-client SDK · EIP-712 signing · FOK/GTC orders           ║
║  Multi-asset UP/DOWN (BTC/ETH/SOL)                               ║
║                                                                    ║
║  Discovery flow:                                                   ║
║    1. Gamma /events/slug/{slug} → get conditionId + prices         ║
║    2. CLOB get_market(conditionId) → get real token IDs            ║
║    3. Trade with real token IDs                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import logging
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

import asyncio

try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = None

logger = logging.getLogger("polymarket")

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs, MarketOrderArgs, OrderType, BookParams, OpenOrderParams,
    )
    from py_clob_client.order_builder.constants import BUY, SELL
    HAS_CLOB_SDK = True
except ImportError:
    HAS_CLOB_SDK = False
    logger.warning("py-clob-client not installed. Run: pip install py-clob-client")

TIMEFRAME_SECONDS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


class MarketStatus(Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


@dataclass
class BinaryMarket:
    condition_id: str
    question: str
    slug: str
    token_id_up: str
    token_id_down: str
    price_up: float
    price_down: float
    volume: float
    liquidity: float
    created_at: str
    end_date: str
    status: MarketStatus
    asset: str = "BTC"  # Which asset this market belongs to (BTC/ETH/SOL)
    neg_risk: bool = True  # BTC up/down markets are neg_risk
    tick_size: str = "0.01"
    resolved: bool = False
    resolution: Optional[str] = None

    @property
    def is_tradeable(self) -> bool:
        return self.status == MarketStatus.ACTIVE and not self.resolved

    @property
    def spread(self) -> float:
        return abs(1.0 - self.price_up - self.price_down)


@dataclass
class TradeRecord:
    trade_id: str
    timestamp: float
    market_condition_id: str
    direction: str
    confidence: float
    entry_price: float
    size_usd: float
    oracle_price_at_entry: float
    outcome: Optional[str] = None
    pnl: float = 0.0
    order_id: Optional[str] = None
    tx_hashes: list = field(default_factory=list)


def _safe_json(val):
    """Parse a value that might be a JSON string, list, or None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def _parse_market_from_event(event: dict, slug: str) -> Optional[BinaryMarket]:
    """
    Extract BinaryMarket from Gamma event response.
    Token IDs from clobTokenIds may be placeholders —
    real IDs get filled in by _enrich_with_clob() later.
    """
    if not event:
        return None
    markets = event.get("markets", [])
    if not markets:
        return None

    m = markets[0]
    cid = m.get("conditionId", m.get("id", ""))
    if not cid:
        return None

    # Parse token IDs (may be placeholders like "1","0")
    token_ids = _safe_json(m.get("clobTokenIds"))
    if len(token_ids) < 2:
        tokens = m.get("tokens", [])
        if len(tokens) >= 2:
            token_ids = [
                tokens[0].get("token_id", tokens[0].get("tokenId", "")),
                tokens[1].get("token_id", tokens[1].get("tokenId", "")),
            ]
        else:
            token_ids = ["", ""]

    # Parse prices
    raw_prices = _safe_json(m.get("outcomePrices"))
    if len(raw_prices) >= 2:
        try:
            price_up = float(raw_prices[0])
            price_down = float(raw_prices[1])
        except (ValueError, TypeError):
            price_up, price_down = 0.5, 0.5
    else:
        tokens = m.get("tokens", [])
        if len(tokens) >= 2:
            price_up = float(tokens[0].get("price", 0.5))
            price_down = float(tokens[1].get("price", 0.5))
        else:
            price_up, price_down = 0.5, 0.5

    return BinaryMarket(
        condition_id=cid,
        question=m.get("question", event.get("title", "")),
        slug=m.get("slug", slug),
        token_id_up=str(token_ids[0]),
        token_id_down=str(token_ids[1]),
        price_up=price_up,
        price_down=price_down,
        volume=float(m.get("volumeNum", m.get("volume", 0))),
        liquidity=float(m.get("liquidityClob", m.get("liquidityNum", 0))),
        created_at=m.get("createdAt", ""),
        end_date=m.get("endDate", event.get("endDate", "")),
        status=MarketStatus.ACTIVE,
    )


class PolymarketClient:
    def __init__(self, config):
        self.config = config.polymarket
        self._session: Optional[Any] = None
        self._clob: Optional[object] = None
        self._clob_initialized = False
        self._active_markets: dict[str, BinaryMarket] = {}
        self._trade_records: list[TradeRecord] = []
        self._archived_trades: list[TradeRecord] = []  # Resolved trades pruned from active list
        # ── Fee cache (Phase 1) ──
        self._fee_cache: dict[str, tuple[float, float]] = {}  # token_id → (fee_rate_bps, cached_at)
        self._fee_cache_ttl: int = getattr(config.polymarket, "fee_cache_ttl_secs", 60)
        self._fee_fallback_pct: float = getattr(config.polymarket, "fee_fallback_pct", 1.56)

    # ── CLOB Init ───────────────────────────────────────────────

    def _init_clob_client(self):
        if self._clob_initialized:
            return
        if not HAS_CLOB_SDK:
            raise RuntimeError("pip install py-clob-client")

        pk = self.config.private_key or os.getenv("POLY_PRIVATE_KEY", "")
        funder = os.getenv("POLY_FUNDER", "")
        sig = int(os.getenv("POLY_SIG_TYPE", "0"))

        if not pk:
            raise RuntimeError("Set POLY_PRIVATE_KEY")

        if sig == 0:
            self._clob = ClobClient(self.config.clob_api_url, key=pk, chain_id=self.config.chain_id)
        elif sig in (1, 2):
            if not funder:
                raise RuntimeError("Set POLY_FUNDER")
            self._clob = ClobClient(self.config.clob_api_url, key=pk, chain_id=self.config.chain_id, signature_type=sig, funder=funder)
        else:
            raise ValueError(f"Invalid POLY_SIG_TYPE: {sig}")

        self._clob.set_api_creds(self._clob.create_or_derive_api_creds())
        self._clob_initialized = True
        logger.info(f"CLOB ready (sig_type={sig})")

    def _ensure_clob(self):
        if not self._clob_initialized:
            self._init_clob_client()

    # ── HTTP ────────────────────────────────────────────────────

    async def _get_session(self) -> Any:
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for HTTP operations. Install with: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), headers={"Content-Type": "application/json"})
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Fee Rate Lookup (Phase 1) ──────────────────────────────

    async def get_fee_rate_bps(self, token_id: str) -> Optional[int]:
        """
        Query Polymarket CLOB for the fee rate (in basis points) for a token.
        Returns cached value if fresh enough. Returns None on failure.
        """
        now = time.time()
        cached = self._fee_cache.get(token_id)
        if cached and (now - cached[1]) < self._fee_cache_ttl:
            return int(cached[0])

        try:
            session = await self._get_session()
            url = f"{self.config.clob_api_url}/fee-rate?token_id={token_id}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.debug(f"Fee lookup failed ({resp.status}) for {token_id[:20]}...")
                    return None
                data = await resp.json()

            raw = data.get("fee_rate_bps", data.get("feeRateBps"))
            if raw is None:
                return None

            bps = int(raw)
            self._fee_cache[token_id] = (bps, now)
            return bps

        except Exception as e:
            logger.debug(f"Fee rate lookup error: {e}")
            return None

    async def get_fee_pct_for_price(self, token_id: str, price: float) -> float:
        """
        Get the effective taker fee % for a trade at a given share price.
        Uses CLOB fee-rate endpoint if available, else parabolic fallback.
        """
        bps = await self.get_fee_rate_bps(token_id)
        if bps is not None:
            fee_rate = bps / 10000.0
            effective = fee_rate * (1.0 - price) * 100
            return round(max(0.0, effective), 4)
        else:
            return round(self._fee_fallback_pct * 4.0 * price * (1.0 - price), 4)

    @staticmethod
    def _extract_balance_usd(payload: Any) -> Optional[float]:
        if payload is None: return None
        if isinstance(payload, (int, float)): return float(payload)
        if isinstance(payload, str):
            try: return float(payload)
            except ValueError: return None
        if isinstance(payload, dict):
            for k in ["available", "balance", "amount", "usdc", "buying_power", "buyingPower", "available_balance", "availableBalance"]:
                if k in payload:
                    v = PolymarketClient._extract_balance_usd(payload.get(k))
                    if v is not None: return v
            for v in payload.values():
                parsed = PolymarketClient._extract_balance_usd(v)
                if parsed is not None: return parsed
        if isinstance(payload, (list, tuple)):
            for v in payload:
                parsed = PolymarketClient._extract_balance_usd(v)
                if parsed is not None: return parsed
        return None

    async def get_available_balance_usd(self) -> Optional[float]:
        try:
            self._ensure_clob()
        except Exception as e:
            logger.warning(f"Balance sync init failed: {e}")
            return None
        for name in ["get_balance_allowance", "get_balance", "get_usdc_balance", "get_collateral", "get_available_balance"]:
            fn = getattr(self._clob, name, None)
            if not callable(fn): continue
            try:
                payload = await asyncio.to_thread(fn)
                bal = self._extract_balance_usd(payload)
                if bal is not None: return float(bal)
            except Exception as e:
                logger.debug(f"Balance method {name} failed: {e}")
        logger.warning("Could not read live balance")
        return None

    # ── Deterministic Slug Generation ────────────────────────────

    @staticmethod
    def _generate_slugs(timeframes: list = None, offsets: list = None, slug_prefix: str = "btc-updown") -> list[tuple[str, str]]:
        if timeframes is None: timeframes = ["15m", "5m"]
        if offsets is None: offsets = [-1, 0, 1, 2]
        now = int(time.time())
        slugs = []
        for tf in timeframes:
            secs = TIMEFRAME_SECONDS.get(tf)
            if not secs: continue
            for offset in offsets:
                ts = (now // secs + offset) * secs
                if ts > 0:
                    slugs.append((f"{slug_prefix}-{tf}-{ts}", tf))
        return slugs

    # ── CLOB Enrichment (get real token IDs) ─────────────────────

    def _enrich_with_clob(self, market: BinaryMarket) -> BinaryMarket:
        """
        Call CLOB get_market(conditionId) to get the REAL token IDs.
        Gamma's clobTokenIds are often placeholders like "1","0".
        CLOB returns: tokens[].token_id with the full ERC1155 IDs.
        """
        try:
            self._ensure_clob()
            clob_market = self._clob.get_market(market.condition_id)
            if clob_market and "tokens" in clob_market:
                tokens = clob_market["tokens"]
                if len(tokens) >= 2:
                    # Match Up/Down by outcome label
                    for t in tokens:
                        outcome = t.get("outcome", "").lower()
                        tid = t.get("token_id", "")
                        price = float(t.get("price", 0))
                        if outcome in ("up", "yes"):
                            market.token_id_up = tid
                            if price > 0:
                                market.price_up = price
                        elif outcome in ("down", "no"):
                            market.token_id_down = tid
                            if price > 0:
                                market.price_down = price
                    logger.info(
                        f"CLOB enriched {market.slug}: "
                        f"UP={market.token_id_up[:20]}... "
                        f"DOWN={market.token_id_down[:20]}... "
                        f"prices={market.price_up:.3f}/{market.price_down:.3f}"
                    )
                # Store neg_risk and tick_size
                market.neg_risk = clob_market.get("neg_risk", True)
                market.tick_size = str(clob_market.get("minimum_tick_size", "0.01"))
        except Exception as e:
            logger.warning(f"CLOB enrich failed for {market.condition_id[:16]}...: {e}")
        return market

    # ── Market Discovery ────────────────────────────────────────

    async def _fetch_event_by_slug(self, slug: str) -> Optional[dict]:
        try:
            session = await self._get_session()
            url = f"{self.config.gamma_api_url}/events/slug/{slug}"
            async with session.get(url) as resp:
                if resp.status != 200: return None
                return await resp.json()
        except Exception:
            return None

    async def _discover_by_slug(self, asset_config=None) -> list[BinaryMarket]:
        """PRIMARY: Gamma event slug → CLOB enrichment."""
        slug_prefix = asset_config.slug_prefix if asset_config else "btc-updown"
        asset_symbol = asset_config.symbol if asset_config else "BTC"
        timeframes = asset_config.timeframes if asset_config else None
        slugs = self._generate_slugs(timeframes=timeframes, slug_prefix=slug_prefix)
        found = []
        seen = set()

        tasks = {slug: self._fetch_event_by_slug(slug) for slug, tf in slugs}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for slug_key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception) or result is None:
                continue
            market = _parse_market_from_event(result, slug_key)
            if market and market.condition_id not in seen:
                market.asset = asset_symbol
                # Enrich with real token IDs from CLOB
                market = await asyncio.to_thread(self._enrich_with_clob, market)
                found.append(market)
                seen.add(market.condition_id)
                self._active_markets[market.condition_id] = market
        return found

    async def _discover_by_pagination(self, asset_config=None) -> list[BinaryMarket]:
        """FALLBACK: paginate /events."""
        asset_symbol = asset_config.symbol if asset_config else "BTC"
        # Keywords to match this asset in event titles/slugs
        asset_keywords = {
            "BTC": ["btc", "bitcoin"],
            "ETH": ["eth", "ethereum"],
            "SOL": ["sol", "solana"],
        }.get(asset_symbol.upper(), [asset_symbol.lower()])
        slug_prefix = asset_config.slug_prefix if asset_config else "btc-updown"

        try:
            session = await self._get_session()
            found = []
            seen = set()
            offset = 0
            for _ in range(6):
                params = {"active": "true", "closed": "false", "limit": 100, "offset": offset, "order": "id", "ascending": "false"}
                async with session.get(f"{self.config.gamma_api_url}/events", params=params) as resp:
                    if resp.status != 200: break
                    data = await resp.json()
                if not data: break
                for ev in data:
                    slug = ev.get("slug", "")
                    combined = f"{ev.get('title', '')} {slug}".lower()
                    is_asset = any(kw in combined for kw in asset_keywords)
                    is_updown = "updown" in slug or "up-or-down" in combined
                    if not (is_asset and is_updown): continue
                    market = _parse_market_from_event(ev, slug)
                    if market and market.condition_id not in seen:
                        market.asset = asset_symbol
                        market = await asyncio.to_thread(self._enrich_with_clob, market)
                        found.append(market)
                        seen.add(market.condition_id)
                        self._active_markets[market.condition_id] = market
                if len(data) < 100: break
                offset += 100
            return found
        except Exception as e:
            logger.error(f"Pagination discovery failed: {e}")
            return []

    async def discover_markets(self, asset_config=None) -> list[BinaryMarket]:
        asset_symbol = asset_config.symbol if asset_config else "BTC"
        slug_prefix = asset_config.slug_prefix if asset_config else "btc-updown"

        markets = await self._discover_by_slug(asset_config=asset_config)
        if markets:
            interval_counts: dict[str, int] = {}
            for m in markets:
                # Generic regex: {slug_prefix}-{tf}-{ts}
                match = re.search(rf"{re.escape(slug_prefix)}-(\d+[mh])-", m.slug)
                if match:
                    interval_counts[match.group(1)] = interval_counts.get(match.group(1), 0) + 1
            logger.info(f"Found {len(markets)} {asset_symbol} directional markets (slug-direct): {interval_counts}")
            return markets

        logger.info(f"[{asset_symbol}] Slug lookup found 0 — trying events pagination...")
        markets = await self._discover_by_pagination(asset_config=asset_config)
        interval_counts: dict[str, int] = {}
        for m in markets:
            match = re.search(rf"{re.escape(slug_prefix)}-(\d+[mh])-", m.slug)
            if match:
                interval_counts[match.group(1)] = interval_counts.get(match.group(1), 0) + 1
        logger.info(f"Found {len(markets)} {asset_symbol} directional markets: {interval_counts}")
        return markets

    @staticmethod
    def get_market_window_ts(market) -> Optional[int]:
        """Extract the window start timestamp from a market slug like {prefix}-{tf}-{ts}."""
        slug = getattr(market, "slug", "") or ""
        # Match any asset prefix: btc-updown-15m-{ts}, eth-updown-5m-{ts}, sol-updown-1h-{ts}
        m = re.search(r"\w+-updown-\d+[mh]-(\d+)", slug)
        return int(m.group(1)) if m else None

    @staticmethod
    def filter_current_window(markets: list, interval_minutes: int = 15) -> list:
        """
        Filter markets to the window the bot is TARGETING — the NEXT boundary.

        The bot wakes up ~60s before a boundary to trade the upcoming window.
        At 08:14, the next boundary is 08:15, so we want the 08:15-08:30 market.
        The 08:00-08:15 market is about to close and already priced at near-certainty.

        If we're early in the window (>90s from next boundary), use current window instead.
        """
        now = int(time.time())
        interval_secs = interval_minutes * 60
        current_boundary = (now // interval_secs) * interval_secs
        next_boundary = current_boundary + interval_secs

        next_window = []
        current_window = []

        for m in markets:
            ts = PolymarketClient.get_market_window_ts(m)
            if ts is None:
                next_window.append(m)
                continue
            if ts == next_boundary:
                next_window.append(m)
            elif ts == current_boundary:
                current_window.append(m)

        # How far into the current window are we?
        elapsed = now - current_boundary
        close_to_boundary = elapsed >= (interval_secs - 90)  # Within 90s of next boundary

        if next_window and close_to_boundary:
            skipped = len(markets) - len(next_window)
            if skipped > 0:
                logger.info(f"Window filter: {len(next_window)} next-window, {skipped} other skipped")
            return next_window

        if current_window and not close_to_boundary:
            skipped = len(markets) - len(current_window)
            if skipped > 0:
                logger.info(f"Window filter: {len(current_window)} current-window (early), {skipped} other skipped")
            return current_window

        # Fallback
        if next_window:
            logger.info(f"Window filter: {len(next_window)} next-window (fallback)")
            return next_window
        if current_window:
            logger.info(f"Window filter: {len(current_window)} current-window (fallback)")
            return current_window

        logger.warning("Window filter: no match, returning all")
        return markets

    # ── CLOB Price ──────────────────────────────────────────────

    def get_clob_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        if not self._clob_initialized: return None
        try:
            p = self._clob.get_price(token_id, side=side)
            if p is None:
                return None
            # SDK may return a dict like {"price": "0.495"} or a raw string/number
            if isinstance(p, dict):
                p = p.get("price", p.get("mid", p.get("bid" if side == "BUY" else "ask")))
            return float(p) if p is not None else None
        except Exception as e:
            logger.error(f"CLOB price: {e}")
            return None

    # ── Order Execution ─────────────────────────────────────────

    async def place_order(self, market: BinaryMarket, direction: str, size_usd: float,
                          price: Optional[float] = None, oracle_price: float = 0.0,
                          confidence: float = 0.0) -> Optional[TradeRecord]:
        token_id = market.token_id_up if direction == "up" else market.token_id_down
        market_price = market.price_up if direction == "up" else market.price_down
        if price is None: price = market_price
        if size_usd < 0.50:
            logger.warning(f"Size ${size_usd:.2f} too small")
            return None

        trade_id = f"T-{int(time.time() * 1000)}-{direction[0].upper()}"
        self._ensure_clob()

        try:
            mkt = self._active_markets.get(market.condition_id)

            clob_price = self.get_clob_price(token_id, side="BUY")
            exec_price = clob_price if clob_price else price
            logger.info(f"Price: {exec_price:.4f} (clob={clob_price}, gamma={price:.4f})")

            if exec_price < 0.01 or exec_price > 0.99:
                logger.error(f"Price {exec_price} out of bounds")
                return None

            shares = round(size_usd / exec_price, 2)
            if shares < 5: shares = 5.0  # Polymarket minimum order size

            mode = self.config.order_type.lower()
            resp = None
            fok_rejected_thin_book = False

            if mode == "market":
                logger.info(f"🔴 MARKET ORDER: {direction.upper()} ${size_usd:.2f} ({shares:.1f} shares)")

                # ── Fetch fee rate for this token (Phase 1) ──
                fee_bps = await self.get_fee_rate_bps(token_id) or 0
                if fee_bps > 0:
                    logger.info(f"📊 Fee rate: {fee_bps} bps for token {token_id[:20]}...")

                # ── Attempt 1: FOK (instant fill, best case) ────────
                try:
                    args = MarketOrderArgs(token_id=token_id, amount=size_usd, side=BUY, fee_rate_bps=fee_bps, order_type=OrderType.FOK)
                    signed = self._clob.create_market_order(args)
                    resp = self._clob.post_order(signed, OrderType.FOK)
                except Exception as fok_err:
                    fok_msg = str(fok_err).lower()
                    if "fully filled or killed" in fok_msg or "couldn't be fully filled" in fok_msg:
                        # SDK threw exception instead of returning error dict
                        logger.warning(f"FOK exception (thin book): {fok_err}")
                        fok_rejected_thin_book = True
                    else:
                        logger.error(f"FOK order error: {fok_err}", exc_info=True)
                        return None

                # Check response dict if FOK returned normally (didn't throw)
                if resp and not fok_rejected_thin_book:
                    fok_success = resp.get("success", False)
                    fok_status = resp.get("status", "unknown")

                    if not fok_success and fok_status not in ("matched", "live"):
                        fok_error = resp.get("errorMsg", "")
                        if "fully filled or killed" in fok_error.lower() or "couldn't be fully filled" in fok_error.lower():
                            fok_rejected_thin_book = True

                # ── Attempt 2: GTC limit with slippage ──────────────
                if fok_rejected_thin_book:
                    slippage_pct = getattr(self.config, 'max_slippage_pct', 2.0)
                    slippage_price = round(min(0.99, exec_price * (1 + slippage_pct / 100)), 2)
                    slippage_shares = round(size_usd / slippage_price, 2)
                    if slippage_shares < 5:
                        slippage_shares = 5.0

                    logger.warning(
                        f"FOK rejected (thin book) — retrying as GTC limit | "
                        f"{exec_price:.4f} → {slippage_price:.4f} (+{slippage_pct:.1f}% slippage) | "
                        f"{slippage_shares:.1f} shares"
                    )

                    try:
                        args2 = OrderArgs(
                            price=slippage_price, size=slippage_shares,
                            side=BUY, token_id=token_id, fee_rate_bps=fee_bps
                        )
                        signed2 = self._clob.create_order(args2)
                        resp = self._clob.post_order(signed2, OrderType.GTC)
                    except Exception as gtc_err:
                        logger.error(f"GTC fallback error: {gtc_err}", exc_info=True)
                        return None

                    gtc_status = resp.get("status", "unknown")
                    gtc_order_id = resp.get("orderID", "")

                    # If GTC is resting (not yet filled), wait then cancel
                    if gtc_status == "live" and gtc_order_id:
                        logger.info(f"🟡 GTC order resting — waiting 10s for fill...")
                        await asyncio.sleep(10)
                        try:
                            self._clob.cancel(gtc_order_id)
                            logger.warning(f"GTC cancelled after 10s — no fill")
                            resp = {"success": False, "status": "cancelled", "errorMsg": "GTC timeout — no fill"}
                        except Exception:
                            # Cancel failed → already filled
                            logger.info(f"GTC order already filled ✅")
                            resp["success"] = True
                            resp["status"] = "matched"

                    # Update exec_price to the slippage price for record keeping
                    if resp.get("success", False) or resp.get("status") in ("matched",):
                        exec_price = slippage_price
            else:
                # ── Fetch fee rate for this token (Phase 1) ──
                fee_bps = await self.get_fee_rate_bps(token_id) or 0
                logger.info(f"🔴 LIMIT ORDER: {direction.upper()} {shares:.1f} @ {exec_price:.4f} (fee={fee_bps}bps)")
                args = OrderArgs(price=exec_price, size=shares, side=BUY, token_id=token_id, fee_rate_bps=fee_bps)
                signed = self._clob.create_order(args)
                resp = self._clob.post_order(signed, OrderType.GTC)

            logger.info(f"Response: {json.dumps(resp, indent=2)}")
            order_id = resp.get("orderID", trade_id)
            success = resp.get("success", False)
            status = resp.get("status", "unknown")
            tx_hashes = resp.get("transactionsHashes", [])

            if not success and status not in ("matched", "live"):
                logger.error(f"FAILED: {resp.get('errorMsg', 'unknown')} ({status})")
                return None

            # ── Handle resting "live" orders (not yet filled) ─────
            # A "live" status means the order is on the book but NOT matched.
            # Wait briefly for a fill, then cancel if still resting.
            if status == "live" and not success:
                live_order_id = resp.get("orderID", "")
                if live_order_id:
                    logger.info(f"🟡 Order resting (status=live) — waiting 12s for fill...")
                    await asyncio.sleep(12)
                    try:
                        order_info = await asyncio.to_thread(
                            self._clob.get_order, live_order_id
                        )
                        live_check_status = (order_info or {}).get("status", "").lower()
                        if live_check_status in ("matched", "filled"):
                            logger.info(f"✅ Resting order filled: {live_order_id[:20]}...")
                            success = True
                            status = "matched"
                            tx_hashes = (order_info or {}).get("transactionsHashes", tx_hashes)
                        else:
                            # Still resting — cancel it
                            try:
                                self._clob.cancel(live_order_id)
                                logger.warning(f"🚫 Resting order cancelled after 12s — no fill (status={live_check_status})")
                            except Exception:
                                # Cancel failed — might have filled in the meantime
                                logger.info(f"Cancel failed (may have filled) — re-checking...")
                                try:
                                    order_info2 = await asyncio.to_thread(
                                        self._clob.get_order, live_order_id
                                    )
                                    recheck_status = (order_info2 or {}).get("status", "").lower()
                                    if recheck_status in ("matched", "filled"):
                                        logger.info(f"✅ Order filled during cancel attempt")
                                        success = True
                                        status = "matched"
                                        tx_hashes = (order_info2 or {}).get("transactionsHashes", tx_hashes)
                                    else:
                                        logger.error(f"🚫 Order in limbo: status={recheck_status}. NOT recording.")
                                        return None
                                except Exception:
                                    logger.error(f"🚫 Cannot verify resting order. NOT recording.")
                                    return None
                            return None
                    except Exception as e:
                        logger.error(f"🚫 Failed to check resting order: {e}. NOT recording.")
                        return None
                else:
                    logger.error(f"🚫 Live order with no orderID. NOT recording.")
                    return None

            # ── Fill Verification ──────────────────────────────────
            # CLOB can return success=true but settlement may fail.
            # Wait briefly then re-check order status to confirm.
            verified = False
            if success or status == "matched":
                if tx_hashes:
                    # Has tx hashes — wait 3s then verify via order lookup
                    await asyncio.sleep(3)
                    try:
                        order_info = await asyncio.to_thread(
                            self._clob.get_order, order_id
                        )
                        if order_info:
                            verified_status = order_info.get("status", "")
                            if verified_status.lower() in ("matched", "filled"):
                                verified = True
                                logger.info(f"✅ Fill VERIFIED: {order_id[:20]}... status={verified_status}")
                            else:
                                logger.warning(
                                    f"⚠️ Fill NOT verified: {order_id[:20]}... "
                                    f"status={verified_status} (expected matched/filled)"
                                )
                        else:
                            logger.warning(f"⚠️ Fill verification: order not found — {order_id[:20]}...")
                    except Exception as ve:
                        logger.warning(f"⚠️ Fill verification error: {ve}")
                else:
                    # No tx hashes at all — ghost fill
                    logger.warning(f"⚠️ GHOST FILL: success=true but no transactionsHashes!")

                if not verified:
                    # Second check: try get_order after another 2s
                    await asyncio.sleep(2)
                    try:
                        order_info2 = await asyncio.to_thread(
                            self._clob.get_order, order_id
                        )
                        if order_info2:
                            v2_status = order_info2.get("status", "")
                            if v2_status.lower() in ("matched", "filled"):
                                verified = True
                                logger.info(f"✅ Fill VERIFIED (retry): status={v2_status}")
                            else:
                                logger.error(
                                    f"🚫 PHANTOM FILL DETECTED: order {order_id[:20]}... "
                                    f"CLOB said matched but status={v2_status}. "
                                    f"NOT recording trade."
                                )
                                return None
                        else:
                            logger.error(
                                f"🚫 PHANTOM FILL: order {order_id[:20]}... "
                                f"not found on second check. NOT recording trade."
                            )
                            return None
                    except Exception as ve2:
                        # Can't verify — log but still record (conservative)
                        logger.warning(f"⚠️ Fill verification retry failed: {ve2} — recording trade anyway")

            # Fill price tracks what we actually paid
            fill_price = exec_price

            record = TradeRecord(trade_id=trade_id, timestamp=time.time(), market_condition_id=market.condition_id, direction=direction, confidence=confidence, entry_price=fill_price, size_usd=size_usd, oracle_price_at_entry=oracle_price, order_id=order_id, tx_hashes=tx_hashes)
            self._trade_records.append(record)
            logger.info(f"✅ {trade_id} | {direction.upper()} | ${size_usd:.2f} @ {fill_price:.4f} | shares={size_usd/fill_price:.1f} | {status}")
            return record
        except Exception as e:
            logger.error(f"Trade FAILED: {e}", exc_info=True)
            return None

    # ── Order Management ────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_clob()
        try: self._clob.cancel(order_id); return True
        except Exception as e: logger.error(f"Cancel: {e}"); return False

    def cancel_all_orders(self) -> bool:
        self._ensure_clob()
        try: self._clob.cancel_all(); return True
        except Exception as e: logger.error(f"Cancel all: {e}"); return False

    # ── Market Maker Support (Phase 4) ─────────────────────────

    def place_maker_order(self, token_id: str, price: float, size: float,
                          side: str = "BUY", fee_bps: int = 0) -> Optional[dict]:
        """
        Place a post-only GTC order. Rejected if it would immediately match.
        Returns the CLOB response dict or None on failure.
        """
        self._ensure_clob()
        try:
            from py_clob_client.order_builder.constants import BUY as SDK_BUY, SELL as SDK_SELL
            sdk_side = SDK_BUY if side.upper() == "BUY" else SDK_SELL

            args = OrderArgs(
                price=price, size=size, side=sdk_side,
                token_id=token_id, fee_rate_bps=fee_bps,
            )
            signed = self._clob.create_order(args)
            resp = self._clob.post_order(signed, OrderType.GTC, post_only=True)

            success = resp.get("success", False)
            status = resp.get("status", "unknown")

            if success or status == "live":
                order_id = resp.get("orderID", "")
                logger.info(f"📘 Maker {side} {size:.1f}@{price:.4f} → {status} ({order_id[:16]}...)")
                return resp
            else:
                err = resp.get("errorMsg", status)
                if "would cross" in str(err).lower() or "post only" in str(err).lower():
                    logger.info(f"📘 Maker {side} {size:.1f}@{price:.4f} rejected (would cross spread)")
                else:
                    logger.warning(f"Maker order failed: {err}")
                return None

        except Exception as e:
            logger.error(f"Maker order error: {e}")
            return None

    def cancel_market_orders(self, condition_id: str) -> bool:
        """Cancel all open orders for a specific market."""
        self._ensure_clob()
        try:
            self._clob.cancel_market_orders(market=condition_id)
            return True
        except Exception as e:
            logger.error(f"Cancel market orders: {e}")
            return False

    def get_open_orders_for_market(self, condition_id: str) -> list[dict]:
        """Get all open orders for a specific market."""
        self._ensure_clob()
        try:
            from py_clob_client.clob_types import OpenOrderParams
            params = OpenOrderParams(market=condition_id)
            result = self._clob.get_orders(params)
            # Result may be a list or paginated object
            if isinstance(result, list):
                return result
            return result if result else []
        except Exception as e:
            logger.debug(f"Get open orders: {e}")
            return []

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        self._ensure_clob()
        try:
            mid = self._clob.get_midpoint(token_id)
            if isinstance(mid, dict):
                mid = mid.get("mid", mid.get("midpoint", mid.get("price")))
            return float(mid) if mid else None
        except Exception as e:
            logger.debug(f"Midpoint: {e}")
            return None

    def get_order_book(self, token_id: str) -> Optional[dict]:
        """Get the full order book for a token. Returns {bids: [...], asks: [...]}."""
        self._ensure_clob()
        try:
            book = self._clob.get_order_book(token_id)
            if book is None:
                return None
            bids = [{"price": float(o.price), "size": float(o.size)} for o in (book.bids or [])]
            asks = [{"price": float(o.price), "size": float(o.size)} for o in (book.asks or [])]
            return {"bids": bids, "asks": asks, "tick_size": book.tick_size}
        except Exception as e:
            logger.debug(f"Order book: {e}")
            return None

    # ── Resolution ──────────────────────────────────────────────

    async def check_resolutions(self) -> list[TradeRecord]:
        """
        Poll CLOB API for each pending trade to check if market resolved.
        Works even after markets expire from _active_markets cache.
        """
        resolved = []
        self._ensure_clob()

        for r in self._trade_records:
            if r.outcome is not None:
                continue

            try:
                # Query CLOB for current market state
                clob_mkt = await asyncio.to_thread(
                    self._clob.get_market, r.market_condition_id
                )

                if not clob_mkt:
                    continue

                # Check if market is closed/resolved
                is_closed = clob_mkt.get("closed", False)
                if not is_closed:
                    continue

                # Find winning token
                tokens = clob_mkt.get("tokens", [])
                winner = None
                for t in tokens:
                    if t.get("winner", False):
                        outcome_str = t.get("outcome", "").lower()
                        if outcome_str in ("up", "yes"):
                            winner = "up"
                        elif outcome_str in ("down", "no"):
                            winner = "down"
                        break

                if not winner:
                    # Market closed but no winner yet (still resolving)
                    continue

                won = r.direction == winner
                r.outcome = "win" if won else "loss"
                # Win: shares pay $1 each, minus what we paid
                # Loss: we lose what we paid
                if won:
                    shares = r.size_usd / r.entry_price
                    r.pnl = round(shares - r.size_usd, 4)
                else:
                    r.pnl = round(-r.size_usd, 4)

                resolved.append(r)
                logger.info(
                    f"{'✅' if won else '❌'} {r.trade_id} | "
                    f"{r.outcome.upper()} ({winner}) | ${r.pnl:+.2f}"
                )

                # Update cached market if present
                m = self._active_markets.get(r.market_condition_id)
                if m:
                    m.resolved = True
                    m.resolution = winner

            except Exception as e:
                logger.debug(f"Resolution check for {r.trade_id}: {e}")
                continue

        self._prune_resolved()
        return resolved

    def _prune_resolved(self):
        """Move resolved trades older than 1 hour to archive. Keeps _trade_records lean."""
        cutoff = time.time() - 3600
        still_active = []
        for r in self._trade_records:
            if r.outcome is not None and r.timestamp < cutoff:
                self._archived_trades.append(r)
            else:
                still_active.append(r)
        pruned = len(self._trade_records) - len(still_active)
        if pruned:
            logger.info(f"🗂️ Pruned {pruned} resolved trades (archived: {len(self._archived_trades)} total)")
        self._trade_records = still_active

    # ── Auto-Sell Winners ────────────────────────────────────────

    async def auto_sell_winners(self) -> list[dict]:
        """
        After resolution, sell winning shares at ~$0.99 to cash out.
        Works with sig_type=1 (email/proxy) since on-chain redeem isn't available.
        Returns list of sell results.
        """
        self._ensure_clob()
        results = []

        for r in self._trade_records:
            if r.outcome != "win":
                continue
            # Skip if already sold (marked by pnl being finalized)
            if getattr(r, '_redeemed', False):
                continue

            try:
                # Get the winning token ID from cached market or CLOB
                m = self._active_markets.get(r.market_condition_id)
                if m:
                    token_id = m.token_id_up if r.direction == "up" else m.token_id_down
                else:
                    # Re-fetch from CLOB
                    clob_mkt = await asyncio.to_thread(
                        self._clob.get_market, r.market_condition_id
                    )
                    if not clob_mkt:
                        continue
                    tokens = clob_mkt.get("tokens", [])
                    if len(tokens) < 2:
                        continue
                    # Match direction to token
                    token_id = None
                    for t in tokens:
                        outcome = t.get("outcome", "").lower()
                        if (r.direction == "up" and outcome in ("up", "yes")) or \
                           (r.direction == "down" and outcome in ("down", "no")):
                            token_id = t.get("token_id")
                            break
                    if not token_id:
                        continue

                shares = round(r.size_usd / r.entry_price, 2)
                if shares < 5:
                    shares = 5.0

                # Sell winning shares at $0.99
                sell_price = 0.99
                logger.info(
                    f"💰 AUTO-SELL: {r.trade_id} | {r.direction.upper()} "
                    f"{shares:.1f} shares @ ${sell_price}"
                )

                args = OrderArgs(
                    price=sell_price, size=shares,
                    side=SELL, token_id=token_id
                )
                signed = self._clob.create_order(args)
                resp = self._clob.post_order(signed, OrderType.GTC)

                success = resp.get("success", False)
                status = resp.get("status", "unknown")

                if success or status in ("matched", "live"):
                    r._redeemed = True
                    results.append({
                        "trade_id": r.trade_id,
                        "direction": r.direction,
                        "shares": shares,
                        "status": status,
                        "order_id": resp.get("orderID", ""),
                    })
                    logger.info(f"💰 SOLD: {r.trade_id} | {status}")
                else:
                    logger.warning(
                        f"Auto-sell failed: {r.trade_id} | "
                        f"{resp.get('errorMsg', status)}"
                    )

            except Exception as e:
                logger.warning(f"Auto-sell error {r.trade_id}: {e}")
                continue

        return results

    # ── Early Exit — Sell Execution ─────────────────────────────

    def get_best_bid(self, token_id: str) -> Optional[dict]:
        """
        Get the best bid price and available depth for a token.
        Returns {"price": float, "size": float, "depth_usd": float} or None.
        """
        book = self.get_order_book(token_id)
        if not book or not book.get("bids"):
            return None
        bids = sorted(book["bids"], key=lambda b: b["price"], reverse=True)
        best = bids[0]
        # Total depth at best bid and below (for large orders)
        total_depth = sum(b["size"] for b in bids)
        return {
            "price": best["price"],
            "size": best["size"],
            "depth_usd": round(total_depth * best["price"], 2),
            "levels": len(bids),
        }

    async def sell_shares(
        self,
        token_id: str,
        shares: float,
        min_price: float = 0.0,
        reason: str = "",
    ) -> Optional[dict]:
        """
        Sell shares back to the CLOB orderbook (for early exit).

        Mirrors the buy path: FOK at best bid, GTC fallback at bid - 1 tick.
        If both fail, returns None (position falls through to normal resolution).

        Args:
            token_id: The ERC1155 token to sell
            shares: Number of shares to sell
            min_price: Minimum acceptable sell price (skip if best bid < this)
            reason: Label for logging (e.g. "take_profit", "stop_loss")

        Returns:
            {"exit_price": float, "shares_sold": float, "order_id": str,
             "status": str, "tx_hashes": list} or None on failure
        """
        self._ensure_clob()
        tag = f"[SELL:{reason.upper()}]" if reason else "[SELL]"

        try:
            # 1. Fetch best bid
            bid_info = self.get_best_bid(token_id)
            if not bid_info or bid_info["price"] <= 0:
                logger.warning(f"{tag} No bids on orderbook for {token_id[:20]}...")
                return None

            best_bid = bid_info["price"]
            bid_depth = bid_info["size"]

            if best_bid < min_price:
                logger.info(
                    f"{tag} Best bid ${best_bid:.4f} < min ${min_price:.4f} — skipping sell"
                )
                return None

            # 2. Check depth
            if bid_depth < shares * 0.5:
                logger.warning(
                    f"{tag} Thin book: {bid_depth:.1f} shares at bid vs {shares:.1f} to sell. "
                    f"Proceeding with caution."
                )

            if shares < 5:
                shares = 5.0  # Polymarket minimum

            logger.info(
                f"{tag} Selling {shares:.1f} shares @ best bid ${best_bid:.4f} "
                f"(depth={bid_depth:.1f} shares, {bid_info['levels']} levels)"
            )

            # 3. Fetch fee rate
            fee_bps = await self.get_fee_rate_bps(token_id) or 0

            # 4. Attempt FOK sell at best bid
            resp = None
            fok_rejected = False

            try:
                args = OrderArgs(
                    price=best_bid, size=shares,
                    side=SELL, token_id=token_id, fee_rate_bps=fee_bps,
                )
                signed = self._clob.create_order(args)
                resp = self._clob.post_order(signed, OrderType.FOK)
            except Exception as fok_err:
                fok_msg = str(fok_err).lower()
                if "fully filled or killed" in fok_msg or "couldn't be fully filled" in fok_msg:
                    logger.warning(f"{tag} FOK rejected (thin book): {fok_err}")
                    fok_rejected = True
                else:
                    logger.error(f"{tag} FOK sell error: {fok_err}", exc_info=True)
                    return None

            if resp and not fok_rejected:
                fok_success = resp.get("success", False)
                fok_status = resp.get("status", "unknown")
                if not fok_success and fok_status not in ("matched", "live"):
                    fok_error = resp.get("errorMsg", "")
                    if "fully filled or killed" in fok_error.lower() or "couldn't be fully filled" in fok_error.lower():
                        fok_rejected = True

            # 5. GTC fallback at bid - 1 tick
            exec_price = best_bid
            if fok_rejected:
                tick = 0.01
                fallback_price = round(max(0.01, best_bid - tick), 2)
                logger.warning(
                    f"{tag} FOK rejected — retrying GTC at ${fallback_price:.4f} "
                    f"(bid ${best_bid:.4f} - {tick})"
                )
                try:
                    args2 = OrderArgs(
                        price=fallback_price, size=shares,
                        side=SELL, token_id=token_id, fee_rate_bps=fee_bps,
                    )
                    signed2 = self._clob.create_order(args2)
                    resp = self._clob.post_order(signed2, OrderType.GTC)
                except Exception as gtc_err:
                    logger.error(f"{tag} GTC sell fallback error: {gtc_err}", exc_info=True)
                    return None

                exec_price = fallback_price

                # If GTC is resting, wait then cancel
                gtc_status = resp.get("status", "unknown")
                gtc_order_id = resp.get("orderID", "")
                if gtc_status == "live" and gtc_order_id:
                    logger.info(f"{tag} GTC sell resting — waiting 8s for fill...")
                    await asyncio.sleep(8)
                    try:
                        order_info = await asyncio.to_thread(
                            self._clob.get_order, gtc_order_id
                        )
                        check_status = (order_info or {}).get("status", "").lower()
                        if check_status in ("matched", "filled"):
                            logger.info(f"{tag} GTC sell filled ✅")
                            resp["success"] = True
                            resp["status"] = "matched"
                        else:
                            try:
                                self._clob.cancel(gtc_order_id)
                                logger.warning(f"{tag} GTC sell cancelled after 8s — no fill")
                            except Exception:
                                logger.info(f"{tag} GTC cancel failed (may have filled)")
                                resp["success"] = True
                                resp["status"] = "matched"
                            if not resp.get("success"):
                                return None
                    except Exception:
                        try:
                            self._clob.cancel(gtc_order_id)
                        except Exception:
                            pass
                        return None

            # 6. Check response
            success = resp.get("success", False)
            status = resp.get("status", "unknown")
            order_id = resp.get("orderID", "")
            tx_hashes = resp.get("transactionsHashes", [])

            if not success and status not in ("matched",):
                logger.warning(
                    f"{tag} Sell not confirmed: {resp.get('errorMsg', status)} — "
                    f"position falls to resolution"
                )
                return None

            # 7. Fill verification (same pattern as buy)
            verified = False
            if success or status == "matched":
                await asyncio.sleep(3)
                try:
                    order_info = await asyncio.to_thread(
                        self._clob.get_order, order_id
                    )
                    if order_info:
                        v_status = order_info.get("status", "").lower()
                        if v_status in ("matched", "filled"):
                            verified = True
                            logger.info(f"{tag} Sell fill VERIFIED ✅: {order_id[:20]}...")
                        else:
                            logger.warning(f"{tag} Sell fill NOT verified: status={v_status}")
                except Exception as ve:
                    logger.warning(f"{tag} Sell verification error: {ve}")

                if not verified:
                    await asyncio.sleep(2)
                    try:
                        order_info2 = await asyncio.to_thread(
                            self._clob.get_order, order_id
                        )
                        if order_info2:
                            v2_status = order_info2.get("status", "").lower()
                            if v2_status in ("matched", "filled"):
                                verified = True
                                logger.info(f"{tag} Sell fill VERIFIED (retry) ✅")
                            else:
                                logger.error(
                                    f"{tag} PHANTOM SELL: status={v2_status}. "
                                    f"Position falls to resolution."
                                )
                                return None
                        else:
                            logger.error(f"{tag} Sell order not found on retry. Falls to resolution.")
                            return None
                    except Exception:
                        logger.warning(f"{tag} Sell verification retry failed — recording anyway")

            result = {
                "exit_price": exec_price,
                "shares_sold": shares,
                "order_id": order_id,
                "status": status,
                "tx_hashes": tx_hashes,
                "verified": verified,
            }

            logger.info(
                f"{tag} ✅ SOLD {shares:.1f} shares @ ${exec_price:.4f} | "
                f"order={order_id[:20]}... | verified={verified}"
            )
            return result

        except Exception as e:
            logger.error(f"{tag} Sell failed: {e}", exc_info=True)
            return None

    # ── Stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        all_trades = self._trade_records + self._archived_trades
        total_wagered = sum(r.size_usd for r in all_trades)
        done = [r for r in all_trades if r.outcome]
        if not done:
            return {"total_trades": len(all_trades), "completed": 0, "pending": len(all_trades), "win_rate": 0.0, "total_pnl": 0.0, "total_wagered": round(total_wagered, 2)}
        w = sum(1 for r in done if r.outcome == "win")
        pnl = sum(r.pnl for r in done)
        return {"total_trades": len(all_trades), "completed": len(done), "pending": len(all_trades) - len(done), "wins": w, "losses": len(done) - w, "win_rate": (w / len(done)) * 100, "total_pnl": pnl, "total_wagered": round(total_wagered, 2)}

    def get_trade_records(self) -> list[TradeRecord]:
        return (self._archived_trades + self._trade_records).copy()
