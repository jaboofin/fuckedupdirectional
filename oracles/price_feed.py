"""
╔══════════════════════════════════════════════════════════════════╗
║  ORACLE ENGINE — Chainlink-First Multi-Asset Price Feed          ║
║                                                                    ║
║  Polymarket markets resolve against Chainlink {ASSET}/USD.       ║
║  This engine:                                                      ║
║    1. Maintains PERSISTENT Chainlink RTDS websocket stream       ║
║    2. Maintains PERSISTENT RTDS Binance feed as secondary        ║
║    3. Falls back to Binance REST + CoinGecko for redundancy       ║
║    4. Tracks the OPENING PRICE of each window per asset           ║
║    5. Provides candles from Binance for technical analysis         ║
║                                                                    ║
║  v3.0 — Multi-asset support (BTC/ETH/SOL) via AssetConfig       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import json
import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional
from statistics import median

import aiohttp

logger = logging.getLogger("oracle")


@dataclass
class PricePoint:
    source: str
    price: float
    timestamp: float
    volume_24h: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_stale(self, max_age: int = 30) -> bool:
        return self.age_seconds > max_age


@dataclass
class ConsensusPrice:
    price: float
    timestamp: float
    sources: list
    spread_pct: float
    confidence: float
    chainlink_price: Optional[float] = None  # The actual resolution oracle

    def __repr__(self):
        src = ", ".join(self.sources)
        cl = f" CL=${self.chainlink_price:,.2f}" if self.chainlink_price else ""
        return f"${self.price:,.2f} | spread={self.spread_pct:.3f}% | [{src}]{cl}"


@dataclass
class WindowAnchor:
    """Tracks the opening price of the current 15-min window."""
    boundary_time: float         # Unix ts of boundary start (e.g. 12:00:00)
    open_price: float            # Chainlink BTC/USD at boundary start
    source: str                  # "chainlink", "rtds_binance", "binance" (fallback)
    captured_at: float           # When we recorded it

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at

    def price_vs_open(self, current: float) -> float:
        """Current price relative to open, as percentage."""
        if self.open_price <= 0:
            return 0.0
        return ((current - self.open_price) / self.open_price) * 100


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = "15m"


class OracleEngine:
    """
    Chainlink-first price oracle with window anchor tracking.
    Instantiate one per asset (BTC, ETH, SOL).

    Priority:
      1. Chainlink {ASSET}/USD (via Polymarket RTDS websocket)
      2. RTDS Binance {PAIR} (same websocket, different topic)
      3. Binance REST {PAIR} (fast, reliable, tracks Chainlink closely)
      4. CoinGecko {ASSET}/USD (fallback)

    Also tracks the opening price of each window,
    since Polymarket resolves: close >= open → UP, else DOWN.
    """

    MAX_DIVERGENCE_PCT = 1.0
    RTDS_URL = "wss://ws-live-data.polymarket.com"

    # ── RTDS connection health tracking ──
    _rtds_consecutive_failures: int = 0
    _rtds_last_success: float = 0.0
    _rtds_total_attempts: int = 0
    _rtds_total_successes: int = 0

    def __init__(self, config, asset_config=None):
        self.config = config.oracle
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_prices: dict[str, PricePoint] = {}
        self._price_history: list[ConsensusPrice] = []
        self._chainlink_price: Optional[float] = None
        self._chainlink_ts: float = 0
        self._window_anchor: Optional[WindowAnchor] = None
        # RTDS health tracking
        self._rtds_consecutive_failures = 0
        self._rtds_last_success = 0.0
        self._rtds_total_attempts = 0
        self._rtds_total_successes = 0
        # ── Persistent RTDS stream state ──
        self._rtds_stream_running = False
        self._rtds_chainlink_latest: Optional[PricePoint] = None
        self._rtds_binance_latest: Optional[PricePoint] = None
        self._rtds_reconnect_backoff = 5.0  # Start at 5s, exponential up to 120s
        self._rtds_ws: Optional[aiohttp.ClientWebSocketResponse] = None

        # ── Multi-asset: derive oracle params from AssetConfig ──
        if asset_config is not None:
            self._symbol = asset_config.symbol                       # "BTC", "ETH", "SOL"
            self._binance_pair = asset_config.binance_pair           # "BTCUSDT", "ETHUSDT", etc.
            self._binance_ws_stream = asset_config.binance_ws_stream # "btcusdt@kline_1m"
            self._coingecko_id = asset_config.coingecko_id           # "bitcoin", "ethereum", etc.
            self._chainlink_filter = asset_config.chainlink_filter   # "btcusdt" (RTDS Binance filter)
            self._chainlink_symbol = f"{asset_config.symbol.lower()}/usd"  # "btc/usd", "eth/usd"
        else:
            # Backward compat: default to BTC if no AssetConfig provided
            self._symbol = "BTC"
            self._binance_pair = "BTCUSDT"
            self._binance_ws_stream = "btcusdt@kline_1m"
            self._coingecko_id = "bitcoin"
            self._chainlink_filter = "btcusdt"
            self._chainlink_symbol = "btc/usd"

        logger.info(f"[{self._symbol}] OracleEngine initialized — pair={self._binance_pair}, chainlink={self._chainlink_symbol}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def close(self):
        self._rtds_stream_running = False
        if self._rtds_ws and not self._rtds_ws.closed:
            await self._rtds_ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Persistent RTDS Stream ──────────────────────────────────

    async def _rtds_watchdog(self):
        """
        Background watchdog that monitors RTDS stream health.
        If no Chainlink data arrives for 30s, force-closes the WebSocket
        to trigger reconnection in start_rtds_stream.
        """
        STALE_THRESHOLD = 30  # seconds with no data = dead connection
        CHECK_INTERVAL = 10  # check every 10s

        while self._rtds_stream_running:
            await asyncio.sleep(CHECK_INTERVAL)
            if not self._rtds_stream_running:
                break

            last = self._rtds_last_success
            if last <= 0:
                continue  # Haven't received first message yet

            age = time.time() - last
            if age > STALE_THRESHOLD:
                logger.warning(
                    f"🐕 RTDS watchdog: no data for {age:.0f}s (threshold={STALE_THRESHOLD}s) "
                    f"— force-closing WebSocket to trigger reconnect"
                )
                if self._rtds_ws and not self._rtds_ws.closed:
                    try:
                        await self._rtds_ws.close()
                    except Exception:
                        pass
                # Reset last_success so we don't spam close
                self._rtds_last_success = 0.0

        logger.info("🐕 RTDS watchdog stopped")

    async def start_rtds_stream(self):
        """
        Persistent websocket loop that maintains a single connection
        to Polymarket RTDS. Subscribes to both Chainlink and Binance
        topics. Streams prices into _rtds_chainlink_latest and
        _rtds_binance_latest buffers. Reconnects with exponential
        backoff on failure.

        Launch as: asyncio.create_task(oracle.start_rtds_stream())
        """
        self._rtds_stream_running = True
        logger.info(f"🔌 [{self._symbol}] RTDS persistent stream starting...")

        # Launch watchdog to detect stale connections
        watchdog_task = asyncio.create_task(self._rtds_watchdog())

        while self._rtds_stream_running:
            self._rtds_total_attempts += 1
            try:
                session = await self._get_session()
                ws_timeout = aiohttp.ClientTimeout(total=0, sock_connect=10)
                self._rtds_ws = await session.ws_connect(
                    self.RTDS_URL,
                    timeout=ws_timeout.sock_connect,
                    heartbeat=5,  # Per Polymarket docs: ping every 5s
                )

                # Subscribe to topics in SEPARATE messages (some RTDS servers
                # don't handle multiple subscriptions in one message properly)
                await self._rtds_ws.send_json({
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                        "filters": "",
                    }]
                })
                await self._rtds_ws.send_json({
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices",
                        "type": "update",
                        "filters": self._chainlink_filter,
                    }]
                })

                # Reset backoff on successful connect
                self._rtds_reconnect_backoff = 5.0
                self._rtds_consecutive_failures = 0
                logger.info(f"🔌 [{self._symbol}] RTDS connected — subscribed to Chainlink ({self._chainlink_symbol}) + Binance ({self._chainlink_filter})")

                # ── Stream loop — stays open until disconnect ──
                msg_count = 0
                async for msg in self._rtds_ws:
                    if not self._rtds_stream_running:
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        raw = msg.data
                        if not raw or raw.strip() == "":
                            continue

                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        topic = data.get("topic", "")
                        payload = data.get("payload", {})

                        # Log first few messages for diagnostics
                        msg_count += 1
                        if msg_count <= 5:
                            logger.info(f"🔌 RTDS msg[{msg_count}]: topic={topic} keys={list(data.keys())}")

                        if topic == "crypto_prices_chainlink":
                            if payload.get("symbol") == self._chainlink_symbol and "value" in payload:
                                price = float(payload["value"])
                                ts = payload.get("timestamp", time.time() * 1000)
                                if ts > 1e12:
                                    ts = ts / 1000
                                self._rtds_chainlink_latest = PricePoint(
                                    source="chainlink", price=price, timestamp=ts,
                                )
                                self._chainlink_price = price
                                self._chainlink_ts = ts
                                self._rtds_last_success = time.time()
                                self._rtds_total_successes += 1
                                if self._rtds_total_successes % 30 == 1:
                                    logger.info(f"✅ Chainlink {self._chainlink_symbol.upper()}: ${price:,.2f} (RTDS, msg #{self._rtds_total_successes})")

                        elif topic == "crypto_prices":
                            if payload.get("symbol") == self._chainlink_filter and "value" in payload:
                                price = float(payload["value"])
                                ts = payload.get("timestamp", time.time() * 1000)
                                if ts > 1e12:
                                    ts = ts / 1000
                                self._rtds_binance_latest = PricePoint(
                                    source="rtds_binance", price=price, timestamp=ts,
                                )

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        logger.warning(f"🔌 RTDS stream closed: {msg.type}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        logger.warning(f"🔌 RTDS CLOSE frame: code={msg.data}")
                        break

                # Connection ended — will reconnect
                logger.warning(f"🔌 [{self._symbol}] RTDS stream disconnected — will reconnect")

            except aiohttp.WSServerHandshakeError as e:
                logger.warning(f"🔌 RTDS handshake failed — {e.status} {e.message}")
                self._rtds_consecutive_failures += 1
            except asyncio.CancelledError:
                logger.info("🔌 RTDS stream cancelled")
                break
            except Exception as e:
                logger.warning(f"🔌 RTDS stream error: {type(e).__name__}: {e}")
                self._rtds_consecutive_failures += 1

            if not self._rtds_stream_running:
                break

            # Exponential backoff: 5s → 10s → 20s → 40s → 80s → 120s max
            backoff = min(self._rtds_reconnect_backoff, 120.0)
            rate = (self._rtds_total_successes / max(1, self._rtds_total_attempts)) * 100
            logger.info(
                f"🔌 RTDS reconnecting in {backoff:.0f}s "
                f"(failures: {self._rtds_consecutive_failures}, "
                f"lifetime: {rate:.0f}% success)"
            )
            await asyncio.sleep(backoff)
            self._rtds_reconnect_backoff = min(self._rtds_reconnect_backoff * 2, 120.0)

        logger.info(f"🔌 [{self._symbol}] RTDS persistent stream stopped")
        watchdog_task.cancel()

    # ── Chainlink via Persistent RTDS Stream ────────────────────

    async def _fetch_chainlink_rtds(self) -> Optional[PricePoint]:
        """
        Read latest Chainlink price from the persistent RTDS stream buffer.
        No connection made here — the stream loop handles connectivity.
        Falls back to cached price if stream hasn't delivered yet.
        """
        if self._rtds_chainlink_latest and not self._rtds_chainlink_latest.is_stale(self.config.max_price_age):
            return self._rtds_chainlink_latest

        # Check slightly stale cache (up to 60s)
        if self._rtds_chainlink_latest and not self._rtds_chainlink_latest.is_stale(60):
            logger.debug(f"Using cached Chainlink (age: {self._rtds_chainlink_latest.age_seconds:.0f}s)")
            return self._rtds_chainlink_latest

        logger.debug("RTDS Chainlink: no recent price in buffer")
        return None

    # ── RTDS Binance (via Persistent Stream) ───────────────────

    async def _fetch_rtds_binance(self) -> Optional[PricePoint]:
        """
        Read latest RTDS Binance price from the persistent stream buffer.
        No connection made here — the stream loop handles connectivity.
        """
        if self._rtds_binance_latest and not self._rtds_binance_latest.is_stale(self.config.max_price_age):
            return self._rtds_binance_latest

        # Check slightly stale cache (up to 60s)
        if self._rtds_binance_latest and not self._rtds_binance_latest.is_stale(60):
            logger.debug(f"Using cached RTDS Binance (age: {self._rtds_binance_latest.age_seconds:.0f}s)")
            return self._rtds_binance_latest

        return None

    # ── Binance REST ─────────────────────────────────────────────

    async def _fetch_binance(self) -> Optional[PricePoint]:
        try:
            session = await self._get_session()
            url = f"{self.config.binance_base_url}/ticker/bookTicker"
            async with session.get(url, params={"symbol": self._binance_pair}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bid, ask = float(data["bidPrice"]), float(data["askPrice"])
                    return PricePoint(source="binance", price=(bid + ask) / 2, timestamp=time.time(), bid=bid, ask=ask)
                logger.warning(f"Binance {resp.status}")
                return None
        except Exception as e:
            logger.error(f"Binance: {e}")
            return None

    # ── CoinGecko ────────────────────────────────────────────────

    async def _fetch_coingecko(self) -> Optional[PricePoint]:
        try:
            session = await self._get_session()
            url = f"{self.config.coingecko_base_url}/simple/price"
            async with session.get(url, params={"ids": self._coingecko_id, "vs_currencies": "usd"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return PricePoint(source="coingecko", price=data[self._coingecko_id]["usd"], timestamp=time.time())
                logger.warning(f"CoinGecko {resp.status}")
                return None
        except Exception as e:
            logger.error(f"CoinGecko: {e}")
            return None

    # ── Consensus ────────────────────────────────────────────────

    async def get_price(self) -> ConsensusPrice:
        """
        Fetch BTC price from all sources in parallel.
        Chainlink is primary (resolution oracle).
        RTDS Binance + REST Binance + CoinGecko provide redundancy.
        """
        results = await asyncio.gather(
            self._fetch_chainlink_rtds(),
            self._fetch_rtds_binance(),
            self._fetch_binance(),
            self._fetch_coingecko(),
            return_exceptions=True,
        )

        valid: list[PricePoint] = []
        chainlink_pp = None
        for r in results:
            if isinstance(r, PricePoint) and r is not None:
                if not r.is_stale(self.config.max_price_age):
                    valid.append(r)
                    self._last_prices[r.source] = r
                    if r.source == "chainlink":
                        chainlink_pp = r

        # Fallback to cache
        if len(valid) < self.config.min_oracle_consensus:
            for src, pp in self._last_prices.items():
                if not pp.is_stale(60) and pp not in valid:
                    valid.append(pp)
                    logger.warning(f"Using cached {src} (age: {pp.age_seconds:.0f}s)")

        if not valid:
            raise RuntimeError("ALL ORACLES DOWN")

        # Price selection: prefer Chainlink, then RTDS Binance, then median
        prices = [pp.price for pp in valid]
        if chainlink_pp:
            price = chainlink_pp.price
        else:
            # If no Chainlink, check for RTDS Binance (closest to what Polymarket sees)
            rtds_binance_pp = next((pp for pp in valid if pp.source == "rtds_binance"), None)
            if rtds_binance_pp:
                price = rtds_binance_pp.price
            else:
                price = median(prices)

        spread_pct = ((max(prices) - min(prices)) / price) * 100 if len(prices) > 1 else 0.0
        if spread_pct > self.MAX_DIVERGENCE_PCT:
            logger.error(f"Divergence {spread_pct:.3f}%: {', '.join(f'{p.source}=${p.price:,.2f}' for p in valid)}")
            confidence = max(0.2, 1.0 - spread_pct / 5.0)
        else:
            confidence = min(1.0, len(valid) / 3.0)

        consensus = ConsensusPrice(
            price=price,
            timestamp=time.time(),
            sources=[pp.source for pp in valid],
            spread_pct=spread_pct,
            confidence=confidence,
            chainlink_price=chainlink_pp.price if chainlink_pp else None,
        )
        self._price_history.append(consensus)
        logger.info(f"Oracle: {consensus}")
        return consensus

    # ── Window Anchor ────────────────────────────────────────────

    def _current_window_boundary(self, window_minutes: int = 15) -> float:
        """Start of the current N-minute window (the one we're inside)."""
        mins = max(1, int(window_minutes))
        now = time.time()
        dt = datetime.datetime.fromtimestamp(now)
        window_start_min = (dt.minute // mins) * mins
        boundary = dt.replace(minute=window_start_min, second=0, microsecond=0)
        return boundary.timestamp()

    async def capture_window_open(self, window_minutes: int = 15) -> WindowAnchor:
        """
        Capture the opening price of the current N-minute window.
        This is the price Polymarket uses as the reference —
        end_price >= open_price → UP wins.

        Should be called right at or just after the boundary.
        """
        boundary_ts = self._current_window_boundary(window_minutes=window_minutes)

        # If we already have an anchor for this window, return it
        if self._window_anchor and self._window_anchor.boundary_time == boundary_ts:
            return self._window_anchor

        # Fetch fresh price — Chainlink preferred
        consensus = await self.get_price()

        source = "chainlink" if consensus.chainlink_price else consensus.sources[0]
        open_price = consensus.chainlink_price or consensus.price

        self._window_anchor = WindowAnchor(
            boundary_time=boundary_ts,
            open_price=open_price,
            source=source,
            captured_at=time.time(),
        )

        boundary_dt = datetime.datetime.fromtimestamp(boundary_ts)
        logger.info(
            f"📌 Window anchor: ${open_price:,.2f} ({source}) "
            f"for {boundary_dt.strftime('%H:%M')} ({window_minutes}m) window"
        )
        return self._window_anchor

    def get_window_anchor(self) -> Optional[WindowAnchor]:
        """Return the current window's opening price anchor."""
        return self._window_anchor

    # ── Candles ──────────────────────────────────────────────────

    async def get_candles(self, interval: str = "15m", limit: int = 100) -> list[Candle]:
        """Fetch historical candles from Binance (best candle source)."""
        try:
            session = await self._get_session()
            url = f"{self.config.binance_base_url}/klines"
            params = {"symbol": self._binance_pair, "interval": interval, "limit": min(limit, 1000)}
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Binance klines {resp.status}")
                data = await resp.json()
            return [
                Candle(
                    timestamp=k[0] / 1000, open=float(k[1]),
                    high=float(k[2]), low=float(k[3]),
                    close=float(k[4]), volume=float(k[5]),
                    interval=interval,
                )
                for k in data
            ]
        except Exception as e:
            logger.error(f"Candles: {e}")
            return []

    def get_price_history(self) -> list[ConsensusPrice]:
        return self._price_history.copy()

    def get_rtds_health(self) -> dict:
        """Return RTDS connection health stats for dashboard/logging."""
        return {
            "consecutive_failures": self._rtds_consecutive_failures,
            "last_success_ago_secs": round(time.time() - self._rtds_last_success, 1) if self._rtds_last_success > 0 else None,
            "lifetime_attempts": self._rtds_total_attempts,
            "lifetime_successes": self._rtds_total_successes,
            "success_rate_pct": round((self._rtds_total_successes / max(1, self._rtds_total_attempts)) * 100, 1),
        }
