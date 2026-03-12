# GRIDPHANTOMDEV — Multi-Asset Directional

Multi-asset directional trading engine for Polymarket UP/DOWN binary markets.

Trades **BTC, ETH, SOL** across **5m, 15m, and 1hr** timeframes — up to 9 independent trading loops running simultaneously.

Built on [BTC-15M-Oracle](https://github.com/jaboofin/directionalbtcv1). Expanded from BTC-only to full multi-asset with per-asset risk isolation, correlation guards, and a live dashboard.

## How It Works

Every window boundary (5m / 15m / 1hr), each enabled asset loop:

1. **Captures the Chainlink anchor price** at the window boundary via persistent RTDS websocket
2. **Waits for price drift** (45-120s depending on timeframe)
3. **Runs the 5-signal strategy** — price_vs_open (dominant), RSI, MACD, momentum, EMA cross — with per-asset indicator tuning and dead zones
4. **Checks risk gates** — per-asset budget, drawdown guard (global), regime filter (per-asset), global position limit (40% max deployed), correlation guard
5. **Executes via Polymarket CLOB** — FOK with GTC fallback, fill verification, fee-aware sizing
6. **Monitors positions** — early exit system (TP/SL/trailing stop) polls CLOB bids every 5s
7. **Resolves** when the window closes — UP if Chainlink close >= open, else DOWN. Auto-redeemed by Polymarket.

## Quickstart

```bash
pip install -r requirements.txt

export POLY_PRIVATE_KEY="your_private_key"
export POLY_FUNDER="0xYourProfileAddress"
export POLY_SIG_TYPE=1

# BTC 15m only
python bot.py --bankroll 100

# BTC 15m + 5m + 1hr
python bot.py --bankroll 100 --5m --1h

# All assets, all timeframes, full stack
python bot.py --bankroll 300 --5m --1h --eth --sol --early-exit --dashboard

# Fixed cycles with dashboard
python bot.py --bankroll 100 --cycles 10 --5m --1h --eth --sol --dashboard
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bankroll` | 500 | Starting capital in USD |
| `--cycles` | 0 | Max entry windows (0 = run forever) |
| `--5m` | off | Enable parallel 5-minute trading loops |
| `--1h` | off | Enable parallel 1-hour trading loops |
| `--eth` | off | Enable ETH directional trading |
| `--sol` | off | Enable SOL directional trading |
| `--early-exit` | off | Enable TP/SL/trailing stop position management |
| `--early-exit-dry-run` | off | Log exits but don't sell (paper mode) |
| `--dashboard` | off | Start live dashboard on http://localhost:8765 |
| `--sync-live-bankroll` | off | Sync bankroll from live Polymarket CLOB balance |
| `--live-bankroll-poll-secs` | 60 | Live bankroll sync interval |
| `--strategy-delay` | 45 | Override seconds to wait after anchor capture |
| `--acknowledge-drawdown` | off | Resume trading after RED drawdown halt |
| `--no-drawdown-guard` | off | Disable drawdown guard (not recommended) |
| `--no-regime-filter` | off | Disable regime filter |
| `--no-sizer` | off | Use flat Kelly sizing instead of confidence-scaled |

## Multi-Asset Architecture

With `--eth --sol`, the bot runs per-asset instances of every core component:

| Component | Scope | Details |
|-----------|-------|---------|
| Oracle Engine | Per-asset | Independent Chainlink RTDS + Binance + CoinGecko feeds per asset |
| Strategy Engine | Per-asset | Per-asset dead zones (BTC 0.04%, ETH 0.05%, SOL 0.03%), confidence thresholds, volatility ceilings |
| Risk Manager | Per-asset | Independent budgets — BTC 50%, ETH 30%, SOL 20% of bankroll. Bad SOL streak can't drain BTC budget |
| Regime Filter | Per-asset | BTC can be TRENDING while ETH is CHOPPY — each gets its own candle feed |
| Early Exit | Per-asset | SOL gets tighter trail offset ($0.05 vs BTC $0.06) due to higher volatility |
| Drawdown Guard | Global | Protects total capital regardless of which asset caused the drawdown |
| Correlation Guard | Global | Cuts size 50% when all 3 assets open same direction, 25% for 2 of 3 |
| Position Limit | Global | Max 40% of total bankroll deployed at once across all assets |

## Capital Allocation Example ($300 bankroll)

```
BTC: $150 (50%) — trades 5m + 15m + 1hr
ETH: $90  (30%) — trades 5m + 15m + 1hr
SOL: $60  (20%) — trades 5m + 15m + 1hr

Max deployed at once: $120 (40% of $300)
9 independent loops, each with own risk budget
```

## Dashboard

`--dashboard` launches a live command center at http://localhost:8765 with:

- Multi-price ticker (BTC/ETH/SOL live from RTDS)
- Per-asset summary strip (capital, P&L, regime state, trade count)
- Active positions with TP/SL/trail threshold bars and live bid tracking
- Strategy signals with confidence scoring breakdown
- Early exit stats (TP/SL/trail counts and P&L)
- Risk panel with drawdown tier, loss streaks, cooldowns
- Activity feed with asset-tagged trade history
- Demo mode for stream display when bot isn't connected

Stream-optimized: high contrast dark theme, JetBrains Mono, large readable numbers.

## Project Structure

```
directionalv3/
├── bot.py                    Main orchestrator — 9-loop multi-asset engine
├── config/
│   └── settings.py           AssetConfig, Active5mConfig, Active1hConfig, all params
├── core/
│   ├── polymarket_client.py  CLOB SDK — multi-asset slug discovery, orders, fees
│   ├── risk_manager.py       Kelly sizing, per-asset budgets, 15m/5m independent tracking
│   ├── early_exit.py         TP/SL/trailing stop with per-asset offset overrides
│   ├── regime_filter.py      DCR + Normalized Range regime classification
│   ├── drawdown_guard.py     HWM-persisted 4-tier kill switch (GREEN/YELLOW/ORANGE/RED)
│   ├── position_sizer.py     Confidence-scaled sizing with bankroll tiers
│   ├── dashboard_server.py   HTTP + WebSocket server with multi-asset state broadcast
│   └── trade_logger.py       Structured JSONL logging
├── oracles/
│   └── price_feed.py         Per-asset RTDS streams + Chainlink/Binance/CoinGecko
├── strategies/
│   └── signal_engine.py      Drift-dominant 5-signal strategy with per-asset overrides
├── dashboard.html            Multi-asset command center frontend
├── tests/                    Test suites
├── logs/                     Runtime logs
└── data/                     Performance snapshots + HWM state
```

## Safety Systems

**Drawdown Guard** — 4-tier kill switch based on high water mark. GREEN (full size) → YELLOW (50% size, -15% from HWM) → ORANGE (25% size, high confidence only, -25% from HWM) → RED (halt trading, -40% from HWM). Persists HWM across restarts.

**Correlation Guard** — BTC, ETH, SOL are correlated. When all 3 have open positions in the same direction, new sizes cut 50%. When 2 of 3 match, cut 25%. Prevents triple-sized drawdowns on correlated dumps.

**Global Position Limit** — Never more than 40% of total bankroll in open positions simultaneously. Prevents over-exposure when many signals fire at shared boundaries (e.g. :00 when all 9 loops activate).

**Regime Filter** — Classifies market as TRENDING/CHOPPY/UNKNOWN using Directional Change Ratio and Normalized Range. Blocks entries during CHOPPY. Requires high conviction override during UNKNOWN. Per-asset — BTC can trade while ETH sits out.

**Per-Asset Risk Isolation** — Each asset has its own daily trade limit, loss streak tracking, cooldown timer, and capital allocation. A SOL losing streak triggers SOL cooldown only — BTC and ETH continue trading.

## Environment Variables

```bash
POLY_PRIVATE_KEY    # Polymarket wallet private key
POLY_FUNDER         # Profile address (for email/Magic Link accounts)
POLY_SIG_TYPE       # 0=EOA, 1=email/proxy (most users), 2=contract
POLYGON_RPC_URL     # Optional: custom Polygon RPC (default: polygon-rpc.com)
```

## Key Design Decisions

- **Email/Magic Link accounts**: Use profile address as funder, signature type 1, let SDK auto-resolve neg_risk
- **Agreement-weighted confidence**: Drift magnitude as base, indicator agreement as adjustment — produces natural spread (0.55-0.92) where agreement matters
- **Per-timeframe indicator tuning**: 5m uses RSI 7, MACD 6/13/5, EMA 3/8. 1hr uses RSI 21, MACD 12/26/9, EMA 8/21. Reusing 15m params across timeframes degrades signal quality.
- **Independent market resolution**: 5m trade at :15 and 15m trade at :15 target different Polymarket markets with independent resolutions — both fire, no skip needed
- **Auto-redemption**: Short-duration BTC/ETH/SOL markets on Polymarket auto-redeem — no sell logic needed for resolution, only for early exit
- **If early exit sell fails**: Position falls through to normal resolution. Capital is never stuck.
