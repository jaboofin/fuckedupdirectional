"""
╔══════════════════════════════════════════════════════════════════════╗
║  BTC-Directional — CONFIGURATION                                     ║
║                                                                      ║
║  15m + 5m directional · Early Exit · Regime Filter                   ║
║  Drawdown Guard · Confidence-Scaled Sizing                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
from dataclasses import dataclass, field
from enum import Enum


class MarketDirection(Enum):
    UP = "up"
    DOWN = "down"
    HOLD = "hold"


@dataclass
class AssetConfig:
    """Per-asset identity and oracle sources for multi-asset trading."""
    symbol: str = "BTC"
    binance_pair: str = "BTCUSDT"
    binance_ws_stream: str = "btcusdt@kline_1m"
    coingecko_id: str = "bitcoin"
    chainlink_filter: str = "btcusdt"          # RTDS subscription filter
    slug_prefix: str = "btc-updown"            # Polymarket slug: {prefix}-{tf}-{ts}
    timeframes: list = field(default_factory=lambda: ["5m", "15m", "1h"])
    budget_pct: float = 50.0                   # % of total bankroll allocated
    enabled: bool = True

    # ── Per-asset strategy overrides (Phase C — None = use global defaults) ──
    dead_zone_pct: float = 0.04
    confidence_threshold: float = 0.72
    max_volatility_pct: float = 3.0
    tp_offset: float = 0.10
    sl_offset: float = 0.08
    trail_offset: float = 0.06


# ── Pre-built asset configs ──────────────────────────────────────────

def default_btc_asset() -> AssetConfig:
    return AssetConfig(
        symbol="BTC",
        binance_pair="BTCUSDT",
        binance_ws_stream="btcusdt@kline_1m",
        coingecko_id="bitcoin",
        chainlink_filter="btcusdt",
        slug_prefix="btc-updown",
        timeframes=["5m", "15m", "1h"],
        budget_pct=50.0,
        enabled=True,
        dead_zone_pct=0.04,
        confidence_threshold=0.72,
        max_volatility_pct=3.0,
        tp_offset=0.10,
        sl_offset=0.08,
        trail_offset=0.06,
    )


def default_eth_asset() -> AssetConfig:
    return AssetConfig(
        symbol="ETH",
        binance_pair="ETHUSDT",
        binance_ws_stream="ethusdt@kline_1m",
        coingecko_id="ethereum",
        chainlink_filter="ethusdt",
        slug_prefix="eth-updown",
        timeframes=["5m", "15m", "1h"],
        budget_pct=30.0,
        enabled=False,
        dead_zone_pct=0.05,
        confidence_threshold=0.72,
        max_volatility_pct=4.0,
        tp_offset=0.10,
        sl_offset=0.08,
        trail_offset=0.06,
    )


def default_sol_asset() -> AssetConfig:
    return AssetConfig(
        symbol="SOL",
        binance_pair="SOLUSDT",
        binance_ws_stream="solusdt@kline_1m",
        coingecko_id="solana",
        chainlink_filter="solusdt",
        slug_prefix="sol-updown",
        timeframes=["5m", "15m", "1h"],
        budget_pct=20.0,
        enabled=False,
        dead_zone_pct=0.03,
        confidence_threshold=0.74,
        max_volatility_pct=5.0,
        tp_offset=0.10,
        sl_offset=0.08,
        trail_offset=0.05,
    )


@dataclass
class OracleConfig:
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    binance_base_url: str = "https://api.binance.com/api/v3"
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    coincap_base_url: str = "https://api.coincap.io/v2"
    poll_interval: int = 10
    max_price_age: int = 30
    min_oracle_consensus: int = 2
    history_candle_count: int = 100
    candle_interval: str = "15m"


@dataclass
class PolymarketConfig:
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137
    rpc_url: str = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
    private_key: str = os.getenv("POLY_PRIVATE_KEY", "")
    funder: str = os.getenv("POLY_FUNDER", "")
    sig_type: int = int(os.getenv("POLY_SIG_TYPE", "0"))
    market_slug_pattern: str = "btc-price"
    market_interval_minutes: int = 15
    order_type: str = "market"
    max_slippage_pct: float = 2.0
    min_liquidity_usd: float = 50.0
    sync_live_bankroll: bool = False
    live_bankroll_poll_secs: int = 60
    # ── Fee handling ──
    fee_cache_ttl_secs: int = 60
    fee_fallback_pct: float = 1.56


@dataclass
class StrategyConfig:
    confidence_threshold: float = 0.72
    strong_signal_threshold: float = 0.75
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    ema_fast: int = 5
    ema_slow: int = 15
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_lookback: int = 3
    min_volatility_pct: float = 0.03
    max_volatility_pct: float = 3.0
    weight_momentum: float = 0.30
    weight_rsi: float = 0.25
    weight_macd: float = 0.25
    weight_ema_cross: float = 0.20


@dataclass
class RiskConfig:
    max_trade_pct: float = 5.0
    max_daily_trades: int = 20
    max_daily_loss_pct: float = 25.0
    max_consecutive_losses: int = 5
    loss_streak_cooldown_mins: int = 60
    kelly_fraction: float = 0.25
    min_trade_size_usd: float = 1.0
    max_trade_size_usd: float = 25.0


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    trade_log_file: str = "logs/trades.jsonl"
    strategy_log_file: str = "logs/strategy.jsonl"
    oracle_log_file: str = "logs/oracle.jsonl"
    error_log_file: str = "logs/errors.log"
    performance_file: str = "data/performance.json"
    alert_on_loss_streak: int = 3
    alert_on_oracle_downtime_secs: int = 60


@dataclass
class EarlyExitConfig:
    """Take Profit + Stop Loss + Trailing Stop position management."""
    # Master toggles
    enabled: bool = False                 # --early-exit to turn on
    dry_run: bool = False                 # --early-exit-dry-run: log but don't sell

    # Poll
    poll_interval_secs: float = 5.0       # 3-15: how often to check CLOB prices
    min_hold_secs: float = 30.0           # 0-120: minimum hold before any exit

    # ── Take Profit ──
    tp_enabled: bool = True
    tp_offset: float = 0.10               # 0.05-0.25: TP = entry + offset (before conviction scaling)
    tp_min_net_profit: float = 0.02       # 0.01-0.05: minimum net profit/share after round-trip fees
    tp_conviction_scale: bool = True      # Scale TP target by conviction tier

    # ── Stop Loss ──
    sl_enabled: bool = True
    sl_offset: float = 0.08               # 0.04-0.15: SL = entry - offset
    sl_time_decay: bool = True            # Tighten SL as market window approaches close
    sl_min_price: float = 0.10            # 0.05-0.20: never set SL below this (avoid extreme slippage)

    # ── Trailing Stop ──
    trail_enabled: bool = True
    trail_offset: float = 0.06            # 0.03-0.12: trail distance from peak
    trail_activation: float = 0.04        # 0.02-0.10: trail only activates after this much above entry
    trail_tighten_near_close: bool = True  # Tighten trail in final 60s

    def __post_init__(self):
        """Validate parameter ranges per spec."""
        self.poll_interval_secs = max(3.0, min(15.0, self.poll_interval_secs))
        self.min_hold_secs = max(0.0, min(120.0, self.min_hold_secs))
        self.tp_offset = max(0.05, min(0.25, self.tp_offset))
        self.tp_min_net_profit = max(0.01, min(0.05, self.tp_min_net_profit))
        self.sl_offset = max(0.04, min(0.15, self.sl_offset))
        self.sl_min_price = max(0.05, min(0.20, self.sl_min_price))
        self.trail_offset = max(0.03, min(0.12, self.trail_offset))
        self.trail_activation = max(0.02, min(0.10, self.trail_activation))


@dataclass
class RegimeConfig:
    """Regime Detection Filter — adapted from GRIDPHANTOM-TARB M1."""
    enabled: bool = True
    # DCR thresholds
    dcr_trend_threshold: float = 0.65
    dcr_chop_threshold: float = 0.55
    # Normalized Range thresholds
    nr_trend_threshold: float = 1.5
    nr_chop_threshold: float = 0.8
    # Lookback
    lookback_candles: int = 30
    min_candles: int = 10
    # Baseline volatility EMA
    baseline_ema_alpha: float = 0.1
    baseline_window: int = 20
    # Override
    regime_override_conviction: float = 0.95

    def validate(self) -> None:
        """Validate config parameters are within acceptable ranges."""
        assert 0.50 <= self.dcr_chop_threshold <= 0.60
        assert 0.55 <= self.dcr_trend_threshold <= 0.80
        assert self.dcr_chop_threshold < self.dcr_trend_threshold
        assert 0.5 <= self.nr_chop_threshold <= 1.2
        assert 1.0 <= self.nr_trend_threshold <= 2.5
        assert self.nr_chop_threshold < self.nr_trend_threshold
        assert 15 <= self.lookback_candles <= 60
        assert 0.05 <= self.baseline_ema_alpha <= 0.25
        assert 10 <= self.baseline_window <= 50
        assert 0.90 <= self.regime_override_conviction <= 0.99


@dataclass
class DrawdownConfig:
    """Cumulative Drawdown Kill Switch — adapted from GRIDPHANTOM-TARB M3."""
    enabled: bool = True
    yellow_threshold: float = 0.15
    orange_threshold: float = 0.25
    red_threshold: float = 0.40
    yellow_size_mult: float = 0.50
    orange_size_mult: float = 0.25
    orange_min_confidence: float = 0.90
    yellow_recovery: float = 0.10
    orange_recovery: float = 0.15
    hwm_state_path: str = "data/hwm_state.json"
    halted_file_path: str = "data/HALTED"


@dataclass
class PositionSizerConfig:
    """Confidence-Scaled Position Sizing — adapted from GRIDPHANTOM-TARB M4."""
    enabled: bool = True
    per_market_budget_pct: float = 0.15       # Full base % (matches TARB live config)
    small_bankroll_budget_pct: float = 0.08   # Reduced base % for < $200 (matches TARB)
    small_bankroll_cap: float = 200.0
    mid_bankroll_cap: float = 500.0
    min_order_usd: float = 1.0
    max_order_usd: float = 50.0


@dataclass
class Active5mConfig:
    """Parallel 5-minute directional trading loop."""
    enabled: bool = False                 # --5m to turn on
    # ── Budget (separate from 15m) ──
    budget_pct: float = 50.0
    max_daily_trades: int = 30
    max_trade_size_usd: float = 3.0
    max_daily_loss_pct: float = 15.0
    max_consecutive_losses: int = 4
    loss_streak_cooldown_mins: int = 30
    # ── Timing ──
    strategy_delay_secs: int = 45
    entry_lead_secs: int = 55
    entry_window_secs: int = 20
    # ── 5m-tuned indicator params ──
    rsi_period: int = 7
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 5
    ema_fast: int = 3
    ema_slow: int = 8
    momentum_lookback: int = 2
    weight_price_vs_open: float = 0.50


@dataclass
class Active1hConfig:
    """Parallel 1-hour directional trading loop."""
    enabled: bool = False                 # --1h to turn on
    # ── Budget (separate from 15m/5m) ──
    budget_pct: float = 30.0
    max_daily_trades: int = 8
    max_trade_size_usd: float = 15.0
    max_daily_loss_pct: float = 15.0
    max_consecutive_losses: int = 3
    loss_streak_cooldown_mins: int = 60
    # ── Timing ──
    strategy_delay_secs: int = 60
    entry_lead_secs: int = 120
    entry_window_secs: int = 45
    # ── 1hr-tuned indicator params ──
    rsi_period: int = 21
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast: int = 8
    ema_slow: int = 21
    momentum_lookback: int = 4
    weight_price_vs_open: float = 0.60


@dataclass
class BotConfig:
    oracle: OracleConfig = field(default_factory=OracleConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    active_5m: Active5mConfig = field(default_factory=Active5mConfig)
    active_1h: Active1hConfig = field(default_factory=Active1hConfig)
    early_exit: EarlyExitConfig = field(default_factory=EarlyExitConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    sizer: PositionSizerConfig = field(default_factory=PositionSizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # ── Multi-asset configs (Phase A) ──
    assets: list = field(default_factory=lambda: [
        default_btc_asset(),
        default_eth_asset(),
        default_sol_asset(),
    ])
    bot_name: str = "BTC-Directional"
    version: str = "1.2.0"
    bankroll: float = 500.0
    entry_lead_secs: int = 60
    entry_window_secs: int = 30
    sleep_poll_secs: int = 5
    strategy_delay_secs: int = 45

    def enabled_assets(self) -> list:
        """Return only the AssetConfig entries that are enabled."""
        return [a for a in self.assets if a.enabled]

    def get_asset(self, symbol: str) -> 'AssetConfig | None':
        """Lookup an asset config by symbol (e.g. 'BTC', 'ETH', 'SOL')."""
        for a in self.assets:
            if a.symbol.upper() == symbol.upper():
                return a
        return None
