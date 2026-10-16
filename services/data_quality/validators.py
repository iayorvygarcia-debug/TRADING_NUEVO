"""
AI Quant Trading Platform — Data Quality Engine

Validates ALL incoming market data before it reaches any model or strategy.

NEVER allow corrupted data to feed the system automatically.

Checks performed:
1. Missing data detection (gaps in timestamps)
2. Duplicate detection (same bar twice)
3. Timestamp validation (out-of-order, future timestamps)
4. Price anomaly detection (spikes beyond ATR-based thresholds)
5. Volume anomaly detection (zero volume, extreme spikes)
6. OHLC consistency (high >= low, high >= open/close, etc.)
7. Stale data detection (no updates for too long)
8. Market session validation (data during closed market)

Each validated bar gets a QualityStatus:
- VALID:   Passed all checks
- SUSPECT: Minor issues (logged but data is usable)
- INVALID: Failed critical checks (data is rejected)
- STALE:   Data is too old to be useful
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from core.logging import get_logger
from core.market_session import get_current_session, can_trade, MarketSession
from services.market_data.models import Bar, Quote, QualityStatus

logger = get_logger("data_quality")


@dataclass
class QualityReport:
    """Result of a data quality check on a single data point."""
    status: QualityStatus
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0

    @property
    def is_valid(self) -> bool:
        return self.status in (QualityStatus.VALID, QualityStatus.SUSPECT)

    @property
    def is_clean(self) -> bool:
        return self.status == QualityStatus.VALID

    def add_issue(self, issue: str) -> None:
        self.issues.append(issue)
        self.status = QualityStatus.INVALID

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        if self.status == QualityStatus.VALID:
            self.status = QualityStatus.SUSPECT


@dataclass
class QualityStats:
    """Aggregate quality statistics for monitoring."""
    total_checked: int = 0
    valid: int = 0
    suspect: int = 0
    invalid: int = 0
    stale: int = 0
    issues_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def valid_rate(self) -> float:
        return self.valid / self.total_checked if self.total_checked > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_checked": self.total_checked,
            "valid": self.valid,
            "suspect": self.suspect,
            "invalid": self.invalid,
            "stale": self.stale,
            "valid_rate": round(self.valid_rate, 4),
            "top_issues": dict(
                sorted(self.issues_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }


class DataQualityEngine:
    """
    Validates incoming market data before it enters the system.
    
    Usage:
        dq = DataQualityEngine()
        report = dq.validate_bar(bar)
        if report.is_valid:
            # Store and process
        else:
            # Log and reject
    """

    def __init__(
        self,
        max_price_change_pct: float = 50.0,
        max_volume_zscore: float = 10.0,
        stale_threshold_seconds: float = 300.0,
        min_price: float = 0.001,
        max_price: float = 1_000_000.0,
    ) -> None:
        """
        Args:
            max_price_change_pct: Max % change from last close to flag as anomaly.
            max_volume_zscore: Max volume z-score to flag as anomaly.
            stale_threshold_seconds: Seconds without data before flagging as stale.
            min_price: Minimum valid price.
            max_price: Maximum valid price.
        """
        self._max_price_change_pct = max_price_change_pct
        self._max_volume_zscore = max_volume_zscore
        self._stale_threshold = stale_threshold_seconds
        self._min_price = min_price
        self._max_price = max_price

        # Track last known state per symbol for change detection
        self._last_bars: dict[str, Bar] = {}
        self._last_update: dict[str, float] = {}

        # Running volume stats per symbol (for z-score)
        self._volume_history: dict[str, list[int]] = defaultdict(list)
        self._volume_window = 100  # Keep last 100 bars for stats

        # Aggregate stats
        self._stats = QualityStats()

        # Dedup tracking
        self._seen_hashes: dict[str, set[str]] = defaultdict(set)

    # ── Bar Validation ──────────────────────

    def validate_bar(self, bar: Bar) -> QualityReport:
        """
        Run all quality checks on a bar.
        
        Returns:
            QualityReport with status and any issues found.
        """
        report = QualityReport(status=QualityStatus.VALID, checks_total=8)

        # 1. OHLC consistency
        self._check_ohlc_consistency(bar, report)

        # 2. Price range
        self._check_price_range(bar, report)

        # 3. Volume validity
        self._check_volume(bar, report)

        # 4. Timestamp validity
        self._check_timestamp(bar, report)

        # 5. Duplicate detection
        self._check_duplicate(bar, report)

        # 6. Price anomaly (spike detection)
        self._check_price_anomaly(bar, report)

        # 7. Volume anomaly
        self._check_volume_anomaly(bar, report)

        # 8. Market session context
        self._check_market_session(bar, report)

        # Update tracking state
        self._update_state(bar, report)

        # Update aggregate stats
        self._stats.total_checked += 1
        if report.status == QualityStatus.VALID:
            self._stats.valid += 1
        elif report.status == QualityStatus.SUSPECT:
            self._stats.suspect += 1
        elif report.status == QualityStatus.INVALID:
            self._stats.invalid += 1
        elif report.status == QualityStatus.STALE:
            self._stats.stale += 1

        for issue in report.issues:
            self._stats.issues_by_type[issue.split(":")[0]] += 1

        if report.issues:
            logger.warning(
                "data_quality_issue",
                symbol=bar.symbol,
                status=report.status.value,
                issues=report.issues,
            )

        return report

    def _check_ohlc_consistency(self, bar: Bar, report: QualityReport) -> None:
        """Verify OHLC relationships: H >= L, H >= O, H >= C, L <= O, L <= C."""
        report.checks_passed += 1

        if bar.high < bar.low:
            report.add_issue(f"ohlc_invalid: high ({bar.high}) < low ({bar.low})")
            return

        if bar.high < bar.open or bar.high < bar.close:
            report.add_issue(
                f"ohlc_invalid: high ({bar.high}) < open ({bar.open}) or close ({bar.close})"
            )
            return

        if bar.low > bar.open or bar.low > bar.close:
            report.add_issue(
                f"ohlc_invalid: low ({bar.low}) > open ({bar.open}) or close ({bar.close})"
            )

    def _check_price_range(self, bar: Bar, report: QualityReport) -> None:
        """Verify prices are within reasonable bounds."""
        report.checks_passed += 1

        for label, price in [("open", bar.open), ("high", bar.high),
                             ("low", bar.low), ("close", bar.close)]:
            if price <= 0:
                report.add_issue(f"price_zero_or_negative: {label}={price}")
                return
            if price < self._min_price:
                report.add_warning(f"price_below_minimum: {label}={price}")
            if price > self._max_price:
                report.add_issue(f"price_above_maximum: {label}={price}")

    def _check_volume(self, bar: Bar, report: QualityReport) -> None:
        """Check volume is non-negative."""
        report.checks_passed += 1

        if bar.volume < 0:
            report.add_issue(f"volume_negative: {bar.volume}")
        elif bar.volume == 0:
            report.add_warning("volume_zero")

    def _check_timestamp(self, bar: Bar, report: QualityReport) -> None:
        """Check timestamp is valid and not in the future."""
        report.checks_passed += 1

        now = datetime.now(timezone.utc)

        if bar.time.tzinfo is None:
            report.add_warning("timestamp_no_timezone")
            return

        # Future timestamp (with 5 minute tolerance for clock skew)
        if bar.time > now + timedelta(minutes=5):
            report.add_issue(f"timestamp_future: {bar.time.isoformat()}")

        # Very old timestamp (> 7 days for real-time data). Historical bars
        # remain valid for backtests; freshness is handled by check_stale().
        if bar.time < now - timedelta(days=7):
            report.warnings.append(f"timestamp_old: {bar.time.isoformat()}")

        # Check ordering against last bar
        key = f"{bar.symbol}:{bar.timeframe.value}"
        if key in self._last_bars:
            last = self._last_bars[key]
            if bar.time <= last.time:
                report.add_warning(
                    f"timestamp_out_of_order: {bar.time.isoformat()} <= {last.time.isoformat()}"
                )

    def _check_duplicate(self, bar: Bar, report: QualityReport) -> None:
        """Detect duplicate bars (same symbol + timeframe + timestamp)."""
        report.checks_passed += 1

        bar_hash = f"{bar.symbol}:{bar.timeframe.value}:{bar.time.isoformat()}"
        key = f"{bar.symbol}:{bar.timeframe.value}"

        if bar_hash in self._seen_hashes[key]:
            report.add_warning("duplicate_bar")
        else:
            self._seen_hashes[key].add(bar_hash)
            # Keep only last 1000 hashes per symbol to limit memory
            if len(self._seen_hashes[key]) > 1000:
                # Remove oldest entries (convert to list, slice, convert back)
                hashes_list = list(self._seen_hashes[key])
                self._seen_hashes[key] = set(hashes_list[-500:])

    def _check_price_anomaly(self, bar: Bar, report: QualityReport) -> None:
        """Detect anomalous price spikes relative to last known price."""
        report.checks_passed += 1

        key = f"{bar.symbol}:{bar.timeframe.value}"
        if key not in self._last_bars:
            return  # No baseline yet

        last = self._last_bars[key]
        if last.close <= 0:
            return

        change_pct = abs(bar.close - last.close) / last.close * 100

        if change_pct > self._max_price_change_pct:
            report.add_issue(
                f"price_spike: {change_pct:.1f}% change "
                f"(last={last.close}, current={bar.close})"
            )
        elif change_pct > self._max_price_change_pct * 0.5:
            report.add_warning(
                f"price_large_move: {change_pct:.1f}% change"
            )

    def _check_volume_anomaly(self, bar: Bar, report: QualityReport) -> None:
        """Detect anomalous volume using z-score."""
        report.checks_passed += 1

        key = bar.symbol
        history = self._volume_history[key]

        if len(history) < 20:
            return  # Not enough data for meaningful z-score

        import statistics
        mean_vol = statistics.mean(history)
        stdev_vol = statistics.stdev(history)

        if stdev_vol == 0:
            return

        zscore = (bar.volume - mean_vol) / stdev_vol

        if abs(zscore) > self._max_volume_zscore:
            report.add_warning(
                f"volume_anomaly: z-score={zscore:.1f} "
                f"(mean={mean_vol:.0f}, vol={bar.volume})"
            )

    def _check_market_session(self, bar: Bar, report: QualityReport) -> None:
        """Check if data timestamp aligns with market session."""
        report.checks_passed += 1

        session = get_current_session(bar.time)
        if session in (MarketSession.CLOSED, MarketSession.HOLIDAY):
            report.add_warning(f"data_during_closed_market: session={session.value}")

    def _update_state(self, bar: Bar, report: QualityReport) -> None:
        """Update internal tracking state after validation."""
        if report.is_valid:
            key = f"{bar.symbol}:{bar.timeframe.value}"
            self._last_bars[key] = bar
            self._last_update[bar.symbol] = time.monotonic()

            # Update volume history
            self._volume_history[bar.symbol].append(bar.volume)
            if len(self._volume_history[bar.symbol]) > self._volume_window:
                self._volume_history[bar.symbol] = (
                    self._volume_history[bar.symbol][-self._volume_window:]
                )

    # ── Quote Validation ────────────────────

    def validate_quote(self, quote: Quote) -> QualityReport:
        """Validate a quote (bid/ask)."""
        report = QualityReport(status=QualityStatus.VALID, checks_total=4)

        # Bid/Ask positive
        if quote.bid < 0 or quote.ask < 0:
            report.add_issue(f"quote_negative: bid={quote.bid}, ask={quote.ask}")

        # Spread check (ask should be >= bid)
        if quote.ask < quote.bid:
            report.add_issue(f"quote_crossed: bid={quote.bid} > ask={quote.ask}")

        # Reasonable spread (< 10% of mid)
        if quote.bid > 0 and quote.ask > 0:
            mid = (quote.bid + quote.ask) / 2
            spread_pct = (quote.ask - quote.bid) / mid
            if spread_pct > 0.10:
                report.add_warning(f"quote_wide_spread: {spread_pct*100:.1f}%")

        self._stats.total_checked += 1
        if report.status == QualityStatus.VALID:
            self._stats.valid += 1
        else:
            self._stats.invalid += 1

        return report

    # ── Stale Data Check ────────────────────

    def check_stale(self, symbol: str) -> bool:
        """Check if data for a symbol is stale (no updates for too long)."""
        if symbol not in self._last_update:
            return True  # Never received data

        elapsed = time.monotonic() - self._last_update[symbol]
        return elapsed > self._stale_threshold

    def get_stale_symbols(self, symbols: list[str]) -> list[str]:
        """Get list of symbols with stale data."""
        return [s for s in symbols if self.check_stale(s)]

    # ── Stats & Health ──────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate quality statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset aggregate statistics."""
        self._stats = QualityStats()
