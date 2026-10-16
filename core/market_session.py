"""
AI Quant Trading Platform — Market Session Engine

Determines the current state of the market for each instrument.
The bot runs 24/7 but must know when trading is actually possible.

States:
- PREMARKET:    04:00-09:30 ET — Limited liquidity, wider spreads
- REGULAR:      09:30-16:00 ET — Full liquidity, normal spreads
- AFTERHOURS:   16:00-20:00 ET — Limited liquidity, wider spreads
- CLOSED:       20:00-04:00 ET — No trading possible
- HOLIDAY:      Full day closure (NYSE/NASDAQ calendar)
- EARLY_CLOSE:  Market closes at 13:00 ET (day before holidays)

Why this module exists:
- Strategies must know session context (don't breakout-trade in afterhours)
- Risk engine adjusts position sizing based on session (wider stops in premarket)
- Execution engine must not submit orders when market is closed
- Data quality checks need to know if "no data" is expected or a feed error

Data sources:
- NYSE/NASDAQ holiday calendars (hardcoded + verified annually)
- Market halts: checked via broker API at runtime
"""

from __future__ import annotations

import datetime
from enum import Enum
from zoneinfo import ZoneInfo

from core.logging import get_logger

logger = get_logger("market_session")

from dataclasses import dataclass, asdict

# US Eastern timezone — all market hours are ET
ET = ZoneInfo("America/New_York")


class MarketSession(str, Enum):
    """Current market session state."""
    PREMARKET = "premarket"
    REGULAR = "regular"
    AFTERHOURS = "afterhours"
    CLOSED = "closed"
    HOLIDAY = "holiday"
    EARLY_CLOSE_REGULAR = "early_close_regular"   # Regular hours on early close day (until 13:00)
    EARLY_CLOSE_AFTER = "early_close_afterhours"  # After 13:00 on early close day


@dataclass(frozen=True)
class SessionCapabilities:
    """Decoupled operation capabilities defining exactly what actions are allowed."""
    session: MarketSession
    granular_session: str
    can_analyze: bool                   # Market data, technicals, indicators
    can_generate_signal: bool           # Opportunity ranking, strategy confluence
    can_enter: bool                     # Opening new positions
    can_exit: bool                      # Closing existing positions (TP/SL/EOD)
    can_cancel: bool                    # Cancelling orphaned or invalidated orders
    can_place_limit: bool               # Submitting limit orders
    can_place_market: bool              # Submitting market orders
    can_fetch_news: bool                # News feeds and SEC EDGAR
    is_premarket: bool
    is_regular: bool
    is_afterhours: bool
    is_weekend: bool
    is_near_close: bool
    max_position_size_multiplier: float # 1.0 = normal, 0.5 = 50% size, 0.0 = no new entries
    recommended_cycle_seconds: float    # Throttle interval to preserve API quotas
    # Granular stop permissions (Audited Refinement)
    can_create_stop: bool = False       # Submitting new Stop Loss / Bracket leg
    can_modify_stop: bool = False       # Order replace / trailing / break-even
    can_cancel_stop: bool = False       # Cancelling existing stop order
    can_execute_stop: bool = False      # Triggered execution by broker
    can_close_manual: bool = False      # Manual / emergency market close
    force_day: bool = False             # Extended hours policy forces time_in_force=DAY for risk control (Alpaca accepts DAY or GTC)
    can_manage_stop: bool = True        # Backward-compatible stop management indicator

    def to_dict(self) -> dict:
        d = asdict(self)
        d["session"] = self.session.value
        return d


def get_dynamic_max_spread(
    symbol: str,
    session: MarketSession | str,
    natr_14: float | None = None,
) -> float:
    """
    Computes dynamic max allowed bid-ask spread threshold.
    
    Loads configurable base thresholds, session multipliers, and hard caps from RiskSettings.
    Adapts:
    - High-liquidity mega-caps (AAPL, NVDA, MSFT, SPY, QQQ) are held to tighter thresholds.
    - Extended sessions (premarket/afterhours) allow natural spread expansion.
    - Volatile assets (higher NATR) receive a scaled proportional threshold.
    - An immutable hard-cap guarantees spreads never exceed risk boundaries.
    """
    sym = symbol.upper()
    is_mega_cap = sym in {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ"}
    session_str = str(session.value if isinstance(session, MarketSession) else session).lower()
    is_extended = "premarket" in session_str or "afterhours" in session_str

    try:
        from core.config import get_settings
        risk_cfg = get_settings().risk
        base_spread = risk_cfg.spread_regular_base_pct if is_mega_cap else (risk_cfg.spread_regular_base_pct * 2.0)
        session_mult = risk_cfg.spread_extended_multiplier if is_extended else 1.0
        hard_cap = risk_cfg.spread_extended_hard_cap_pct if is_extended else risk_cfg.spread_max_hard_cap_pct
    except Exception:
        base_spread = 0.0012 if is_mega_cap else 0.0030
        session_mult = 1.5 if is_extended else 1.0
        hard_cap = 0.0070 if is_extended else 0.0050

    # Volatility adjustment via NATR (Normalized ATR in %)
    if natr_14 and natr_14 > 0:
        vol_factor = max(0.8, min(1.8, (natr_14 / 3.0) ** 0.5))
    else:
        vol_factor = 1.0

    max_spread = base_spread * session_mult * vol_factor
    return round(max(0.0010, min(hard_cap, max_spread)), 5)


# ── Session Time Boundaries (Eastern Time) ──
SESSION_TIMES = {
    MarketSession.PREMARKET:  {"start": datetime.time(4, 0), "end": datetime.time(9, 30)},
    MarketSession.REGULAR:    {"start": datetime.time(9, 30), "end": datetime.time(16, 0)},
    MarketSession.AFTERHOURS: {"start": datetime.time(16, 0), "end": datetime.time(20, 0)},
    # CLOSED is anything from 20:00 to 04:00 next day
}

# Early close: market closes at 13:00 ET
EARLY_CLOSE_TIME = datetime.time(13, 0)



def _get_holidays(year: int) -> set[datetime.date]:
    """
    NYSE/NASDAQ observed holidays for a given year.
    
    Source: NYSE Rule 7.2, verified against:
    https://www.nyse.com/markets/hours-calendars
    
    These are the standard observed holidays. The actual dates shift
    when a holiday falls on a weekend (observed on Friday or Monday).
    
    NOTE: This list should be verified annually against the official
    NYSE calendar. Dates below are calculated for common years.
    """
    from datetime import date, timedelta

    holidays: set[date] = set()

    # New Year's Day — January 1 (or observed)
    nyd = date(year, 1, 1)
    if nyd.weekday() == 6:  # Sunday → Monday
        holidays.add(date(year, 1, 2))
    elif nyd.weekday() == 5:  # Saturday → prev Friday (Dec 31)
        holidays.add(date(year - 1, 12, 31))
    else:
        holidays.add(nyd)

    # Martin Luther King Jr. Day — 3rd Monday of January
    jan1 = date(year, 1, 1)
    first_monday = jan1 + timedelta(days=(7 - jan1.weekday()) % 7)
    holidays.add(first_monday + timedelta(weeks=2))

    # Presidents' Day — 3rd Monday of February
    feb1 = date(year, 2, 1)
    first_monday = feb1 + timedelta(days=(7 - feb1.weekday()) % 7)
    holidays.add(first_monday + timedelta(weeks=2))

    # Good Friday — varies (Friday before Easter)
    # Using anonymous Gregorian algorithm for Easter
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = date(year, month, day)
    good_friday = easter - timedelta(days=2)
    holidays.add(good_friday)

    # Memorial Day — last Monday of May
    may31 = date(year, 5, 31)
    holidays.add(may31 - timedelta(days=may31.weekday()))

    # Juneteenth — June 19 (observed since 2022)
    june19 = date(year, 6, 19)
    if june19.weekday() == 6:
        holidays.add(date(year, 6, 20))
    elif june19.weekday() == 5:
        holidays.add(date(year, 6, 18))
    else:
        holidays.add(june19)

    # Independence Day — July 4
    july4 = date(year, 7, 4)
    if july4.weekday() == 6:
        holidays.add(date(year, 7, 5))
    elif july4.weekday() == 5:
        holidays.add(date(year, 7, 3))
    else:
        holidays.add(july4)

    # Labor Day — 1st Monday of September
    sep1 = date(year, 9, 1)
    holidays.add(sep1 + timedelta(days=(7 - sep1.weekday()) % 7))

    # Thanksgiving — 4th Thursday of November
    nov1 = date(year, 11, 1)
    first_thursday = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    holidays.add(first_thursday + timedelta(weeks=3))

    # Christmas — December 25
    xmas = date(year, 12, 25)
    if xmas.weekday() == 6:
        holidays.add(date(year, 12, 26))
    elif xmas.weekday() == 5:
        holidays.add(date(year, 12, 24))
    else:
        holidays.add(xmas)

    return holidays


def _get_early_close_dates(year: int) -> set[datetime.date]:
    """
    NYSE early close dates (13:00 ET instead of 16:00 ET).
    
    Typically:
    - Day before Independence Day (July 3, unless it falls on weekend)
    - Black Friday (day after Thanksgiving)
    - Christmas Eve (Dec 24, unless it falls on weekend)
    
    Source: NYSE Rule 7.2
    """
    from datetime import date, timedelta

    early_close: set[date] = set()

    # Day before Independence Day
    july3 = date(year, 7, 3)
    if july3.weekday() < 5:  # weekday only
        early_close.add(july3)

    # Black Friday (day after Thanksgiving)
    nov1 = date(year, 11, 1)
    first_thursday = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    thanksgiving = first_thursday + timedelta(weeks=3)
    black_friday = thanksgiving + timedelta(days=1)
    early_close.add(black_friday)

    # Christmas Eve
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5:  # weekday only
        early_close.add(dec24)

    return early_close


# Cache holidays per year
_holiday_cache: dict[int, set[datetime.date]] = {}
_early_close_cache: dict[int, set[datetime.date]] = {}


def is_holiday(dt: datetime.date) -> bool:
    """Check if a date is a market holiday."""
    year = dt.year
    if year not in _holiday_cache:
        _holiday_cache[year] = _get_holidays(year)
    return dt in _holiday_cache[year]


def is_early_close(dt: datetime.date) -> bool:
    """Check if a date is an early close day."""
    year = dt.year
    if year not in _early_close_cache:
        _early_close_cache[year] = _get_early_close_dates(year)
    return dt in _early_close_cache[year]


def get_current_session(
    now: datetime.datetime | None = None,
) -> MarketSession:
    """
    Determine the current market session.
    
    Args:
        now: Optional datetime override (for testing/backtesting).
             Must be timezone-aware. If None, uses current time.
    
    Returns:
        Current MarketSession state.
    """
    if now is None:
        now = datetime.datetime.now(ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    current_date = now.date()
    current_time = now.time()

    # Weekend
    if current_date.weekday() >= 5:
        return MarketSession.CLOSED

    # Holiday
    if is_holiday(current_date):
        return MarketSession.HOLIDAY

    # Early close day
    if is_early_close(current_date):
        if current_time < SESSION_TIMES[MarketSession.PREMARKET]["start"]:
            return MarketSession.CLOSED
        elif current_time < SESSION_TIMES[MarketSession.REGULAR]["start"]:
            return MarketSession.PREMARKET
        elif current_time < EARLY_CLOSE_TIME:
            return MarketSession.EARLY_CLOSE_REGULAR
        else:
            return MarketSession.CLOSED  # After 13:00 on early close day

    # Normal trading day
    if current_time < SESSION_TIMES[MarketSession.PREMARKET]["start"]:
        return MarketSession.CLOSED
    elif current_time < SESSION_TIMES[MarketSession.REGULAR]["start"]:
        return MarketSession.PREMARKET
    elif current_time < SESSION_TIMES[MarketSession.REGULAR]["end"]:
        return MarketSession.REGULAR
    elif current_time < SESSION_TIMES[MarketSession.AFTERHOURS]["end"]:
        return MarketSession.AFTERHOURS
    else:
        return MarketSession.CLOSED


def can_trade(
    session: MarketSession | datetime.datetime | None = None,
    allow_premarket: bool = True,
    allow_afterhours: bool = True,
) -> bool:
    """Check if trading is possible in the current/given session or datetime."""
    if isinstance(session, datetime.datetime):
        session = get_current_session(session)
    elif session is None:
        session = get_current_session()

    valid_sessions = {
        MarketSession.REGULAR,
        MarketSession.EARLY_CLOSE_REGULAR,
    }
    if allow_premarket:
        valid_sessions.add(MarketSession.PREMARKET)
    if allow_afterhours:
        valid_sessions.add(MarketSession.AFTERHOURS)

    return session in valid_sessions


def is_regular_hours(session: MarketSession | None = None) -> bool:
    """Check if we're in regular trading hours (best liquidity)."""
    if session is None:
        session = get_current_session()
    return session in {
        MarketSession.REGULAR,
        MarketSession.EARLY_CLOSE_REGULAR,
    }


def get_session_capabilities(
    now: datetime.datetime | None = None,
) -> SessionCapabilities:
    """
    Evaluates current time and returns a decoupled SessionCapabilities object.
    
    This provides fine-grained capability checks for:
    - can_analyze: indicators, universe scanning
    - can_generate_signal: opportunity ranking & confluence
    - can_enter: open new positions (disabled near close, weekend, overnight, early premarket)
    - can_exit: close positions (TP, SL, EOD)
    - can_manage_stop: adjust trailing stop, break-even
    - can_cancel: cancel orphaned or invalidated orders
    - can_place_limit / can_place_market: execution order type permissions
    - dynamic sizing multipliers and recommended throttle intervals
    """
    if now is None:
        now = datetime.datetime.now(ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    current_date = now.date()
    current_time = now.time()
    t_hour, t_min = current_time.hour, current_time.minute

    # 1. Weekend
    if current_date.weekday() >= 5:
        return SessionCapabilities(
            session=MarketSession.CLOSED,
            granular_session="weekend_analysis",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=False,
            can_exit=False,
            can_manage_stop=False,
            can_cancel=False,
            can_place_limit=False,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=False,
            is_regular=False,
            is_afterhours=False,
            is_weekend=True,
            is_near_close=False,
            max_position_size_multiplier=0.0,
            recommended_cycle_seconds=1800.0,
        )

    # 2. Holiday
    if is_holiday(current_date):
        return SessionCapabilities(
            session=MarketSession.HOLIDAY,
            granular_session="holiday_analysis",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=False,
            can_exit=False,
            can_manage_stop=False,
            can_cancel=False,
            can_place_limit=False,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=False,
            is_regular=False,
            is_afterhours=False,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=0.0,
            recommended_cycle_seconds=1800.0,
        )

    # 3. Overnight (20:00 to 04:00 ET)
    if current_time >= datetime.time(20, 0) or current_time < datetime.time(4, 0):
        return SessionCapabilities(
            session=MarketSession.CLOSED,
            granular_session="overnight_analysis",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=False,
            can_exit=False,
            can_manage_stop=False,
            can_cancel=False,
            can_place_limit=False,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=False,
            is_regular=False,
            is_afterhours=False,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=0.0,
            recommended_cycle_seconds=900.0,
        )

    # 4. Pre-Market Early (04:00 to 07:00 ET) - Illiquid, no new entries, allow exits/cancels
    if current_time < datetime.time(7, 0):
        return SessionCapabilities(
            session=MarketSession.PREMARKET,
            granular_session="premarket_early",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=False,  # strictly no new entries in deep premarket
            can_exit=True,   # allow SL/emergency exits
            can_manage_stop=True,
            can_cancel=True,
            can_place_limit=True,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=True,
            is_regular=False,
            is_afterhours=False,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=0.0,
            recommended_cycle_seconds=120.0,
        )

    # 5. Pre-Market Active (07:00 to 09:25 ET) - Active premarket, limit entries permitted
    if current_time < datetime.time(9, 25):
        return SessionCapabilities(
            session=MarketSession.PREMARKET,
            granular_session="premarket_active",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=True,
            can_exit=True,
            can_manage_stop=True,
            can_cancel=True,
            can_place_limit=True,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=True,
            is_regular=False,
            is_afterhours=False,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=0.75,
            recommended_cycle_seconds=60.0,
        )

    # 6. Pre-Open Lock (09:25 to 09:30 ET) - Freeze modifications right before bell
    if current_time < datetime.time(9, 30):
        return SessionCapabilities(
            session=MarketSession.PREMARKET,
            granular_session="pre_open_lock",
            can_analyze=True,
            can_generate_signal=False,
            can_enter=False,
            can_exit=True,
            can_manage_stop=False,
            can_cancel=True,
            can_place_limit=False,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=True,
            is_regular=False,
            is_afterhours=False,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=0.0,
            recommended_cycle_seconds=30.0,
        )

    # Early close handling check
    early_day = is_early_close(current_date)
    reg_end = EARLY_CLOSE_TIME if early_day else datetime.time(16, 0)

    # 7. Regular Hours
    if current_time < reg_end:
        # Check sub-windows: Near Close vs Risk Descalation vs Regular Full
        if early_day and current_time >= datetime.time(12, 40):
            # Near close on early day (12:40 - 13:00)
            return SessionCapabilities(
                session=MarketSession.EARLY_CLOSE_REGULAR,
                granular_session="near_close",
                can_analyze=True,
                can_generate_signal=False,
                can_enter=False,
                can_exit=True,
                can_manage_stop=True,
                can_cancel=True,
                can_place_limit=True,
                can_place_market=True,
                can_fetch_news=True,
                is_premarket=False,
                is_regular=True,
                is_afterhours=False,
                is_weekend=False,
                is_near_close=True,
                max_position_size_multiplier=0.0,
                recommended_cycle_seconds=30.0,
            )
        elif not early_day and (t_hour == 15 and t_min >= 40):
            # Near close (15:40 - 16:00 ET)
            return SessionCapabilities(
                session=MarketSession.REGULAR,
                granular_session="near_close",
                can_analyze=True,
                can_generate_signal=False,
                can_enter=False,
                can_exit=True,
                can_manage_stop=True,
                can_cancel=True,
                can_place_limit=True,
                can_place_market=True,
                can_fetch_news=True,
                is_premarket=False,
                is_regular=True,
                is_afterhours=False,
                is_weekend=False,
                is_near_close=True,
                max_position_size_multiplier=0.0,
                recommended_cycle_seconds=30.0,
            )
        elif not early_day and (t_hour == 15 and t_min >= 30):
            # Risk descalation (15:30 - 15:40 ET)
            return SessionCapabilities(
                session=MarketSession.REGULAR,
                granular_session="risk_descalation",
                can_analyze=True,
                can_generate_signal=True,
                can_enter=True,
                can_exit=True,
                can_manage_stop=True,
                can_cancel=True,
                can_place_limit=True,
                can_place_market=True,
                can_fetch_news=True,
                is_premarket=False,
                is_regular=True,
                is_afterhours=False,
                is_weekend=False,
                is_near_close=False,
                max_position_size_multiplier=0.5,
                recommended_cycle_seconds=30.0,
            )
        else:
            # Full regular session
            sess_type = MarketSession.EARLY_CLOSE_REGULAR if early_day else MarketSession.REGULAR
            return SessionCapabilities(
                session=sess_type,
                granular_session="regular_full",
                can_analyze=True,
                can_generate_signal=True,
                can_enter=True,
                can_exit=True,
                can_manage_stop=True,
                can_cancel=True,
                can_place_limit=True,
                can_place_market=True,
                can_fetch_news=True,
                is_premarket=False,
                is_regular=True,
                is_afterhours=False,
                is_weekend=False,
                is_near_close=False,
                max_position_size_multiplier=1.0,
                recommended_cycle_seconds=60.0,
            )

    # 8. After-Hours (16:00 to 20:00 ET)
    if current_time < datetime.time(20, 0):
        return SessionCapabilities(
            session=MarketSession.AFTERHOURS,
            granular_session="afterhours",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=True,
            can_exit=True,
            can_manage_stop=True,
            can_cancel=True,
            can_place_limit=True,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=False,
            is_regular=False,
            is_afterhours=True,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=0.75,
            recommended_cycle_seconds=60.0,
        )

    # 9. Fallback Closed
    return SessionCapabilities(
        session=MarketSession.CLOSED,
        granular_session="closed",
        can_analyze=True,
        can_generate_signal=False,
        can_enter=False,
        can_exit=False,
        can_manage_stop=False,
        can_cancel=False,
        can_place_limit=False,
        can_place_market=False,
        can_fetch_news=True,
        is_premarket=False,
        is_regular=False,
        is_afterhours=False,
        is_weekend=False,
        is_near_close=False,
        max_position_size_multiplier=0.0,
        recommended_cycle_seconds=900.0,
    )


def get_session_info(
    now: datetime.datetime | None = None,
) -> dict:
    """
    Get complete session information for monitoring/dashboard.
    
    Returns:
        Dictionary with session state, tradeable status, capabilities, and timing info.
    """
    if now is None:
        now = datetime.datetime.now(ET)

    session = get_current_session(now)
    caps = get_session_capabilities(now)

    return {
        "session": session.value,
        "granular_session": caps.granular_session,
        "can_trade": can_trade(session),
        "is_regular": is_regular_hours(session),
        "capabilities": caps.to_dict(),
        "timestamp": now.isoformat(),
        "timezone": "US/Eastern",
        "date": now.date().isoformat(),
        "time": now.time().isoformat(),
        "weekday": now.strftime("%A"),
        "is_holiday": is_holiday(now.date()),
        "is_early_close": is_early_close(now.date()),
    }

