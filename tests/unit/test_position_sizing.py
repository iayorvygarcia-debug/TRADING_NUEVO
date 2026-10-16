"""Tests for risk-based sizing, sector exposure cap and market trend multiplier."""

from services.risk_engine.position_sizing import (
    check_sector_exposure,
    market_trend_multiplier,
    risk_based_shares,
)


class TestRiskBasedShares:
    def test_risk_limits_size_when_stop_is_wide(self):
        # $1000 capital, 1% risk = $10; stop $5 away -> 2 shares (notional cap would allow 4 @ $50)
        assert risk_based_shares(1000, 50.0, 45.0, 0.01, 0.20) == 2.0

    def test_notional_cap_binds_when_stop_is_tight(self):
        # risk allows 10/0.5 = 20 shares, notional cap 200/50 = 4
        assert risk_based_shares(1000, 50.0, 49.5, 0.01, 0.20) == 4.0

    def test_no_stop_uses_notional_cap(self):
        assert risk_based_shares(1000, 50.0, None) == 4.0

    def test_never_returns_negative_or_invents_shares(self):
        assert risk_based_shares(1000, 5000.0, 4900.0) == 0.0
        assert risk_based_shares(0, 50.0, 45.0) == 0.0

    def test_short_stop_above_entry(self):
        assert risk_based_shares(1000, 50.0, 55.0, 0.01, 0.20) == 2.0


class TestSectorExposure:
    def test_blocks_third_position_in_same_sector(self):
        res = check_sector_exposure("MSFT", ["NVDA", "AVGO"], max_per_sector=2)
        assert not res.allowed and res.sector == "technology"

    def test_allows_other_sector(self):
        assert check_sector_exposure("JPM", ["NVDA", "AVGO", "MSFT"], max_per_sector=2).allowed

    def test_symbol_itself_not_counted(self):
        assert check_sector_exposure("NVDA", ["NVDA", "AVGO"], max_per_sector=2).allowed

    def test_unknown_sector_not_blocked(self):
        assert check_sector_exposure("ZZZZ", ["ZZZ1", "ZZZ2", "ZZZ3"], max_per_sector=1).allowed


class TestMarketTrend:
    def test_bull_market_longs_full_shorts_reduced(self):
        assert market_trend_multiplier(500, 450, is_long=True) == 1.0
        assert market_trend_multiplier(500, 450, is_long=False) == 0.5

    def test_bear_market_longs_reduced(self):
        assert market_trend_multiplier(400, 450, is_long=True) == 0.5
        assert market_trend_multiplier(400, 450, is_long=False) == 1.0

    def test_missing_data_never_blocks(self):
        assert market_trend_multiplier(None, None, is_long=True) == 1.0
        assert market_trend_multiplier(500, 0, is_long=False) == 1.0


class TestPartialExit:
    def test_sells_half_at_one_r(self):
        from services.risk_engine.position_sizing import partial_exit_qty
        # entry 100, stop 95 -> R=5; mark 105 = 1R
        assert partial_exit_qty(10, 100, 105, 95, True) == 5

    def test_no_sale_before_trigger(self):
        from services.risk_engine.position_sizing import partial_exit_qty
        assert partial_exit_qty(10, 100, 104, 95, True) == 0

    def test_only_once(self):
        from services.risk_engine.position_sizing import partial_exit_qty
        assert partial_exit_qty(10, 100, 110, 95, True, already_taken=True) == 0

    def test_never_sells_whole_position(self):
        from services.risk_engine.position_sizing import partial_exit_qty
        assert partial_exit_qty(2, 100, 110, 95, True) == 1
        assert partial_exit_qty(1, 100, 110, 95, True) == 0

    def test_short_side(self):
        from services.risk_engine.position_sizing import partial_exit_qty
        assert partial_exit_qty(-8, 100, 95, 105, False) == 4

    def test_no_stop_no_partial(self):
        from services.risk_engine.position_sizing import partial_exit_qty
        assert partial_exit_qty(10, 100, 120, None, True) == 0
