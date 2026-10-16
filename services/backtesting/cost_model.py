"""
AI Quant Trading Platform — Backtesting Cost Model.

Accurately simulates:
- Broker commissions (e.g., $0.00 for Alpaca, $0.0035/share for IBKR)
- Half-spread friction (e.g., 2-5 basis points)
- Market impact & slippage (fixed or volatility-adjusted)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CostModel:
    """
    Transaction cost simulator.
    
    Defaults reflect US Equities via Alpaca (commission-free retail, 2 bps spread, 3 bps slippage).
    """
    commission_per_share: float = 0.0000   # $0 for Alpaca
    commission_min: float = 0.00           # Minimum commission per order
    half_spread_bps: float = 2.0           # 2 basis points (0.02%)
    slippage_bps: float = 3.0              # 3 basis points (0.03%)
    sec_fee_rate: float = 0.0000278        # SEC fee on sells (~$27.80 per million)
    taf_fee_per_share: float = 0.000166    # FINRA TAF on sells (max $8.30)

    def calculate_fill_price(
        self,
        base_price: float,
        is_buy: bool,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate realistic fill price incorporating half-spread and slippage.
        
        Buys are filled higher; sells are filled lower.
        """
        total_friction_bps = (self.half_spread_bps + self.slippage_bps) * volatility_multiplier
        friction_pct = total_friction_bps / 10000.0

        if is_buy:
            return round(base_price * (1.0 + friction_pct), 4)
        else:
            return round(base_price * (1.0 - friction_pct), 4)

    def calculate_commission(self, shares: float, price: float, is_buy: bool) -> float:
        """Calculate total broker and regulatory commissions."""
        comm = max(self.commission_min, shares * self.commission_per_share)

        if not is_buy:
            # Regulatory fees apply on sells
            notional = shares * price
            sec_fee = notional * self.sec_fee_rate
            taf_fee = min(8.30, shares * self.taf_fee_per_share)
            comm += sec_fee + taf_fee

        return round(comm, 4)
