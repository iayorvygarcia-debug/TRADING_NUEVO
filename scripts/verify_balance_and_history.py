"""
Verification script for Balance and Closed Trades History (WIN / LOSS).
Tests:
1. Initial cash and realized PnL
2. Position opening with bracket orders (TP & SL)
3. Take-Profit hit -> Position closed with WIN (+PnL) -> Recorded in trade_history
4. Stop-Loss hit -> Position closed with LOSS (-PnL) -> Recorded in trade_history
5. QuantTradingWorker telemetry snapshot contains realized_pnl and trade_history
"""

import asyncio
from datetime import datetime, timezone
from services.broker_engine.paper_broker import PaperBroker
from services.broker_engine.base import OrderRequest, OrderSide, OrderType, OrderClass
from apps.worker.quant_trading_worker import QuantTradingWorker


async def test_balance_and_history():
    print("=" * 70)
    print("  VERIFICACION: BALANCE EN TIEMPO REAL E HISTORIAL DE CIERRES")
    print("=" * 70)

    # 1. Initial State
    broker = PaperBroker(initial_cash=100000.0)
    await broker.connect()

    assert broker.cash == 100000.0, f"Cash inicial incorrecto: {broker.cash}"
    assert broker.realized_pnl == 0.0, f"PnL inicial incorrecto: {broker.realized_pnl}"
    assert broker.get_trade_history() == [], "El historial inicial debe estar vacio"
    print("[OK] 1. Estado inicial verificado: Cash = $100,000.00 | PnL Realizado = $0.00 | Historial = []")

    # 2. Open Long Bracket Order in AAPL (Entry: 150, TP: 160, SL: 140, Qty: 10)
    broker.update_market_price("AAPL", 150.0)
    order_req = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        order_class=OrderClass.BRACKET,
        take_profit_price=160.0,
        stop_loss_price=140.0,
    )
    result = await broker.submit_order(order_req)
    assert result.status.value == "filled"
    assert "AAPL" in broker._positions
    pos = broker._positions["AAPL"]
    print(f"[OK] 2. Posicion abierta AAPL: {pos.quantity} acciones @ ${pos.avg_entry_price:.2f} (TP: $160, SL: $140)")
    print(f"     Cash restante: ${broker.cash:,.2f}")

    # 3. Simulate Price Rising to $161 (Triggers Take-Profit -> WIN)
    print("\n--- Simulando movimiento alcista a $161.00 (debe activar Take-Profit) ---")
    broker.update_market_price("AAPL", 161.0, high=161.0, low=150.0)

    assert "AAPL" not in broker._positions, "AAPL deberia haberse cerrado por Take-Profit"
    history = broker.get_trade_history()
    assert len(history) == 1, f"Se esperaba 1 trade cerrado, hay {len(history)}"
    trade1 = history[0]
    print(f"[OK] 3. Cierre detectado:")
    print(f"     Ticker: {trade1['symbol']} | Lado: {trade1['side']} | Cant: {trade1['qty']}")
    print(f"     Entrada: ${trade1['entry_price']:.2f} -> Salida: ${trade1['exit_price']:.2f}")
    print(f"     PnL: ${trade1['pnl']:+.2f} ({trade1['pnl_pct']:+.2f}%)")
    print(f"     Resultado: {trade1['outcome']} | Causa: {trade1['reason']}")
    assert trade1["outcome"] == "WIN", f"Esperado WIN, obtenido {trade1['outcome']}"
    assert trade1["pnl"] > 0, f"PnL deberia ser positivo: {trade1['pnl']}"
    assert broker.realized_pnl > 0, f"PnL realizado global debe ser positivo: {broker.realized_pnl}"

    # 4. Open Long Bracket Order in TSLA (Entry: 200, TP: 220, SL: 190, Qty: 5)
    broker.update_market_price("TSLA", 200.0)
    order_tsla = OrderRequest(
        symbol="TSLA",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5.0,
        order_class=OrderClass.BRACKET,
        take_profit_price=220.0,
        stop_loss_price=190.0,
    )
    res_tsla = await broker.submit_order(order_tsla)
    assert "TSLA" in broker._positions
    print(f"\n[OK] 4. Posicion abierta TSLA: 5 acciones @ $200 (TP: $220, SL: $190)")

    # 5. Simulate Price Dropping to $189 (Triggers Stop-Loss -> LOSS)
    print("\n--- Simulando movimiento bajista a $189.00 (debe activar Stop-Loss) ---")
    broker.update_market_price("TSLA", 189.0, high=200.0, low=189.0)

    assert "TSLA" not in broker._positions, "TSLA deberia haberse cerrado por Stop-Loss"
    history = broker.get_trade_history()
    assert len(history) == 2, f"Se esperaban 2 trades cerrados, hay {len(history)}"
    trade2 = history[0]  # Most recent first
    print(f"[OK] 5. Cierre con perdida detectado:")
    print(f"     Ticker: {trade2['symbol']} | Lado: {trade2['side']} | Cant: {trade2['qty']}")
    print(f"     Entrada: ${trade2['entry_price']:.2f} -> Salida: ${trade2['exit_price']:.2f}")
    print(f"     PnL: ${trade2['pnl']:+.2f} ({trade2['pnl_pct']:+.2f}%)")
    print(f"     Resultado: {trade2['outcome']} | Causa: {trade2['reason']}")
    assert trade2["outcome"] == "LOSS", f"Esperado LOSS, obtenido {trade2['outcome']}"
    assert trade2["pnl"] < 0, f"PnL deberia ser negativo: {trade2['pnl']}"

    # 6. Telemetry Snapshot Integration
    print("\n--- Verificando Snapshot de Telemetria para la Web Dashboard ---")
    worker = QuantTradingWorker(paper_broker=broker)
    snap = worker.get_telemetry_snapshot()

    port = snap["portfolio"]
    print(f"[OK] Balance Total: ${port['portfolio_value']:,.2f}")
    print(f"[OK] Saldo Disponible (Cash): ${port['cash']:,.2f}")
    print(f"[OK] Poder de Compra: ${port['buying_power']:,.2f}")
    print(f"[OK] P&L Realizado Acumulado: ${port['realized_pnl']:+,.2f}")
    print(f"[OK] Retorno Total: {port['total_return_pct']:+.2f}%")
    print(f"[OK] Historial de Operaciones en Telemetria: {len(snap['trade_history'])} registros")

    assert port["realized_pnl"] == round(broker.realized_pnl, 2)
    assert len(snap["trade_history"]) == 2
    assert snap["trade_history"][0]["symbol"] == "TSLA"
    assert snap["trade_history"][1]["symbol"] == "AAPL"

    print("\n" + "=" * 70)
    print("  TODAS LAS PRUEBAS COMPLETADAS CON EXITO! (100% OPERATIVO)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_balance_and_history())
