# 🧪 Reporte Cuantitativo de Shadow Trading — 2026-10-03

> **MODO SHADOW ACTIVO:** Cero órdenes reales enviadas al broker.
> Generado el: `2026-10-03 01:33:50 UTC`

## 0. Certificación de Aislamiento del Broker (Shadow Gate Audit)

| Invariante de Seguridad | Valor Registrado | Estado de Certificación |
| :--- | :--- | :--- |
| **Órdenes Candidatas hacia Broker** | `3` | 🟢 Evaluadas y Elegibles |
| **Órdenes Interceptadas por Shadow Gate** | `3` | 🟢 100.0% Interceptadas |
| **Llamadas a `broker.submit_order()`** | `0` | 🟢 Cero Llamadas |
| **Órdenes Reales Creadas en Broker** | `0` | 🟢 Cero Creadas |

---

## 1. Embudo de Conversión (Separación Señal vs Ejecución)

| Etapa del Pipeline | Cantidad | Tasa de Conversión |
| :--- | :--- | :--- |
| **1. Señales Generadas por Estrategias** | `5` | `100.0%` |
| **2. Aprobadas por Risk / Guards** | `3` | `60.0%` |
| **3. Elegibles para Orden (Qty > 0)** | `3` | `60.0%` |
| **4. Entradas Virtuales Ejecutadas** | `3` | `100.0%` |
| **5. Operaciones Cerradas (Round-trip)** | `2` | `-` |

---

## 2. Resumen Cuantitativo & Expectancy

| Métrica | Valor |
| :--- | :--- |
| **Win Rate** | `50.0%` (1 Ganadoras / 1 Perdedoras) |
| **Loss Rate** | `50.0%` |
| **Average Win ($)** | `$147.23` |
| **Average Loss ($)** | `$102.77` |
| **Expectancy por Operación ($)** | **`$22.23`** |
| **Profit Factor** | `1.43` |
| **P&L Bruto Total** | `$51.00` |
| **Fricción Total Slippage** | `$2.31` |
| **Costo Total Spread** | `$4.23` |
| **Comisiones Totales** | `$0.00` |
| **P&L Neto Realizado (Trades Cerrados)** | **`$44.46`** |
| **P&L Flotante No Realizado (Posiciones Abiertas)** | **`$0.00`** |
| **Variación Total de Equity Teórico** | **`$44.46`** |
| **Equity Teórico Actual** | **`$100,190.18`** (Base: `$100,145.72`) |
| **R-Multiple Acumulado (Cumulative R)** | **`+0.43R`** |
| **Expectancy en R** | **`+0.21R`** |
| **R-Multiple Promedio Realizado** | `+0.21R` |
| **R-Multiple Mediano Realizado** | `+0.21R` |
| **Desviación Estándar de R (Std Dev R)** | `1.75R (sample standard deviation, ddof=1)` |
| **R Máximo / R Mínimo** | `+1.45R` / `-1.02R` |
| **Duración Media por Operación** | `0.0 min` (BAR_SIMULATION) |
| **Duración Mediana por Operación** | `0.0 min` |
| **Excursión Favorable Media (MFE)** | `+1.92%` |
| **Excursión Adversa Media (MAE)** | `-1.62%` |
| **Max Drawdown ($)** | `$102.77` |
| **Max Drawdown (%)** | `1.01%` |

> ⚠️ **Nota Metodológica de Muestra:** Expectancy observada en muestra preliminar reducida (2 trades cerrados). No constituye inferencia estadística de régimen estable. La desviación estándar de R se calcula muestralmente (ddof=1); con N=2 no debe usarse para inferir estabilidad de la estrategia.
> ℹ️ **Nota de Duración:** `duration_source = BAR_SIMULATION`. En simulación de barras sin ticks intrabar, la duración registrada es 0.0 min.

### 2.1. Distribución Empírica de R-Multiple

| Rango de Rendimiento (R) | Operaciones | % Muestra | Interpretación |
| :--- | :--- | :--- | :--- |
| **< -2.0R** | `0` | `0.0%` | Pérdida severa / Salto de Stop |
| **[-2.0R, -1.0R)** | `1` | `50.0%` | Pérdida normal por Stop-Loss |
| **[-1.0R, 0.0R)** | `0` | `0.0%` | Pérdida mitigada / Salida técnica |
| **[0.0R, 1.0R)** | `0` | `0.0%` | Ganancia reducida / Break-even |
| **[1.0R, 2.0R)** | `1` | `50.0%` | Ganancia objetivo (Take-Profit) |
| **>= 2.0R** | `0` | `0.0%` | Ganancia extendida / Runner |

---

## 3. Distribución Empírica de Slippage & Spread

| Parámetro de Costo | Promedio | Mediana | Máximo Observado |
| :--- | :--- | :--- | :--- |
| **Slippage Simulado** | `0.016%` (1.6 bps) | `0.015%` (1.5 bps) | `0.018%` (1.8 bps) |
| **Spread de Mercado** | `0.114%` (11.4 bps) | `0.050%` (5.0 bps) | `-` |

---

## 4. Rendimiento Segregado por Estrategia

| Estrategia | Señales | Entradas | Salidas | Trades | Win Rate | P&L Neto | Profit Factor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trend Following** | `2` | `1` | `0` | `0` | `0.0%` | 🟢 $0.00 | `0.00` |
| **Breakout** | `2` | `1` | `1` | `1` | `100.0%` | 🟢 $147.23 | `N/A (sin pérdidas en muestra)` |
| **Mean Reversion** | `1` | `1` | `1` | `1` | `0.0%` | 🔴 $-102.77 | `0.00` |

---

## 5. Filtros de Seguridad y Rechazos por Risk Guard

| Risk Guard | Total Rechazos |
| :--- | :--- |
| **PreTradeRiskGuard** | `1` |
| **CircuitBreaker** | `1` |

### Muestra de Motivos Exactos de Rechazo

- **SPY** (PreTradeRiskGuard): `PreTradeGuard: Adverse gap -2.8% exceeds NATR threshold; Spread 0.35% > cap 0.25%`
- **NVDA** (CircuitBreaker): `RiskEngine: CircuitBreaker Tier 2 active due to market-wide volatility spike`

---

## 6. Historial de Operaciones Virtuales Cerradas (Ledger de Costos)

| ID | Símbolo | Lado | Qty | Base In | Shadow In | Shadow Out | P&L Bruto | Slippage | Spread Cost | Comisiones | P&L Neto | R-Mult | Duración | MFE | MAE | Motivo Salida |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tr_MSFT_b_ent_f3e4cc2da3_1` | **MSFT** | LONG | 10.0 | $400.00 | $400.14 | $414.86 | $150.00 | $1.14 | $1.63 | $0.00 | 🟢 $147.23 | `+1.45R` | `0.0m` | +3.8% | 0.0% | `TAKE_PROFIT` |
| `tr_TSLA_b_ent_f2d4dc27bc_2` | **TSLA** | LONG | 15.0 | $220.00 | $220.13 | $213.28 | $-99.00 | $1.17 | $2.60 | $0.00 | 🔴 $-102.77 | `-1.02R` | `0.0m` | +0.0% | -3.2% | `STOP_LOSS` |

> 📐 **Identidad Contable Auditada:** `P&L Neto = P&L Bruto − Slippage − Spread Cost − Comisiones` (tolerancia de redondeo: ±$0.00).

---

## 7. Posiciones Virtuales Abiertas

| Símbolo | Lado | Cantidad | Entrada | Actual | P&L Flotante | MFE Actual | MAE Actual | Stop Loss | Take Profit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | LONG | 25.0 | $180.07 | $180.07 | 🟢 $0.00 (0.00%) | +0.0% | 0.0% | $174.00 | $192.00 |

---

## 8. Ejemplo Forense de Decisión Reconstruida (¿Por qué quiso operar?)

*Sin decisiones registradas.*