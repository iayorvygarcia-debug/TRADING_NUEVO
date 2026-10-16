# 📊 Reporte Cuantitativo Acumulativo de Shadow Trading

> **FASE DE ACUMULACIÓN CUANTITATIVA:** Cero órdenes reales enviadas al broker.
> **ESTADO DE ESTRATEGIAS:** 🔒 CONGELADAS (Trend Following, Breakout, Mean Reversion). Cero modificaciones a indicadores, pesos o parámetros.
> Generado el: `2026-10-03 01:33:24 UTC` | Días Observados: `1`

## 0. Certificación de Aislamiento del Broker (Shadow Gate Audit)

| Invariante de Seguridad | Valor Registrado | Estado de Certificación |
| :--- | :--- | :--- |
| **Órdenes Candidatas hacia Broker** | `3` | 🟢 Evaluadas y Elegibles |
| **Órdenes Interceptadas por Shadow Gate** | `3` | 🟢 100.0% Interceptadas |
| **Llamadas a `broker.submit_order()`** | `0` | 🟢 Cero Llamadas |
| **Órdenes Reales Creadas en Broker** | `0` | 🟢 Cero Creadas |

---

## 1. Resumen Acumulativo de Rendimiento & R-Multiple

| Métrica Global | Valor Acumulado |
| :--- | :--- |
| **Días Observados** | `1` sesión(es) |
| **Total Operaciones Cerradas** | `2` |
| **Win Rate Acumulado** | `50.0%` (1 Ganadoras / 1 Perdedoras) |
| **P&L Bruto Total** | `$51.00` |
| **Fricción Total (Slippage + Spread + Fees)** | `$6.54` |
| **P&L Neto Realizado** | **`$44.46`** |
| **Profit Factor Acumulado** | `1.43` |
| **Expectancy por Operación ($)** | **`$22.23`** |
| **Expectancy en R** | **`+0.21R`** |
| **R-Multiple Acumulado (Cumulative R)** | **`+0.43R`** |
| **R-Multiple Promedio Realizado** | `+0.21R` |
| **R-Multiple Mediano Realizado** | `+0.21R` |
| **Desviación Estándar de R (Std Dev R)** | `1.75R (sample standard deviation, ddof=1)` |
| **R Máximo / R Mínimo** | `+1.45R` / `-1.02R` |
| **Max Drawdown ($)** | `$102.77` (`1.01%`) |
| **Duración Media / Mediana** | `0.0m` / `0.0m` (`BAR_SIMULATION`) |
| **MFE / MAE Promedio** | `+1.92%` / `-1.62%` |

> ⚠️ **Nota Metodológica de Muestra:** Con N=2, la muestra actual es estrictamente observacional. Un umbral futuro de 30–50 trades representará una primera evaluación estadística preliminar, no una certificación automática de ventaja (edge). La desviación estándar muestral (ddof=1) con tamaño muestral reducido no debe interpretarse como estabilidad.
> ℹ️ **Nota de Duración:** `duration_source = BAR_SIMULATION`. La duración registrada es 0.0m mientras la simulación no disponga de granularidad temporal intrabar por segundo/tick.

---

## 2. Distribución Estadística de R-Multiple

| Rango de Rendimiento (R) | Operaciones | % Muestra | Interpretación Metodológica |
| :--- | :--- | :--- | :--- |
| **< -2.0R** | `0` | `0.0%` | Pérdida severa / Salto de Stop |
| **[-2.0R, -1.0R)** | `1` | `50.0%` | Pérdida normal por Stop-Loss |
| **[-1.0R, 0.0R)** | `0` | `0.0%` | Pérdida mitigada / Salida técnica |
| **[0.0R, 1.0R)** | `0` | `0.0%` | Ganancia reducida / Break-even |
| **[1.0R, 2.0R)** | `1` | `50.0%` | Ganancia objetivo (Take-Profit) |
| **>= 2.0R** | `0` | `0.0%` | Ganancia extendida / Runner |

---

## 3. Serie Histórica de Progresión Diaria

| Fecha | Señales | Entradas | Cerrados | P&L Neto ($) | Profit Factor | Max DD ($) | MFE Prom | MAE Prom | R Acumulado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-10-03** | `5` | `3` | `2` | 🟢 $44.46 | `1.43` | `$102.77` | +1.92% | -1.62% | **`+0.43R`** |

---

## 4. Rendimiento Segregado por Estrategia (Muestra de Observación)

> ⚠️ **REGLA DE AUDITORÍA:** Registro observacional puro. No se autoriza optimización ni descarte de estrategias por varianza de corto plazo.

| Estrategia | Señales | Entradas | Cerrados | Win Rate | P&L Neto | Profit Factor | Estado del Código |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trend Following** | `2` | `1` | `0` | `0.0%` | 🟢 $0.00 | `0.00` | 🔒 CONGELADA |
| **Breakout** | `2` | `1` | `1` | `100.0%` | 🟢 $147.23 | `N/A (sin pérdidas en muestra)` | 🔒 CONGELADA |
| **Mean Reversion** | `1` | `1` | `1` | `0.0%` | 🔴 $-102.77 | `0.00` | 🔒 CONGELADA |

---

## 5. Rendimiento Segregado por Régimen de Mercado

| Régimen de Mercado | Señales | Rechazadas | Entradas | Cerrados | P&L Neto ($) | Win Rate | R Promedio | Observación Cuantitativa |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REGULAR (09:30 - 16:00 ET)** | `5` | `0` | `3` | `2` | `$44.46` | `50.0%` | `+0.21R` | Fricción estándar (1-2 bps slippage) |
| **PREMARKET / AFTER-HOURS** | `0` | `0` | `0` | `0` | `$0.00` | `0.0%` | `0.00R` | Extended hours activo pero sin fills cerrados |
| **ALTA VOLATILIDAD (NATR > 3.0%)** | `0` | `0` | `0` | `0` | `$0.00` | `0.0%` | `0.00R` | Filtrado preventivo por PreTradeRiskGuard / CircuitBreaker |
| **VOLATILIDAD NORMAL (NATR <= 3.0%)** | `5` | `0` | `3` | `2` | `$44.46` | `50.0%` | `+0.21R` | Operaciones dentro de umbrales normales |

---

## 6. Registro Forense Completo de Operaciones (Ledger de Costos)

| ID | Símbolo | Lado | Qty | Entrada | Salida | P&L Bruto | Slippage | Spread Cost | Comisiones | P&L Neto | R-Mult | Duración | MFE | MAE | Motivo Salida |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tr_MSFT_b_ent_f3e4cc2da3_1` | **MSFT** | LONG | 10.0 | $400.14 | $414.86 | $150.00 | $1.14 | $1.63 | $0.00 | 🟢 $147.23 | `+1.45R` | `0.0m` | +3.8% | 0.0% | `TAKE_PROFIT` |
| `tr_TSLA_b_ent_f2d4dc27bc_2` | **TSLA** | LONG | 15.0 | $220.13 | $213.28 | $-99.00 | $1.17 | $2.60 | $0.00 | 🔴 $-102.77 | `-1.02R` | `0.0m` | +0.0% | -3.2% | `STOP_LOSS` |

> 📐 **Identidad Contable Auditada:** `P&L Neto = P&L Bruto − Slippage − Spread Cost − Comisiones` (tolerancia de redondeo: ±$0.00).