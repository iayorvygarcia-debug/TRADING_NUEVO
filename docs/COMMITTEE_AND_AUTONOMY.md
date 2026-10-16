# Comité autónomo de análisis y ejecución

El worker ejecuta un comité de **10 agentes deterministas**. Cada agente analiza el mismo vector de mercado, pero con una familia de señales distinta. Los agentes producen señales tipadas; no tienen acceso al broker ni pueden enviar órdenes.

## Agentes

| Grupo | Agente | Enfoque |
|---|---|---|
| Estrategias existentes | `trend_following` | Tendencia, EMA, Supertrend y ADX |
| Estrategias existentes | `mean_reversion` | Reversión estadística y osciladores |
| Estrategias existentes | `breakout` | Rupturas, rango y expansión de volumen |
| Estrategias existentes | `vwap_reversion` | Distancia respecto a VWAP |
| Estrategias existentes | `mtf_confluence` | Confluencia de temporalidades |
| Comité adicional | `momentum_confirmation` | RSI y MACD |
| Comité adicional | `volume_flow` | RVOL y CMF |
| Comité adicional | `regime_classifier` | ADX y dirección de Supertrend |
| Comité adicional | `vwap_reversion_guard` | Exhaustión y retorno a VWAP |
| Comité adicional | `volatility_breakout` | Squeeze, volumen y dirección |

## Regla de consenso

El `SignalCombiner` agrupa las señales por dirección. Si existen señales BUY y SELL simultáneas, la decisión es HOLD. Para enviar una orden se requiere una puntuación mínima y una proporción mínima de consenso de **0.60**. Si el comité no confirma la dirección del Opportunity Engine, el sistema registra `NO TRADE` y no utiliza una señal de respaldo.

Esta regla es deliberadamente conservadora: es preferible perder una oportunidad que convertir una discrepancia de agentes en una operación sin confirmación.

## Noticias y veto

Cada ciclo intenta obtener noticias recientes de Alpaca para los candidatos, las procesa mediante deduplicación, sentimiento, catalizadores y decaimiento temporal. Un símbolo con `composite_news_score <= -50` se bloquea antes del análisis de estrategias y no puede llegar al Risk Gatekeeper.

La ausencia de noticias no genera por sí sola una señal de compra o venta. Las noticias funcionan como contexto y protección, no como sustituto de precios, liquidez y riesgo.

## Operación autónoma

El worker se mantiene en un ciclo continuo de 60 segundos mientras el proceso de la API permanezca activo. La autonomía 24/7 requiere ejecutar la API con un supervisor de procesos, un contenedor reiniciable o un servicio persistente. El código no pretende abrir posiciones nuevas cuando el mercado regular está cerrado; fuera de sesión realiza vigilancia, reconciliación y controles.

## Modo seguro

La instalación inicial mantiene `auto_trade=false` y `TRADING_MODE=paper`. Activa `auto_trade=true` únicamente después de validar Paper Trading, los límites de riesgo, las respuestas del proveedor de datos y el Kill Switch. Ningún agente puede saltarse el Risk Gatekeeper ni enviar órdenes directamente.

> El comité aumenta la trazabilidad y la disciplina de decisión; no garantiza rentabilidad. Debe evaluarse con backtesting fuera de muestra, paper trading prolongado y métricas de slippage, drawdown, tasa de aciertos y expectativa neta.
