# INSTRUCCIONES DE INSTALACIÓN Y ACTUALIZACIÓN (TRADING_NUEVO)

Este archivo ZIP contiene **únicamente los archivos modificados y nuevos** para actualizar tu sistema de trading cuantitativo con:
1. **Los 10 Super-Agentes mejorados** (`strategies/committee_agents.py`) sensibles a noticias en vivo, flujo institucional y régimen macro, corrigiendo el bloqueo de `vwap_reversion_guard`.
2. **Conector de Inteligencia Financiera con Google Gemini** (`services/news_engine/gemini_news_analyzer.py`, `catalyst_extractor.py`, `impact.py`, `models.py`).
3. **Resolución Inteligente de Conflictos por Supermayoría** (`services/strategy_engine/signal_combiner.py`).
4. **Desbloqueo de órdenes de salida (SELL/Cierre)** en `services/risk_engine/pre_trade_checker.py`.
5. **Integración completa en el Worker 24/7** (`apps/worker/quant_trading_worker.py`) y configuración `.env` lista con tus llaves de Alpaca Paper, FRED y 5 llaves de Twelve Data.

---

## PASO 1: Descomprimir y Reemplazar en tu Carpeta del Proyecto

1. Abre la carpeta raíz de tu repositorio local `TRADING_NUEVO` (o `/home/u114856156/trading_platform` en tu servidor Hostinger).
2. Descomprime el contenido de `TRADING_NUEVO_MEJORAS.zip` directamente dentro de la carpeta raíz de tu proyecto.
3. Cuando tu sistema operativo te pregunte si deseas **Reemplazar los archivos existentes**, selecciona **"Sí a todo" / "Reemplazar todos"**, ya que todos los archivos vienen organizados en sus subcarpetas exactas (`strategies/`, `services/`, `core/`, `apps/`).

### Lista exacta de rutas incluidas en este ZIP:
- `.env` *(Tus credenciales configuradas de Alpaca Paper, FRED y las 5 API keys de Twelve Data)*
- `strategies/committee_agents.py` *(Los 10 Super-Agentes conectados a noticias/Gemini)*
- `strategies/trend_following.py`
- `strategies/mean_reversion.py`
- `services/news_engine/gemini_news_analyzer.py` *(Nuevo módulo de IA con Google Gemini)*
- `services/news_engine/models.py`
- `services/news_engine/catalyst_extractor.py`
- `services/news_engine/sentiment.py`
- `services/news_engine/dedup.py`
- `services/news_engine/impact.py`
- `services/news_engine/__init__.py`
- `services/news_engine/alpaca_news.py`
- `services/strategy_engine/signal_combiner.py` *(Votación ponderada y supermayoría)*
- `services/opportunity_engine/ranker.py`
- `services/technical_analysis/feature_engine.py`
- `services/risk_engine/pre_trade_checker.py` *(Exención para que cierres/salidas nunca se bloqueen)*
- `services/broker_engine/execution_engine.py`
- `services/broker_engine/alpaca_broker.py`
- `services/ml_pipeline/models.py`
- `services/ml_pipeline/trainer.py`
- `services/ml_pipeline/predictor.py`
- `services/ml_pipeline/governance.py`
- `services/macro_sec/fred_macro.py`
- `services/macro_sec/sec_edgar.py`
- `core/config.py` *(Soporte para GEMINI_API_KEY y GEMINI_MODEL)*
- `core/logging.py`
- `core/redis_client.py`
- `apps/worker/quant_trading_worker.py` *(Worker 24/7 integrado con Gemini y salidas por comité)*
- `aplicar_mejoras_y_probar_en_vivo.py` *(Script de verificación y prueba en vivo)*

---

## PASO 2: (Opcional) Activar tu API Key de Google Gemini
En el archivo `.env` verás la variable:
```env
GEMINI_API_KEY=tu_clave_de_google_ai_studio_aqui
GEMINI_MODEL=gemini-2.5-flash
```
- Si colocas tu API Key de Google AI Studio, el motor consultará `gemini-2.5-flash` en tiempo real para cada noticia de Alpaca.
- Si la dejas vacía, el sistema funcionará automáticamente con el motor determinístico mejorado (detector de *sell-the-news*, catalizadores de semiconductores/IA y mapa de contagio `TSM`/`ASML`/`NVDA`/`AMD`).

---

## PASO 3: Probar las Mejoras en Vivo
Desde tu terminal dentro de la carpeta del proyecto, ejecuta:
```bash
python aplicar_mejoras_y_probar_en_vivo.py
```
O para iniciar el daemon autónomo 24/7:
```bash
python apps/worker/quant_trading_worker.py
```

---

## PASO 4: Subir los Cambios a tu GitHub
Para guardar estos cambios en tu repositorio `https://github.com/iayorvygarcia-debug/TRADING_NUEVO` sin exponer tus llaves privadas (`.env`):

```bash
git add strategies/ services/ core/ apps/ aplicar_mejoras_y_probar_en_vivo.py
git commit -m "Upgrade 10 super-agents committee, Gemini news analyzer, and risk exit fixes"
git push origin main
```
