"""Script Autoejecutable Todo-en-Uno: Configura .env, Aplica Parches de Soporte y Prueba en Vivo.

Uso en tu computadora (dentro de la carpeta TRADING_NUEVO):
    python aplicar_mejoras_y_probar_en_vivo.py

Opciones adicionales:
    python aplicar_mejoras_y_probar_en_vivo.py --auto-trade   # Ejecuta un ciclo real con envío de órdenes a Alpaca Paper
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import httpx


ENV_CONTENT = """# ============================================
# AI QUANT TRADING PLATFORM
# Environment Configuration
# ============================================

# ── System ──────────────────────────────────
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false

# ── Trading Mode ────────────────────────────
# CRITICAL: Must be "paper" during development
TRADING_MODE=paper

# ── Database & Redis ────────────────────────
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=trading_platform
DATABASE_USER=trading
DATABASE_PASSWORD=CHANGE_ME_trading_secure_password
DATABASE_URL=postgresql+asyncpg://${DATABASE_USER}:${DATABASE_PASSWORD}@${DATABASE_HOST}:${DATABASE_PORT}/${DATABASE_NAME}

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_URL=redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}

# ── Alpaca Markets (Paper) ──────────────────
ALPACA_API_KEY=PKIM6TQ3U4FZ5BDXDJQSJAPXGA
ALPACA_SECRET_KEY=CAQc5h4UL4uz1bgoZCJmTx9iA2R6ZUyhhtaoo3kjeiN6
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
ALPACA_STREAM_URL=wss://stream.data.alpaca.markets/v2
ALPACA_FEED=iex

# ── Google Gemini AI (News & Catalyst Intelligence) ──
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# ── SEC EDGAR ───────────────────────────────
SEC_EDGAR_USER_AGENT=AIQuantPlatform contact@example.com
SEC_EDGAR_RATE_LIMIT=10

# ── FRED API ────────────────────────────────
FRED_API_KEY=13e7e9e520e18cd8eb5f25b580222c10

# ── Twelve Data (5 keys para rotación) ──────
# Cada key: 800 credits/día, 8 req/min → Total: 4000 credits/día, 40 req/min
TWELVE_DATA_API_KEY=631824861e2748f98a40c6643c919759
TWELVE_DATA_API_KEY_2=18f2907e50834006bfd5d1faa4417bf7
TWELVE_DATA_API_KEY_3=c4b0a5cbb61748c6aebcc24aa72288a0
TWELVE_DATA_API_KEY_4=9f138efdae9a4575861a5224525f49fa
TWELVE_DATA_API_KEY_5=44a78879690d48e2a8b5908fa6e8e65f

# ── Finnhub (Optional) ──────────────────────
FINNHUB_API_KEY=

# ── Notifications ───────────────────────────
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
ALERT_EMAIL_TO=

# ── Risk Defaults ───────────────────────────
INITIAL_CAPITAL=100000
RISK_PER_TRADE=0.01
MAX_DAILY_LOSS=0.03
MAX_WEEKLY_LOSS=0.05
MAX_DRAWDOWN=0.10
MAX_POSITION_SIZE=0.05
MAX_SECTOR_EXPOSURE=0.25
MAX_TRADES_PER_DAY=20
MAX_ORDERS_PER_MINUTE=5
OPERATIONAL_CAPITAL=1000.0
MAX_OPERATIONAL_POSITION_SIZE=0.20
MIN_PROFIT_LOCK_USD=20.0
MAX_PROFIT_GIVEBACK_PCT=35.0
"""


def write_env_and_load(root: Path) -> None:
    """Writes .env if not present or updates keys, and loads variables into os.environ."""
    env_path = root / ".env"
    existing_gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    existing_gemini_key = val

    content = ENV_CONTENT
    if existing_gemini_key:
        content = content.replace("GEMINI_API_KEY=", f"GEMINI_API_KEY={existing_gemini_key}")

    env_path.write_text(content, encoding="utf-8")
    for line in content.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ[k.strip()] = v.strip()
    print(f"[+] Archivo .env actualizado y cargado en: {env_path}")


def apply_support_patches(root: Path) -> None:
    """Applies surgical patches to ranker.py, execution_engine.py, and pre_trade_checker.py."""
    ranker_p = root / "services/opportunity_engine/ranker.py"
    if ranker_p.exists():
        txt = ranker_p.read_text(encoding="utf-8")
        txt = txt.replace("batch_size = min(len(symbols), 5)", "batch_size = min(len(symbols), 15)")
        txt = txt.replace("outputsize=60", "outputsize=220")
        ranker_p.write_text(txt, encoding="utf-8")
        print("[+] Parche aplicado: services/opportunity_engine/ranker.py (outputsize=220, batch=15)")

    ee_p = root / "services/broker_engine/execution_engine.py"
    if ee_p.exists():
        txt = ee_p.read_text(encoding="utf-8")
        txt = txt.replace(
            'risk_pct = float(getattr(settings.risk, "risk_per_trade_pct", 0.01))',
            'risk_pct = float(getattr(settings.risk, "risk_per_trade", getattr(settings.risk, "risk_per_trade_pct", 0.01)))',
        )
        ee_p.write_text(txt, encoding="utf-8")
        print("[+] Parche aplicado: services/broker_engine/execution_engine.py (risk_per_trade)")

    ptc_p = root / "services/risk_engine/pre_trade_checker.py"
    if ptc_p.exists():
        txt = ptc_p.read_text(encoding="utf-8")
        for target in [
            "portfolio_state.daily_pnl_pct <= -(self.settings.max_daily_loss * 100.0)",
            "portfolio_state.drawdown_pct >= (self.settings.max_drawdown * 100.0)",
            "portfolio_state.daily_trades_count >= self.settings.max_trades_per_day",
            "signal_score < self.settings.min_signal_score",
            "projected_sector_exp > self.settings.max_sector_exposure",
            "notional > max_allowed_notional",
        ]:
            old_if = f"if {target}:"
            new_if = f"if not is_exit and {target}:"
            if old_if in txt:
                txt = txt.replace(old_if, new_if)
        ptc_p.write_text(txt, encoding="utf-8")
        print("[+] Parche aplicado: services/risk_engine/pre_trade_checker.py (desbloqueo de salidas is_exit=True)")


async def probe_live_apis() -> None:
    """Tests live connectivity against Alpaca Paper, Alpaca News, Twelve Data (5 keys), and FRED."""
    print("\n" + "=" * 72)
    print("  1. PRUEBA DE CONECTIVIDAD EN TIEMPO REAL CONTRA TUS APIS (.env)")
    print("=" * 72)

    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_sec = os.environ.get("ALPACA_SECRET_KEY", "")
    alpaca_base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    alpaca_data = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
    headers = {"APCA-API-KEY-ID": alpaca_key, "APCA-API-SECRET-KEY": alpaca_sec}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r_acc = await client.get(f"{alpaca_base}/v2/account", headers=headers)
            if r_acc.status_code == 200:
                acc = r_acc.json()
                print(
                    f"[OK] Alpaca Paper Account: ID={acc.get('id')} | Status={acc.get('status')} | "
                    f"Equity=${float(acc.get('equity', 0)):,.2f} | Cash=${float(acc.get('cash', 0)):,.2f} | "
                    f"BuyingPower=${float(acc.get('buying_power', 0)):,.2f}"
                )
            else:
                print(f"[!] Alpaca Account HTTP {r_acc.status_code}: {r_acc.text[:160]}")

            r_pos = await client.get(f"{alpaca_base}/v2/positions", headers=headers)
            if r_pos.status_code == 200:
                positions = r_pos.json()
                print(f"[OK] Alpaca Posiciones Abiertas ({len(positions)}):")
                for p in positions:
                    print(
                        f"     - {p.get('symbol'):<6} | Qty={p.get('qty')} | Entrada=${float(p.get('avg_entry_price', 0)):.2f} | "
                        f"Actual=${float(p.get('current_price', 0)):.2f} | PnL=${float(p.get('unrealized_pl', 0)):+.2f} "
                        f"({float(p.get('unrealized_plpc', 0))*100:+.2f}%)"
                    )
        except Exception as exc:
            print(f"[!] Error conectando a Alpaca Paper: {exc}")

        try:
            r_news = await client.get(
                f"{alpaca_data}/v1beta1/news",
                headers=headers,
                params={"symbols": "NVDA,AAPL,MSFT,TSLA,GOOGL,AMD,NFLX,NOW", "limit": 10},
            )
            if r_news.status_code == 200:
                news_items = r_news.json().get("news", [])
                print(f"[OK] Alpaca News API: {len(news_items)} noticias recientes recibidas en vivo.")
                for item in news_items[:4]:
                    print(f"     * [{','.join(item.get('symbols', []))}] {item.get('headline')} ({item.get('source')})")
            else:
                print(f"[!] Alpaca News HTTP {r_news.status_code}: {r_news.text[:160]}")
        except Exception as exc:
            print(f"[!] Error conectando a Alpaca News: {exc}")

        td_keys = [
            os.environ.get("TWELVE_DATA_API_KEY", ""),
            os.environ.get("TWELVE_DATA_API_KEY_2", ""),
            os.environ.get("TWELVE_DATA_API_KEY_3", ""),
            os.environ.get("TWELVE_DATA_API_KEY_4", ""),
            os.environ.get("TWELVE_DATA_API_KEY_5", ""),
        ]
        td_keys = [k for k in td_keys if k]
        print(f"[OK] Verificando rotación de {len(td_keys)} llaves de Twelve Data:")
        for idx, k in enumerate(td_keys, 1):
            try:
                r_td = await client.get(
                    "https://api.twelvedata.com/api_usage",
                    params={"apikey": k},
                )
                if r_td.status_code == 200:
                    u = r_td.json()
                    print(
                        f"     - Key #{idx} ({k[:6]}...{k[-4:]}): Uso actual={u.get('current_usage', 0)}/{u.get('plan_limit', 800)} créditos"
                    )
            except Exception as exc:
                print(f"     - Key #{idx}: Error de red ({exc})")

        fred_key = os.environ.get("FRED_API_KEY", "")
        if fred_key:
            try:
                r_fred = await client.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": "VIXCLS", "api_key": fred_key, "file_type": "json", "sort_order": "desc", "limit": 3},
                )
                if r_fred.status_code == 200:
                    obs = r_fred.json().get("observations", [])
                    latest_vix = next((o.get("value") for o in obs if o.get("value") not in (None, ".")), "N/A")
                    print(f"[OK] FRED Macro API: Conectada | Último VIXCLS={latest_vix}")
            except Exception as exc:
                print(f"[!] Error conectando a FRED API: {exc}")


async def run_live_worker_cycle(auto_trade: bool = False) -> None:
    """Runs one full QuantTradingWorker cycle with the 10 Super-Agents."""
    print("\n" + "=" * 72)
    print(f"  2. EJECUTANDO CICLO DE LOS 10 SUPER-AGENTES (auto_trade={auto_trade})")
    print("=" * 72)
    from apps.worker.quant_trading_worker import get_global_worker
    worker = get_global_worker()
    snapshot = await worker.execute_cycle(auto_trade=auto_trade)

    pf = snapshot.get("portfolio", {})
    print(
        f"Resumen de Cartera -> Equity: ${pf.get('portfolio_value', 0):,.2f} | "
        f"Cash: ${pf.get('cash', 0):,.2f} | PnL Abierto: ${pf.get('unrealized_pnl', 0):+,.2f} | "
        f"Posiciones Abiertas: {pf.get('open_positions_count', 0)}"
    )
    print("Top Candidatos Escaneados:", snapshot.get("latest_opportunities", []))
    print("Puntajes de Noticias (News & Gemini):", snapshot.get("news_scores", {}))
    print("\nÚltimos Eventos del Motor:")
    for log in snapshot.get("activity_logs", [])[:8]:
        print(f"  [{log.get('time')}] ({log.get('level', '').upper()}) {log.get('message')}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Configura .env, aplica mejoras y prueba TRADING_NUEVO en tiempo real")
    parser.add_argument("--auto-trade", action="store_true", help="Habilita el envío real de órdenes a Alpaca Paper en este ciclo")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    write_env_and_load(root)
    apply_support_patches(root)
    await probe_live_apis()
    await run_live_worker_cycle(auto_trade=args.auto_trade)


if __name__ == "__main__":
    asyncio.run(main())
