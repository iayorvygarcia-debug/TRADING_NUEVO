# AI Quant Trading Platform

Professional AI-powered quantitative trading platform for US Equities & ETFs.

## ⚠️ IMPORTANT

This system is in **PAPER TRADING** mode by default. Live trading requires explicit multi-step activation.

## Quick Start

```bash
# 1. Copy environment template
cp .env.example .env
# Edit .env with your API keys

# 2. Start infrastructure (PostgreSQL + TimescaleDB + Redis)
docker compose up -d

# 3. Install Python dependencies
pip install -e ".[dev]"

# 4. Run the API server
uvicorn apps.api.main:app --reload --port 8000

# 5. Check health
curl http://localhost:8000/health
```

## Architecture

```
╔══════════════════════════════════════════╗
║          PLANO DE DATOS                  ║
║  Market │ News │ SEC │ Macro │ Funds     ║
╠══════════════════════════════════════════╣
║       PLANO DE INTELIGENCIA              ║
║  Features │ Technical │ NLP │ ML │ Strat ║
╠══════════════════════════════════════════╣
║          PLANO DE TRADING                ║
║  Portfolio │ Risk │ Execution │ Broker   ║
╠══════════════════════════════════════════╣
║        PLANO DE CONTROL 24/7             ║
║  Watchdog │ Monitor │ Audit │ Kill Switch║
╚══════════════════════════════════════════╝
```

## Project Structure

```
trading-platform/
├── apps/api/          # FastAPI Gateway
├── core/              # Config, logging, DB, Redis, events
├── services/          # Domain services (market data, risk, etc.)
├── strategies/        # Trading strategies
├── ml/                # Machine learning pipeline
├── backtesting/       # Backtesting engine
├── database/          # SQL schemas and migrations
├── infra/             # Docker, monitoring
├── tests/             # Test suite
└── docs/              # Documentation
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Database](docs/DATABASE.md)
- [Risk Management](docs/RISK.md)
- [Strategies](docs/STRATEGIES.md)

## License

Proprietary — All rights reserved.
