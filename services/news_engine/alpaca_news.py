"""Alpaca News API adapter for real-time financial news ingestion."""

from datetime import datetime, timezone
from typing import List, Optional
import httpx
import structlog
from core.config import get_settings
from services.news_engine.models import NewsArticle

logger = structlog.get_logger(__name__)


class AlpacaNewsAdapter:
    """Client for fetching structured financial news from the Alpaca News API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.alpaca.api_key
        configured_secret = settings.alpaca.secret_key
        if hasattr(configured_secret, "get_secret_value"):
            configured_secret = configured_secret.get_secret_value()
        self.secret_key = secret_key or str(configured_secret)
        # Alpaca news lives at data.alpaca.markets/v1beta1/news
        self.base_url = (base_url or "https://data.alpaca.markets").rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    async def get_latest_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 50,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        include_content: bool = False,
    ) -> List[NewsArticle]:
        """Fetches latest corporate news articles matching the criteria.

        Args:
            symbols: List of ticker symbols to filter by.
            limit: Maximum articles to fetch (1-50).
            start: Start datetime filter.
            end: End datetime filter.
            include_content: Whether to fetch full article body text.

        Returns:
            List[NewsArticle]: Normalized news article models.
        """
        endpoint = f"{self.base_url}/v1beta1/news"
        params: dict = {"limit": min(50, max(1, limit))}

        if symbols:
            params["symbols"] = ",".join(symbols)
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()
        if include_content:
            params["include_content"] = "true"

        articles: List[NewsArticle] = []

        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=10.0) as client:
                resp = await client.get(endpoint, params=params)
                if resp.status_code != 200:
                    logger.warning(
                        "alpaca_news_fetch_failed",
                        status_code=resp.status_code,
                        detail=resp.text[:200],
                    )
                    return []

                payload = resp.json()
                raw_news = payload.get("news", [])
                for item in raw_news:
                    created_at_str = item.get("created_at")
                    try:
                        pub_dt = (
                            datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                            if created_at_str
                            else datetime.now(timezone.utc)
                        )
                    except Exception:
                        pub_dt = datetime.now(timezone.utc)

                    article = NewsArticle(
                        id=str(item.get("id", "")),
                        headline=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        content=item.get("content") if include_content else None,
                        source=item.get("source", "Alpaca"),
                        url=item.get("url"),
                        symbols=item.get("symbols", []),
                        published_at=pub_dt,
                    )
                    articles.append(article)

        except Exception as exc:
            logger.error("alpaca_news_network_error", error=str(exc))
            return []

        return articles
