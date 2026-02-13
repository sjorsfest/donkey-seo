"""DataForSEO API integration for keyword research, metrics, AND SERP.

Single provider for all SEO data needs - simpler ops, consistent fields, one billing model.
"""

import base64
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.core.exceptions import APIKeyMissingError, ExternalAPIError, RateLimitExceededError

logger = logging.getLogger(__name__)


class DataForSEOClient:
    """Client for DataForSEO API.

    Provides methods for:
    - Keyword suggestions
    - Related keywords
    - Keyword questions (People Also Ask)
    - Keyword metrics (volume, CPC, difficulty)
    - SERP results (single provider for everything!)
    """

    BASE_URL = "https://api.dataforseo.com/v3"

    def __init__(
        self,
        login: str | None = None,
        password: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.login = login or settings.dataforseo_login
        self.password = password or settings.dataforseo_password
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

        if not self.login or not self.password:
            raise APIKeyMissingError("DataForSEO")

    @property
    def _auth_header(self) -> str:
        """Generate Basic Auth header."""
        credentials = f"{self.login}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    async def __aenter__(self) -> "DataForSEOClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": self._auth_header,
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client must be used as async context manager")
        return self._client

    async def _make_request(
        self,
        endpoint: str,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Make a POST request to DataForSEO API."""
        logger.info("DataForSEO API request", extra={"endpoint": endpoint})
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = await self.client.post(url, json=data)

            if response.status_code == 429:
                logger.warning("DataForSEO rate limit hit", extra={"endpoint": endpoint})
                raise RateLimitExceededError("DataForSEO")

            response.raise_for_status()
            result = response.json()

            if result.get("status_code") != 20000:
                logger.warning("DataForSEO API error", extra={"endpoint": endpoint, "status": result.get("status_message")})
                raise ExternalAPIError(
                    "DataForSEO",
                    result.get("status_message", "Unknown error"),
                )

            # Extract tasks results
            tasks = result.get("tasks", [])
            results = []
            for task in tasks:
                if task.get("status_code") == 20000 and task.get("result"):
                    results.extend(task["result"])

            return results

        except httpx.HTTPError as e:
            logger.warning("DataForSEO HTTP error", extra={"endpoint": endpoint, "error": str(e)})
            raise ExternalAPIError("DataForSEO", str(e)) from e

    async def get_keyword_suggestions(
        self,
        seeds: list[str],
        location_code: int = 2840,  # US
        language_code: str = "en",
        limit: int = 700,
    ) -> list[dict[str, Any]]:
        """Get keyword suggestions for seed keywords.

        Args:
            seeds: Seed keywords to expand (batched into one API call)
            location_code: DataForSEO location code (2840 = US)
            language_code: Language code (en, de, etc.)
            limit: Maximum keywords to return

        Returns:
            List of keyword suggestions with metrics (deduplicated)
        """
        if not seeds:
            return []

        logger.info("Fetching keyword suggestions", extra={"seeds": len(seeds), "location": location_code, "limit": limit})
        data = [
            {
                "keywords": seeds,
                "location_code": location_code,
                "language_code": language_code,
                "limit": limit,
            }
        ]

        results = await self._make_request(
            "dataforseo_labs/google/keyword_ideas/live",
            data,
        )

        keywords = []
        seen = set()
        for result in results:
            for item in result.get("items", []):
                kw_text = (item.get("keyword") or "").lower()
                if kw_text and kw_text not in seen:
                    seen.add(kw_text)
                    info = item.get("keyword_info") or {}
                    keywords.append({
                        "keyword": item.get("keyword"),
                        "search_volume": info.get("search_volume"),
                        "cpc": info.get("cpc"),
                        "competition": info.get("competition"),
                        "competition_index": info.get("competition_level"),
                        "monthly_searches": info.get("monthly_searches", []),
                    })

        return keywords

    async def get_keyword_questions(
        self,
        keywords: list[str],
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get question-based keywords (People Also Ask).

        Args:
            keywords: Base keywords to generate question variants for
            location_code: DataForSEO location code
            language_code: Language code
            limit: Maximum questions to return

        Returns:
            List of question keywords
        """
        if not keywords:
            return []

        # Build all question variants in one batch
        question_words = ["how", "what", "why"]
        question_seeds = [f"{q} {kw}" for kw in keywords for q in question_words]

        logger.info("Fetching keyword questions", extra={"base_keywords": len(keywords), "question_seeds": len(question_seeds), "limit": limit})

        data = [
            {
                "keywords": question_seeds,
                "location_code": location_code,
                "language_code": language_code,
                "limit": limit,
            }
        ]

        results = await self._make_request(
            "dataforseo_labs/google/keyword_ideas/live",
            data,
        )

        all_question_words = ["how", "what", "why", "when", "where", "which", "who"]
        questions = []
        seen = set()
        for result in results:
            for item in result.get("items", []):
                kw_text = item.get("keyword") or ""
                kw_lower = kw_text.lower()
                if kw_lower not in seen and any(kw_lower.startswith(w) for w in all_question_words):
                    seen.add(kw_lower)
                    info = item.get("keyword_info") or {}
                    questions.append({
                        "keyword": kw_text,
                        "search_volume": info.get("search_volume"),
                        "cpc": info.get("cpc"),
                    })

        return questions[:limit]

    async def get_keyword_metrics(
        self,
        keywords: list[str],
        location_code: int = 2840,
        language_code: str = "en",
    ) -> list[dict[str, Any]]:
        """Get search metrics for a list of keywords.

        Args:
            keywords: List of keywords (max 700 per request)
            location_code: DataForSEO location code
            language_code: Language code

        Returns:
            List of keyword metrics including difficulty
        """
        logger.info("Fetching keyword metrics", extra={"keyword_count": len(keywords), "location": location_code})
        if not keywords:
            return []

        batch_size = 700
        all_metrics = []

        for i in range(0, len(keywords), batch_size):
            batch = keywords[i : i + batch_size]

            data = [
                {
                    "keywords": batch,
                    "location_code": location_code,
                    "language_code": language_code,
                }
            ]

            results = await self._make_request(
                "dataforseo_labs/google/keyword_overview/live",
                data,
            )

            for result in results:
                for item in result.get("items", []):
                    info = item.get("keyword_info") or {}
                    props = item.get("keyword_properties") or {}
                    monthly = info.get("monthly_searches") or []
                    trend = [m.get("search_volume", 0) for m in monthly] if monthly else []

                    all_metrics.append({
                        "keyword": item.get("keyword"),
                        "search_volume": info.get("search_volume"),
                        "cpc": info.get("cpc"),
                        "competition": info.get("competition"),
                        "competition_index": info.get("competition_level"),
                        "difficulty": props.get("keyword_difficulty"),
                        "trend_data": trend,
                    })

        return all_metrics

    async def get_keyword_difficulty(
        self,
        keywords: list[str],
        location_code: int = 2840,
        language_code: str = "en",
    ) -> list[dict[str, Any]]:
        """Get keyword difficulty scores.

        Args:
            keywords: List of keywords (max 1000 per request)
            location_code: DataForSEO location code
            language_code: Language code

        Returns:
            List of keyword difficulty scores
        """
        if not keywords:
            return []

        batch_size = 1000
        all_difficulty = []

        for i in range(0, len(keywords), batch_size):
            batch = keywords[i : i + batch_size]

            data = [
                {
                    "keywords": batch,
                    "location_code": location_code,
                    "language_code": language_code,
                }
            ]

            try:
                results = await self._make_request(
                    "dataforseo_labs/google/bulk_keyword_difficulty/live",
                    data,
                )

                for result in results:
                    for item in result.get("items", []):
                        all_difficulty.append({
                            "keyword": item.get("keyword"),
                            "difficulty": item.get("keyword_difficulty"),
                        })
            except ExternalAPIError:
                for kw in batch:
                    all_difficulty.append({
                        "keyword": kw,
                        "difficulty": None,
                    })

        return all_difficulty

    # ========== SERP Methods (Single Provider!) ==========

    async def get_serp_results(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        device: str = "desktop",
        depth: int = 10,
    ) -> dict[str, Any]:
        """Get SERP results for a keyword.

        Args:
            keyword: Keyword to search
            location_code: DataForSEO location code
            language_code: Language code
            device: "desktop" or "mobile"
            depth: Number of results to return (max 100)

        Returns:
            Dict with organic results, SERP features, and metadata
        """
        logger.info("Fetching SERP results", extra={"keyword": keyword, "device": device})
        data = [
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
                "device": device,
                "depth": depth,
            }
        ]

        results = await self._make_request(
            "serp/google/organic/live/regular",
            data,
        )

        if not results:
            return {
                "keyword": keyword,
                "organic_results": [],
                "serp_features": [],
                "provider_response_id": None,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

        result = results[0]

        # Extract organic results
        organic_results = []
        for item in result.get("items", []):
            if item.get("type") == "organic":
                organic_results.append({
                    "position": item.get("rank_absolute"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "domain": item.get("domain"),
                    "snippet": item.get("description"),
                    "breadcrumb": item.get("breadcrumb"),
                })

        # Extract SERP features
        serp_features = []
        feature_types = {
            "featured_snippet": "featured_snippet",
            "people_also_ask": "paa",
            "knowledge_graph": "knowledge_graph",
            "local_pack": "local",
            "video": "video",
            "images": "images",
            "shopping": "shopping",
            "top_stories": "news",
            "related_searches": "related_searches",
        }

        for item in result.get("items", []):
            item_type = item.get("type", "")
            if item_type in feature_types:
                serp_features.append(feature_types[item_type])

        return {
            "keyword": keyword,
            "organic_results": organic_results,
            "serp_features": list(set(serp_features)),
            "total_results": result.get("se_results_count"),
            "provider_response_id": result.get("id"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_serp_batch(
        self,
        keywords: list[str],
        location_code: int = 2840,
        language_code: str = "en",
        device: str = "desktop",
        depth: int = 10,
    ) -> list[dict[str, Any]]:
        """Get SERP results for multiple keywords.

        Args:
            keywords: List of keywords to search
            location_code: DataForSEO location code
            language_code: Language code
            device: "desktop" or "mobile"
            depth: Number of results to return per keyword

        Returns:
            List of SERP results
        """
        results = []
        for keyword in keywords:
            try:
                result = await self.get_serp_results(
                    keyword=keyword,
                    location_code=location_code,
                    language_code=language_code,
                    device=device,
                    depth=depth,
                )
                results.append(result)
            except ExternalAPIError:
                results.append({
                    "keyword": keyword,
                    "organic_results": [],
                    "serp_features": [],
                    "error": "Failed to fetch SERP",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                })
        return results

    async def check_serp_overlap(
        self,
        keywords: list[str],
        location_code: int = 2840,
        language_code: str = "en",
        top_n: int = 10,
    ) -> float:
        """Check SERP overlap between keywords (for clustering).

        Returns a score 0-1 indicating how much the top results overlap.
        High overlap suggests keywords should be in the same cluster.

        Args:
            keywords: List of keywords to compare (2-5 recommended)
            location_code: DataForSEO location code
            language_code: Language code
            top_n: Number of top results to compare

        Returns:
            Overlap score 0-1 (1 = identical SERPs)
        """
        if len(keywords) < 2:
            return 1.0

        # Get SERP results for all keywords
        serp_results = await self.get_serp_batch(
            keywords=keywords,
            location_code=location_code,
            language_code=language_code,
            depth=top_n,
        )

        # Extract top domains from each SERP
        domain_sets = []
        for serp in serp_results:
            domains = set()
            for item in serp.get("organic_results", [])[:top_n]:
                if domain := item.get("domain"):
                    domains.add(domain)
            domain_sets.append(domains)

        if not domain_sets or not any(domain_sets):
            return 0.0

        # Calculate pairwise overlap
        total_overlap = 0.0
        pair_count = 0

        for i in range(len(domain_sets)):
            for j in range(i + 1, len(domain_sets)):
                if domain_sets[i] and domain_sets[j]:
                    intersection = len(domain_sets[i] & domain_sets[j])
                    union = len(domain_sets[i] | domain_sets[j])
                    if union > 0:
                        total_overlap += intersection / union
                        pair_count += 1

        return total_overlap / pair_count if pair_count > 0 else 0.0


# Location codes for common countries
LOCATION_CODES = {
    "us": 2840,
    "uk": 2826,
    "de": 2276,
    "fr": 2250,
    "es": 2724,
    "it": 2380,
    "nl": 2528,
    "au": 2036,
    "ca": 2124,
    "in": 2356,
}


def get_location_code(locale: str) -> int:
    """Convert locale string to DataForSEO location code."""
    country = locale.split("-")[-1].lower() if "-" in locale else locale.lower()
    return LOCATION_CODES.get(country, 2840)  # Default to US
