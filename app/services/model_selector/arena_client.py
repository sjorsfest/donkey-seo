"""Arena leaderboard collector."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ARENA_SITEMAP_URLS = (
    "https://arena.ai/sitemap.xml/0.xml",
    "https://arena.ai/sitemap.xml",
    "https://arena.ai/sitemap_index.xml",
)
ARENA_LEADERBOARD_URL = "https://arena.ai/nl/leaderboard"
ARENA_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


class ArenaClient:
    """Client for scraping Arena use-case leaderboard signals."""

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    async def fetch_available_use_cases(self) -> tuple[set[str], dict[str, Any]]:
        """Fetch valid use-case slugs from Arena sitemap(s), then leaderboard fallback."""
        errors: dict[str, str] = {}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for sitemap_url in ARENA_SITEMAP_URLS:
                try:
                    response = await client.get(sitemap_url, headers=ARENA_REQUEST_HEADERS)
                    response.raise_for_status()
                except Exception as exc:
                    errors[sitemap_url] = str(exc)
                    continue

                slugs = extract_use_case_slugs_from_sitemap_xml(response.text)
                if slugs:
                    return slugs, {
                        "source": "arena",
                        "ok": True,
                        "discovery_method": "sitemap",
                        "sitemap_url": sitemap_url,
                        "use_case_count": len(slugs),
                    }
                errors[sitemap_url] = "sitemap_returned_no_use_cases"

            # Fallback: scrape use_case options from leaderboard page HTML.
            try:
                response = await client.get(
                    ARENA_LEADERBOARD_URL,
                    params={"sort_by": "score", "time_period": "day"},
                    headers=ARENA_REQUEST_HEADERS,
                )
                response.raise_for_status()
                slugs = extract_use_case_slugs_from_leaderboard_html(response.text)
                if slugs:
                    return slugs, {
                        "source": "arena",
                        "ok": True,
                        "discovery_method": "leaderboard_fallback",
                        "leaderboard_url": ARENA_LEADERBOARD_URL,
                        "use_case_count": len(slugs),
                        "sitemap_errors": errors,
                    }
                errors["leaderboard_fallback"] = "leaderboard_returned_no_use_cases"
            except Exception as exc:
                errors["leaderboard_fallback"] = str(exc)

        logger.info("Arena use-case discovery unavailable, using direct slug mode")
        return set(), {
            "source": "arena",
            "ok": True,
            "discovery_method": "direct_slug_mode",
            "warning": errors,
            "use_case_count": 0,
        }

    async def fetch_leaderboard_scores(
        self,
        use_case_slug: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Fetch raw leaderboard scores for one Arena use case."""
        params = {
            "sort_by": "score",
            "time_period": "day",
            "use_case": use_case_slug,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    ARENA_LEADERBOARD_URL,
                    params=params,
                    headers=ARENA_REQUEST_HEADERS,
                )
                response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "Arena leaderboard fetch failed",
                extra={"use_case": use_case_slug, "error": str(exc)},
            )
            return {}, {
                "source": "arena",
                "ok": False,
                "use_case": use_case_slug,
                "error": str(exc),
            }

        rows = parse_arena_leaderboard_rows(response.text)
        raw_scores: dict[str, float] = {}
        for row in rows:
            name = row.get("model_name")
            score = row.get("score")
            if isinstance(name, str) and isinstance(score, float):
                raw_scores[name] = score

        return raw_scores, {
            "source": "arena",
            "ok": True,
            "use_case": use_case_slug,
            "rows_parsed": len(rows),
            "scores_kept": len(raw_scores),
        }


def extract_use_case_slugs_from_sitemap_xml(xml_text: str) -> set[str]:
    """Extract Arena use-case slugs from sitemap loc entries."""
    soup = BeautifulSoup(xml_text, "xml")
    slugs: set[str] = set()

    for loc in soup.find_all("loc"):
        if not loc.text:
            continue
        parsed = urlparse(loc.text)
        params = parse_qs(parsed.query)
        for value in params.get("use_case", []):
            cleaned = value.strip()
            if cleaned:
                slugs.add(cleaned)

    return slugs


def parse_arena_leaderboard_rows(html: str) -> list[dict[str, Any]]:
    """Parse leaderboard rows from Arena HTML payload."""
    rows: list[dict[str, Any]] = []
    soup = BeautifulSoup(html, "html.parser")

    for tr in soup.select("table tbody tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if len(cells) < 2:
            continue

        model_name = cells[0].strip()
        score = parse_score_from_text(cells[1])
        if not model_name or score is None:
            continue

        shown_cost = extract_shown_cost(" ".join(cells))
        rows.append({"model_name": model_name, "score": score, "shown_cost": shown_cost})

    if rows:
        return rows

    # Fallback for script-rendered pages containing JSON-ish data.
    json_like = re.findall(
        r'"model(?:_name)?"\s*:\s*"([^"]+)"[^\n\r]*?"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        html,
        flags=re.IGNORECASE,
    )
    for model_name, score_text in json_like:
        score = parse_score_from_text(score_text)
        if score is None:
            continue
        rows.append({"model_name": model_name, "score": score, "shown_cost": None})

    return rows


def extract_use_case_slugs_from_leaderboard_html(html: str) -> set[str]:
    """Extract `use_case` values from leaderboard page links or embedded JSON."""
    slugs: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select("a[href]"):
        href = tag.get("href")
        if not isinstance(href, str):
            continue
        parsed = urlparse(href)
        params = parse_qs(parsed.query)
        for value in params.get("use_case", []):
            cleaned = value.strip()
            if cleaned:
                slugs.add(cleaned)

    # Fallback for script-rendered pages.
    for slug in re.findall(r'"use_case"\s*:\s*"([^"]+)"', html, flags=re.IGNORECASE):
        cleaned = slug.strip()
        if cleaned:
            slugs.add(cleaned)

    return slugs


def parse_score_from_text(text: str) -> float | None:
    """Parse leaderboard score value from text."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_shown_cost(text: str) -> str | None:
    """Extract visible cost snippet from row text, if present."""
    match = re.search(r"(\$\s*[0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return match.group(1).replace(" ", "")
