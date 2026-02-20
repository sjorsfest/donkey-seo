"""Website scraper for extracting content from web pages."""

import asyncio
import logging
import re
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebsiteScraper:
    """Scraper for extracting content from websites."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_pages: int = 10,
        user_agent: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_pages = max_pages
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; DonkeySEO/1.0; +https://donkeyseo.com)"
        )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "WebsiteScraper":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Scraper must be used as async context manager")
        return self._client

    async def scrape_page(self, url: str) -> dict:
        """Scrape a single page and extract content.

        Returns:
            Dict with url, title, meta_description, headings, text_content, links
        """
        logger.info("Scraping page", extra={"url": url})
        try:
            response = await self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to scrape page", extra={"url": url, "error": str(e)})
            return {"url": url, "error": str(e)}

        soup = BeautifulSoup(response.text, "lxml")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract metadata
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]

        # Extract headings
        headings = []
        for level in range(1, 4):
            for h in soup.find_all(f"h{level}"):
                text = h.get_text(strip=True)
                if text:
                    headings.append({"level": level, "text": text})

        # Extract main content
        main_content = soup.find("main") or soup.find("article") or soup.body
        text_content = ""
        if main_content:
            text_content = main_content.get_text(separator="\n", strip=True)

        # Extract internal links
        base_domain = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not isinstance(href, str):
                continue
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain:
                links.append(full_url)

        asset_candidates = self._extract_asset_candidates(soup=soup, page_url=url)

        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "headings": headings,
            "text_content": text_content[:50000],  # Limit content size
            "links": list(set(links)),
            "asset_candidates": asset_candidates,
        }

    async def scrape_website(self, domain: str) -> dict:
        """Scrape key pages from a website.

        Crawls homepage and discovers important pages like
        about, pricing, features, contact, etc.

        Returns:
            Dict with domain, pages[], combined_content, source_urls[]
        """
        logger.info("Scraping website", extra={"domain": domain, "max_pages": self.max_pages})
        # Ensure domain has scheme
        if not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"

        # Key page patterns to look for
        key_patterns = [
            "/about",
            "/pricing",
            "/features",
            "/product",
            "/services",
            "/solutions",
            "/contact",
            "/faq",
            "/how-it-works",
            "/why-",
            "/use-cases",
        ]

        # Start with homepage
        homepage = await self.scrape_page(domain)
        if "error" in homepage:
            logger.warning(
                "Homepage scrape failed",
                extra={"domain": domain, "error": homepage["error"]},
            )
            return {"domain": domain, "error": homepage["error"], "pages": []}

        pages = [homepage]
        crawled_urls = {domain}

        # Find and crawl key pages
        candidate_urls = []
        for link in homepage.get("links", []):
            link_lower = link.lower()
            for pattern in key_patterns:
                if pattern in link_lower and link not in crawled_urls:
                    candidate_urls.append(link)
                    break

        # Limit to max_pages
        candidate_urls = candidate_urls[: self.max_pages - 1]

        # Crawl key pages concurrently
        if candidate_urls:
            tasks = [self.scrape_page(url) for url in candidate_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict) and "error" not in result:
                    pages.append(result)
                    crawled_urls.add(result["url"])

        # Combine content for LLM processing
        combined_parts = []
        aggregated_asset_candidates: dict[str, dict] = {}
        for page in pages:
            if "error" in page:
                continue
            combined_parts.append(f"## Page: {page['url']}")
            combined_parts.append(f"Title: {page.get('title', 'N/A')}")
            combined_parts.append(f"Description: {page.get('meta_description', 'N/A')}")

            headings = page.get("headings", [])
            if headings:
                combined_parts.append("Headings:")
                for h in headings[:20]:  # Limit headings
                    combined_parts.append(f"  {'#' * h['level']} {h['text']}")

            text = page.get("text_content", "")
            if text:
                combined_parts.append("Content:")
                combined_parts.append(text[:5000])  # Limit per page

            for candidate in page.get("asset_candidates", []):
                if not isinstance(candidate, dict):
                    continue
                candidate_url = str(candidate.get("url") or "").strip()
                if not candidate_url:
                    continue

                existing = aggregated_asset_candidates.get(candidate_url)
                if not existing:
                    aggregated_asset_candidates[candidate_url] = candidate
                    continue

                existing_confidence = float(existing.get("role_confidence") or 0.0)
                candidate_confidence = float(candidate.get("role_confidence") or 0.0)
                if candidate_confidence > existing_confidence:
                    aggregated_asset_candidates[candidate_url] = candidate

            combined_parts.append("")

        logger.info(
            "Website scrape complete",
            extra={
                "domain": domain,
                "pages_scraped": len(pages),
                "urls_found": len(crawled_urls),
            },
        )
        return {
            "domain": domain,
            "pages": pages,
            "combined_content": "\n".join(combined_parts),
            "source_urls": list(crawled_urls),
            "asset_candidates": sorted(
                aggregated_asset_candidates.values(),
                key=lambda item: float(item.get("role_confidence") or 0.0),
                reverse=True,
            ),
        }

    def _extract_asset_candidates(self, *, soup: BeautifulSoup, page_url: str) -> list[dict]:
        candidates: dict[str, dict] = {}

        def _record_candidate(
            url: str | None,
            *,
            role: str,
            role_confidence: float,
            origin: str,
        ) -> None:
            if not url:
                return
            full_url = urljoin(page_url, url)
            parsed = urlparse(full_url)
            if parsed.scheme not in {"http", "https"}:
                return

            existing = candidates.get(full_url)
            payload = {
                "url": full_url,
                "role": role,
                "role_confidence": role_confidence,
                "origin": origin,
                "source_page": page_url,
            }
            if not existing:
                candidates[full_url] = payload
                return

            if role_confidence > float(existing.get("role_confidence") or 0.0):
                candidates[full_url] = payload

        for meta in soup.find_all("meta"):
            property_name = str(meta.get("property") or "").strip().lower()
            name_attr = str(meta.get("name") or "").strip().lower()
            content = str(meta.get("content") or "").strip()
            if not content:
                continue
            if property_name == "og:image":
                _record_candidate(
                    content,
                    role="hero",
                    role_confidence=0.9,
                    origin="meta_og_image",
                )
            elif name_attr == "twitter:image":
                _record_candidate(
                    content,
                    role="hero",
                    role_confidence=0.85,
                    origin="meta_twitter_image",
                )

        for link in soup.find_all("link"):
            rel_values = link.get("rel") or []
            rel_text = " ".join(str(rel).lower() for rel in rel_values)
            href = str(link.get("href") or "").strip()
            if not href:
                continue
            if "icon" in rel_text:
                _record_candidate(
                    href,
                    role="icon",
                    role_confidence=0.88,
                    origin="link_icon",
                )

        logo_signal_pattern = re.compile(r"(logo|brand|wordmark|logotype)")
        for image in soup.find_all("img"):
            src = str(image.get("src") or "").strip()
            if not src:
                continue

            attrs = " ".join(
                [
                    str(image.get("alt") or ""),
                    str(image.get("id") or ""),
                    " ".join(image.get("class") or []),
                    src,
                ]
            ).lower()

            if logo_signal_pattern.search(attrs):
                _record_candidate(
                    src,
                    role="logo",
                    role_confidence=0.95,
                    origin="img_logo_signal",
                )
                continue

            if "hero" in attrs:
                _record_candidate(
                    src,
                    role="hero",
                    role_confidence=0.6,
                    origin="img_hero_signal",
                )

        return list(candidates.values())


async def scrape_website(domain: str, max_pages: int = 10) -> dict:
    """Convenience function to scrape a website.

    Args:
        domain: Domain to scrape (with or without https://)
        max_pages: Maximum number of pages to crawl

    Returns:
        Scraped content dict
    """
    async with WebsiteScraper(max_pages=max_pages) as scraper:
        return await scraper.scrape_website(domain)
