"""Sitemap fetcher for extracting URLs from XML sitemaps."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from lxml import etree

logger = logging.getLogger(__name__)


@dataclass
class SitemapPage:
    """Represents a single page from a sitemap."""

    url: str
    lastmod: datetime | None = None
    priority: float | None = None
    changefreq: str | None = None

    @property
    def sort_score(self) -> float:
        """Calculate a score for sorting pages by importance.

        Higher scores indicate more important pages.
        Uses priority (if available) and recency.
        """
        score = 0.0

        # Priority contributes up to 0.5
        if self.priority is not None:
            score += self.priority * 0.5
        else:
            score += 0.25  # Default middle value

        # Recency contributes up to 0.5
        if self.lastmod:
            # Pages modified in last 30 days get higher scores
            now = datetime.now(self.lastmod.tzinfo or None)
            days_old = (now - self.lastmod).days
            recency_score = max(0, 1 - (days_old / 365))  # Decay over a year
            score += recency_score * 0.5

        return score


class SitemapFetcher:
    """Fetcher for parsing XML sitemaps from websites."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_urls: int = 1000,
        max_sitemaps: int = 10,
        user_agent: str | None = None,
    ) -> None:
        """Initialize sitemap fetcher.

        Args:
            timeout: Request timeout in seconds
            max_urls: Maximum number of URLs to return
            max_sitemaps: Maximum number of sitemap files to parse (for sitemap indexes)
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.max_urls = max_urls
        self.max_sitemaps = max_sitemaps
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; DonkeySEO/1.0; +https://donkeyseo.com)"
        )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SitemapFetcher":
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
            raise RuntimeError("SitemapFetcher must be used as async context manager")
        return self._client

    async def fetch_sitemap(self, domain: str) -> list[SitemapPage]:
        """Fetch and parse sitemap from a domain.

        Tries multiple common sitemap locations:
        1. https://domain.com/sitemap.xml
        2. https://domain.com/sitemap_index.xml
        3. Parse robots.txt for Sitemap: directive

        Args:
            domain: Domain to fetch sitemap from (e.g., "example.com")

        Returns:
            List of SitemapPage objects, sorted by importance and limited to max_urls
        """
        # Ensure domain has protocol
        if not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"

        # Normalize domain (remove trailing slash)
        domain = domain.rstrip("/")

        logger.info("Fetching sitemap", extra={"domain": domain})

        # Try common sitemap locations
        sitemap_urls = [
            f"{domain}/sitemap.xml",
            f"{domain}/sitemap_index.xml",
        ]

        pages = []
        for sitemap_url in sitemap_urls:
            try:
                pages = await self._fetch_and_parse_sitemap(sitemap_url)
                if pages:
                    logger.info(
                        "Successfully fetched sitemap",
                        extra={"url": sitemap_url, "pages": len(pages)},
                    )
                    break
            except Exception as e:
                logger.debug(
                    "Failed to fetch sitemap at standard location",
                    extra={"url": sitemap_url, "error": str(e)},
                )

        # If no sitemap found, try robots.txt
        if not pages:
            try:
                robots_sitemaps = await self._fetch_sitemaps_from_robots(domain)
                for sitemap_url in robots_sitemaps[:self.max_sitemaps]:
                    try:
                        pages = await self._fetch_and_parse_sitemap(sitemap_url)
                        if pages:
                            logger.info(
                                "Successfully fetched sitemap from robots.txt",
                                extra={"url": sitemap_url, "pages": len(pages)},
                            )
                            break
                    except Exception as e:
                        logger.debug(
                            "Failed to fetch sitemap from robots.txt",
                            extra={"url": sitemap_url, "error": str(e)},
                        )
            except Exception as e:
                logger.debug(
                    "Failed to parse robots.txt",
                    extra={"domain": domain, "error": str(e)},
                )

        if not pages:
            logger.warning("No sitemap found", extra={"domain": domain})
            return []

        # Sort by importance and limit
        pages.sort(key=lambda p: p.sort_score, reverse=True)
        limited_pages = pages[:self.max_urls]

        logger.info(
            "Sitemap fetch complete",
            extra={
                "domain": domain,
                "total_pages": len(pages),
                "returned_pages": len(limited_pages),
            },
        )

        return limited_pages

    async def _fetch_sitemaps_from_robots(self, domain: str) -> list[str]:
        """Parse robots.txt to find sitemap URLs.

        Args:
            domain: Domain to fetch robots.txt from

        Returns:
            List of sitemap URLs found in robots.txt
        """
        robots_url = f"{domain}/robots.txt"

        try:
            response = await self.client.get(robots_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug(
                "Failed to fetch robots.txt",
                extra={"url": robots_url, "error": str(e)},
            )
            return []

        sitemap_urls = []
        for line in response.text.split("\n"):
            line = line.strip()
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                sitemap_urls.append(sitemap_url)

        return sitemap_urls

    async def _fetch_and_parse_sitemap(self, sitemap_url: str) -> list[SitemapPage]:
        """Fetch and parse a single sitemap XML file.

        Handles both regular sitemaps and sitemap indexes.

        Args:
            sitemap_url: URL of the sitemap to fetch

        Returns:
            List of SitemapPage objects
        """
        try:
            response = await self.client.get(sitemap_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug(
                "Failed to fetch sitemap",
                extra={"url": sitemap_url, "error": str(e)},
            )
            raise

        try:
            root = etree.fromstring(response.content)
        except etree.XMLSyntaxError as e:
            logger.warning(
                "Invalid XML in sitemap",
                extra={"url": sitemap_url, "error": str(e)},
            )
            raise

        # Determine if this is a sitemap index or regular sitemap
        # Check for sitemap elements (sitemap index)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        sitemap_elements = root.xpath("//ns:sitemap/ns:loc", namespaces=namespace)

        if sitemap_elements:
            # This is a sitemap index - recursively fetch child sitemaps
            return await self._parse_sitemap_index(root, namespace)
        else:
            # This is a regular sitemap - parse URL elements
            return self._parse_sitemap(root, namespace)

    async def _parse_sitemap_index(
        self,
        root: etree._Element,
        namespace: dict[str, str],
    ) -> list[SitemapPage]:
        """Parse a sitemap index and fetch all child sitemaps.

        Args:
            root: XML root element
            namespace: XML namespace mapping

        Returns:
            Combined list of SitemapPage objects from all child sitemaps
        """
        sitemap_locs = root.xpath("//ns:sitemap/ns:loc", namespaces=namespace)
        child_sitemap_urls = [loc.text for loc in sitemap_locs if loc.text]

        # Limit number of child sitemaps to fetch
        child_sitemap_urls = child_sitemap_urls[:self.max_sitemaps]

        logger.info(
            "Found sitemap index",
            extra={
                "total_sitemaps": len(sitemap_locs),
                "fetching": len(child_sitemap_urls),
            },
        )

        # Fetch all child sitemaps concurrently
        tasks = [
            self._fetch_and_parse_sitemap(url)
            for url in child_sitemap_urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results, filtering out errors
        all_pages = []
        for result in results:
            if isinstance(result, list):
                all_pages.extend(result)
            else:
                logger.debug(
                    "Failed to fetch child sitemap",
                    extra={"error": str(result)},
                )

        return all_pages

    def _parse_sitemap(
        self,
        root: etree._Element,
        namespace: dict[str, str],
    ) -> list[SitemapPage]:
        """Parse a regular sitemap XML and extract URLs.

        Args:
            root: XML root element
            namespace: XML namespace mapping

        Returns:
            List of SitemapPage objects
        """
        pages = []

        url_elements = root.xpath("//ns:url", namespaces=namespace)

        for url_elem in url_elements:
            # Extract loc (required)
            loc_elem = url_elem.find("ns:loc", namespaces=namespace)
            if loc_elem is None or not loc_elem.text:
                continue

            url = loc_elem.text.strip()

            # Extract lastmod (optional)
            lastmod = None
            lastmod_elem = url_elem.find("ns:lastmod", namespaces=namespace)
            if lastmod_elem is not None and lastmod_elem.text:
                try:
                    lastmod = self._parse_datetime(lastmod_elem.text.strip())
                except Exception as e:
                    logger.debug(
                        "Failed to parse lastmod",
                        extra={"url": url, "lastmod": lastmod_elem.text, "error": str(e)},
                    )

            # Extract priority (optional)
            priority = None
            priority_elem = url_elem.find("ns:priority", namespaces=namespace)
            if priority_elem is not None and priority_elem.text:
                try:
                    priority = float(priority_elem.text.strip())
                except (ValueError, TypeError):
                    pass

            # Extract changefreq (optional)
            changefreq = None
            changefreq_elem = url_elem.find("ns:changefreq", namespaces=namespace)
            if changefreq_elem is not None and changefreq_elem.text:
                changefreq = changefreq_elem.text.strip()

            pages.append(
                SitemapPage(
                    url=url,
                    lastmod=lastmod,
                    priority=priority,
                    changefreq=changefreq,
                )
            )

        return pages

    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime from sitemap lastmod field.

        Supports multiple formats:
        - 2024-02-21
        - 2024-02-21T10:30:00Z
        - 2024-02-21T10:30:00+00:00

        Args:
            date_str: Date string to parse

        Returns:
            datetime object
        """
        # Try common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",  # 2024-02-21T10:30:00+00:00
            "%Y-%m-%dT%H:%M:%SZ",   # 2024-02-21T10:30:00Z
            "%Y-%m-%d",             # 2024-02-21
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If none of the formats work, try fromisoformat
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Unable to parse datetime: {date_str}")
