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

    _HEX_COLOR_PATTERN = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
    _FONT_FAMILY_PATTERN = re.compile(r"font-family\s*:\s*([^;}{]+)", re.IGNORECASE)

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

        # Extract metadata
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]

        visual_signals = self._extract_visual_signals(soup=soup)
        asset_candidates = self._extract_asset_candidates(soup=soup, page_url=url)

        # Remove non-content shells for text extraction after visual/asset extraction.
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

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

        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "headings": headings,
            "text_content": text_content[:50000],  # Limit content size
            "links": list(set(links)),
            "asset_candidates": asset_candidates,
            "visual_signals": visual_signals,
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
        visual_signal_pages: list[dict[str, list[str]]] = []
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

            page_visual_signals = page.get("visual_signals")
            if isinstance(page_visual_signals, dict):
                visual_signal_pages.append(
                    {
                        key: [
                            str(item).strip()
                            for item in values
                            if str(item).strip()
                        ]
                        for key, values in page_visual_signals.items()
                        if isinstance(values, list)
                    }
                )

            combined_parts.append("")

        homepage_visual_signals = (
            homepage.get("visual_signals")
            if isinstance(homepage.get("visual_signals"), dict)
            else {}
        )
        site_visual_signals = self._merge_visual_signals(visual_signal_pages)

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
            "homepage_visual_signals": homepage_visual_signals,
            "site_visual_signals": site_visual_signals,
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

    @classmethod
    def _extract_visual_signals(cls, *, soup: BeautifulSoup) -> dict[str, list[str]]:
        style_chunks: list[str] = []
        for style_tag in soup.find_all("style"):
            text = style_tag.get_text(" ", strip=True)
            if text:
                style_chunks.append(text)

        for node in soup.find_all(attrs={"style": True}):
            inline_style = str(node.get("style") or "").strip()
            if inline_style:
                style_chunks.append(inline_style)

        style_blob = "\n".join(style_chunks)

        class_tokens: list[str] = []
        for node in soup.find_all(class_=True):
            for token in node.get("class") or []:
                normalized = str(token).strip().lower()
                if normalized:
                    class_tokens.append(normalized)

        class_blob = " ".join(class_tokens[:1500])
        combined_style_blob = f"{style_blob}\n{class_blob}"

        shape_cues: list[str] = []
        surface_cues: list[str] = []
        if re.search(r"border-radius|rounded|pill|capsule|radius", combined_style_blob, re.IGNORECASE):
            shape_cues.append("Rounded corners and pill-like controls")
        if re.search(r"\bborder\b|\boutline\b|\bstroke\b|\bring-\d", combined_style_blob, re.IGNORECASE):
            shape_cues.append("Outlined or stroked UI elements")
        if re.search(r"box-shadow|\bshadow\b|drop-shadow", combined_style_blob, re.IGNORECASE):
            surface_cues.append("Drop-shadows used for depth")
        if re.search(r"gradient\(", combined_style_blob, re.IGNORECASE):
            surface_cues.append("Gradient surface treatments")
        if re.search(r"texture|grain|noise", combined_style_blob, re.IGNORECASE):
            surface_cues.append("Textured or grain overlays")
        if re.search(r"backdrop-filter|blur", combined_style_blob, re.IGNORECASE):
            surface_cues.append("Blurred/translucent UI layers")

        cta_labels: list[str] = []
        cta_keywords = (
            "start",
            "get",
            "book",
            "try",
            "join",
            "demo",
            "contact",
            "free",
            "pricing",
            "works",
            "learn",
            "sign",
        )
        for node in soup.find_all(["a", "button"]):
            label = " ".join(node.get_text(" ", strip=True).split())
            if not label or len(label) > 44:
                continue
            if any(keyword in label.lower() for keyword in cta_keywords):
                cta_labels.append(label)

        hero_headlines: list[str] = []
        for heading in soup.find_all(["h1", "h2"]):
            text = " ".join(heading.get_text(" ", strip=True).split())
            if text:
                hero_headlines.append(text)

        imagery_blob_parts: list[str] = []
        for image in soup.find_all("img"):
            imagery_blob_parts.extend(
                [
                    str(image.get("alt") or ""),
                    str(image.get("src") or ""),
                    " ".join(image.get("class") or []),
                ]
            )
        imagery_blob = " ".join(imagery_blob_parts).lower()
        imagery_cues: list[str] = []
        imagery_patterns = [
            (r"illustration|vector|drawn|sticker|cartoon|mascot|avatar|emoji", "Illustrative/mascot imagery"),
            (r"screenshot|dashboard|ui|widget|mockup|interface|chat", "Product UI/screenshot context"),
            (r"photo|photography|portrait", "Photography-led visuals"),
        ]
        for pattern, cue in imagery_patterns:
            if re.search(pattern, imagery_blob):
                imagery_cues.append(cue)

        color_word_hints: list[str] = []
        color_keywords = (
            "pink",
            "rose",
            "red",
            "orange",
            "amber",
            "yellow",
            "lime",
            "green",
            "emerald",
            "teal",
            "cyan",
            "blue",
            "indigo",
            "violet",
            "purple",
            "fuchsia",
            "slate",
            "gray",
            "neutral",
            "black",
            "white",
            "cream",
            "beige",
        )
        for token in class_tokens:
            for color_keyword in color_keywords:
                if color_keyword in token:
                    color_word_hints.append(color_keyword)
                    break

        return {
            "observed_hex_colors": cls._extract_hex_colors(style_blob),
            "observed_font_families": cls._extract_font_families(style_blob),
            "shape_cues": cls._dedupe_and_limit(shape_cues, limit=6),
            "surface_cues": cls._dedupe_and_limit(surface_cues, limit=6),
            "cta_labels": cls._dedupe_and_limit(cta_labels, limit=8),
            "hero_headlines": cls._dedupe_and_limit(hero_headlines, limit=6),
            "imagery_cues": cls._dedupe_and_limit(imagery_cues, limit=6),
            "color_word_hints": cls._dedupe_and_limit(color_word_hints, limit=8),
        }

    @classmethod
    def _merge_visual_signals(cls, page_signals: list[dict[str, list[str]]]) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {
            "observed_hex_colors": [],
            "observed_font_families": [],
            "shape_cues": [],
            "surface_cues": [],
            "cta_labels": [],
            "hero_headlines": [],
            "imagery_cues": [],
            "color_word_hints": [],
        }
        limits = {
            "observed_hex_colors": 12,
            "observed_font_families": 8,
            "shape_cues": 8,
            "surface_cues": 8,
            "cta_labels": 12,
            "hero_headlines": 10,
            "imagery_cues": 8,
            "color_word_hints": 10,
        }

        for signals in page_signals:
            for key, values in signals.items():
                if key not in merged or not isinstance(values, list):
                    continue
                merged[key].extend(str(value).strip() for value in values if str(value).strip())

        for key, values in merged.items():
            merged[key] = cls._dedupe_and_limit(values, limit=limits[key])
        return merged

    @classmethod
    def _extract_hex_colors(cls, style_blob: str) -> list[str]:
        colors = []
        for match in cls._HEX_COLOR_PATTERN.findall(style_blob):
            normalized = cls._normalize_hex_color(match)
            if normalized:
                colors.append(normalized)
        return cls._dedupe_and_limit(colors, limit=12)

    @classmethod
    def _extract_font_families(cls, style_blob: str) -> list[str]:
        fonts: list[str] = []
        generic_families = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"}

        for raw_group in cls._FONT_FAMILY_PATTERN.findall(style_blob):
            parts = [part.strip(" '\"") for part in raw_group.split(",")]
            for part in parts:
                normalized = " ".join(part.split())
                if not normalized:
                    continue
                if normalized.casefold() in generic_families:
                    continue
                fonts.append(normalized)

        return cls._dedupe_and_limit(fonts, limit=6)

    @staticmethod
    def _normalize_hex_color(raw: str) -> str | None:
        value = str(raw).strip()
        if not value.startswith("#"):
            return None
        hex_value = value[1:]
        if len(hex_value) == 3 and all(char in "0123456789abcdefABCDEF" for char in hex_value):
            expanded = "".join(char * 2 for char in hex_value)
            return f"#{expanded.upper()}"
        if len(hex_value) == 6 and all(char in "0123456789abcdefABCDEF" for char in hex_value):
            return f"#{hex_value.upper()}"
        return None

    @staticmethod
    def _dedupe_and_limit(values: list[str], *, limit: int) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(str(value).split())
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
            if len(unique) >= limit:
                break
        return unique


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
