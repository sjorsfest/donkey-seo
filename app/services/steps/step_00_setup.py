"""Step 0: Project Setup & Isolation.

Validates domain accessibility, detects language/locale, estimates site maturity,
and sets up project configuration.
"""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pipeline import StepExecution
from app.models.project import Project
from app.services.steps.base_step import BaseStepService, StepResult


@dataclass
class SetupInput:
    """Input for Step 0."""

    project_id: str


@dataclass
class SetupOutput:
    """Output from Step 0."""

    project_id: str
    domain: str
    final_url: str  # After redirects
    redirect_chain: list[str]  # Track redirect hops
    domain_accessible: bool
    robots_allowed: bool  # Can we crawl based on robots.txt?
    primary_language: str
    primary_locale: str
    site_maturity: str
    maturity_signals: dict[str, Any]
    sitemap_url: str | None
    sitemap_page_count: int
    compliance_suggestions: list[str]  # SUGGESTED flags (user can override)


class Step00SetupService(BaseStepService[SetupInput, SetupOutput]):
    """Step 0: Project Setup & Isolation.

    Validates and configures a new project:
    1. Verify domain is accessible
    2. Detect language from HTML
    3. Check for sitemap.xml
    4. Estimate site maturity
    5. Set up default configurations
    """

    step_number = 0
    step_name = "setup"
    is_optional = False

    async def _validate_preconditions(self, input_data: SetupInput) -> None:
        """No preconditions for Step 0."""
        pass

    async def _execute(self, input_data: SetupInput) -> SetupOutput:
        """Execute project setup validation."""
        # Load project
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()
        domain = project.domain

        await self._update_progress(10, "Validating domain accessibility...")

        # Ensure domain has scheme
        if not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"

        # Validate domain accessibility with redirect tracking
        access_result = await self._check_domain_accessible(domain)
        domain_accessible = access_result["accessible"]
        html_content = access_result["html"]
        final_url = access_result["final_url"]
        redirect_chain = access_result["redirect_chain"]

        await self._update_progress(30, "Detecting language and locale...")

        # Detect language
        primary_language, primary_locale = self._detect_language(html_content, domain)

        await self._update_progress(50, "Checking sitemap...")

        # Check sitemap
        sitemap_url, sitemap_page_count = await self._check_sitemap(final_url)

        await self._update_progress(70, "Checking robots.txt...")

        # Check robots.txt and determine if we're allowed to crawl
        robots_allowed = await self._check_robots_allowed(final_url)

        await self._update_progress(85, "Estimating site maturity...")

        # Estimate site maturity
        site_maturity, maturity_signals = self._estimate_maturity(
            sitemap_page_count,
            html_content,
        )

        # Infer compliance SUGGESTIONS (not hard flags - user can override)
        compliance_suggestions = self._infer_compliance_suggestions(html_content)

        await self._update_progress(100, "Setup complete")

        return SetupOutput(
            project_id=input_data.project_id,
            domain=domain,
            final_url=final_url,
            redirect_chain=redirect_chain,
            domain_accessible=domain_accessible,
            robots_allowed=robots_allowed,
            primary_language=primary_language,
            primary_locale=primary_locale,
            site_maturity=site_maturity,
            maturity_signals=maturity_signals,
            sitemap_url=sitemap_url,
            sitemap_page_count=sitemap_page_count,
            compliance_suggestions=compliance_suggestions,
        )

    async def _persist_results(self, result: SetupOutput) -> None:
        """Update project with setup results."""
        db_result = await self.session.execute(
            select(Project).where(Project.id == result.project_id)
        )
        project = db_result.scalar_one()

        # Update project fields
        project.primary_language = result.primary_language
        project.primary_locale = result.primary_locale
        project.site_maturity = result.site_maturity
        project.maturity_signals = {
            **result.maturity_signals,
            "final_url": result.final_url,
            "redirect_chain": result.redirect_chain,
            "robots_allowed": result.robots_allowed,
            "compliance_suggestions": result.compliance_suggestions,
        }
        project.current_step = 0
        project.status = "running"

        # Set result summary
        self.set_result_summary({
            "domain_accessible": result.domain_accessible,
            "final_url": result.final_url,
            "redirect_count": len(result.redirect_chain) - 1,
            "robots_allowed": result.robots_allowed,
            "primary_language": result.primary_language,
            "primary_locale": result.primary_locale,
            "site_maturity": result.site_maturity,
            "sitemap_page_count": result.sitemap_page_count,
            "compliance_suggestions": result.compliance_suggestions,
        })

        await self.session.commit()

    async def _check_domain_accessible(self, domain: str) -> dict[str, Any]:
        """Check if domain is accessible using GET (not HEAD - many sites block HEAD).

        Returns dict with:
        - accessible: bool
        - html: str
        - final_url: str (after redirects)
        - redirect_chain: list[str]
        """
        redirect_chain = [domain]
        try:
            # Use GET with small timeout and max bytes
            # follow_redirects=True but track the chain via history
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                max_redirects=10,
            ) as client:
                response = await client.get(domain)

                # Build redirect chain from response history
                for resp in response.history:
                    if str(resp.url) not in redirect_chain:
                        redirect_chain.append(str(resp.url))

                final_url = str(response.url)
                if final_url not in redirect_chain:
                    redirect_chain.append(final_url)

                # Only read first 50KB to avoid memory issues
                html = response.text[:50000] if response.text else ""

                return {
                    "accessible": response.status_code == 200,
                    "html": html,
                    "final_url": final_url,
                    "redirect_chain": redirect_chain,
                }
        except httpx.HTTPError:
            return {
                "accessible": False,
                "html": "",
                "final_url": domain,
                "redirect_chain": redirect_chain,
            }

    def _detect_language(self, html: str, domain: str) -> tuple[str, str]:
        """Detect language and locale from HTML content."""
        # Default values
        language = "en"
        locale = "en-US"

        if not html:
            return language, locale

        # Check <html lang="...">
        lang_match = re.search(r'<html[^>]*\slang=["\']([^"\']+)["\']', html, re.I)
        if lang_match:
            lang_value = lang_match.group(1)
            if "-" in lang_value:
                parts = lang_value.split("-")
                language = parts[0].lower()
                locale = f"{parts[0].lower()}-{parts[1].upper()}"
            else:
                language = lang_value.lower()
                locale = f"{language}-{language.upper()}"

        # Check hreflang tags
        hreflang_match = re.search(
            r'<link[^>]*hreflang=["\']([^"\']+)["\'][^>]*rel=["\']alternate["\']',
            html,
            re.I,
        )
        if hreflang_match:
            hreflang = hreflang_match.group(1)
            if hreflang != "x-default" and "-" in hreflang:
                parts = hreflang.split("-")
                language = parts[0].lower()
                locale = f"{parts[0].lower()}-{parts[1].upper()}"

        return language, locale

    async def _check_sitemap(self, domain: str) -> tuple[str | None, int]:
        """Check for sitemap.xml and count pages."""
        sitemap_urls = [
            f"{domain}/sitemap.xml",
            f"{domain}/sitemap_index.xml",
            f"{domain}/sitemap-index.xml",
        ]

        async with httpx.AsyncClient(timeout=30) as client:
            for url in sitemap_urls:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        page_count = self._count_sitemap_urls(response.text)
                        return url, page_count
                except httpx.HTTPError:
                    continue

        return None, 0

    def _count_sitemap_urls(self, sitemap_content: str) -> int:
        """Count URLs in sitemap XML."""
        try:
            root = ET.fromstring(sitemap_content)
            # Handle namespace
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            # Try with namespace
            urls = root.findall(".//sm:url", ns)
            if urls:
                return len(urls)

            # Try without namespace
            urls = root.findall(".//url")
            if urls:
                return len(urls)

            # Check for sitemap index
            sitemaps = root.findall(".//sm:sitemap", ns) or root.findall(".//sitemap")
            return len(sitemaps) * 100  # Estimate

        except ET.ParseError:
            return 0

    async def _check_robots_allowed(self, domain: str) -> bool:
        """Check if robots.txt allows crawling.

        Returns True if:
        - No robots.txt exists (default allow)
        - robots.txt exists and allows our user-agent
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"{domain}/robots.txt")
                if response.status_code != 200:
                    return True  # No robots.txt = allowed

                robots_content = response.text.lower()

                # Check for blanket disallow
                # This is a simple check - a full parser would be more accurate
                if "disallow: /" in robots_content:
                    # Check if there's a user-agent: * section with disallow: /
                    lines = robots_content.split("\n")
                    in_star_section = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith("user-agent:"):
                            in_star_section = "*" in line
                        elif in_star_section and line == "disallow: /":
                            return False  # Blanket disallow for all bots

                return True  # Allowed by default
        except httpx.HTTPError:
            return True  # Assume allowed if we can't fetch

    def _infer_compliance_suggestions(self, html: str) -> list[str]:
        """Infer compliance SUGGESTIONS from content.

        These are SUGGESTIONS only - user can override in project settings.
        Returns list of suggested compliance flags.
        """
        if not html:
            return []

        suggestions = []
        html_lower = html.lower()

        # Medical/health indicators
        medical_terms = [
            "medical", "health", "doctor", "patient", "diagnosis",
            "treatment", "symptom", "disease", "medication", "clinical",
            "hospital", "healthcare", "medicine"
        ]
        if any(term in html_lower for term in medical_terms):
            suggestions.append("medical_ymyl")

        # Financial indicators
        finance_terms = [
            "investment", "financial", "banking", "insurance", "mortgage",
            "loan", "credit", "tax", "retirement", "stock", "trading"
        ]
        if any(term in html_lower for term in finance_terms):
            suggestions.append("finance_ymyl")

        # Legal indicators
        legal_terms = [
            "legal", "attorney", "lawyer", "law firm", "litigation",
            "lawsuit", "court", "legal advice"
        ]
        if any(term in html_lower for term in legal_terms):
            suggestions.append("legal_ymyl")

        # E-commerce indicators
        ecommerce_terms = [
            "add to cart", "checkout", "buy now", "shopping cart",
            "payment", "shipping", "order"
        ]
        if any(term in html_lower for term in ecommerce_terms):
            suggestions.append("ecommerce")

        return suggestions

    def _estimate_maturity(
        self,
        sitemap_count: int,
        html: str,
    ) -> tuple[str, dict[str, Any]]:
        """Estimate site maturity based on signals.

        Returns:
            Tuple of (maturity_level, signals_dict)
            maturity_level: "new" | "mid" | "strong"
        """
        signals = {
            "sitemap_pages": sitemap_count,
            "has_blog": False,
            "has_pricing": False,
            "has_about": False,
        }

        # Check for common page indicators in navigation/links
        if html:
            html_lower = html.lower()
            signals["has_blog"] = "/blog" in html_lower or "blog" in html_lower
            signals["has_pricing"] = "/pricing" in html_lower
            signals["has_about"] = "/about" in html_lower

        # Calculate maturity score
        score = 0

        # Sitemap size scoring
        if sitemap_count > 500:
            score += 3
        elif sitemap_count > 100:
            score += 2
        elif sitemap_count > 20:
            score += 1

        # Page type scoring
        if signals["has_blog"]:
            score += 1
        if signals["has_pricing"]:
            score += 1
        if signals["has_about"]:
            score += 1

        # Determine maturity level
        if score >= 5:
            maturity = "strong"
        elif score >= 2:
            maturity = "mid"
        else:
            maturity = "new"

        signals["maturity_score"] = score

        return maturity, signals
