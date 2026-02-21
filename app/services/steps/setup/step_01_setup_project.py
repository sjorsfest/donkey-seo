"""Step 1: Project setup and isolation checks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
import xml.etree.ElementTree as ET

import httpx
from sqlalchemy import select

from app.models.project import Project
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class SetupProjectInput:
    """Input for setup step 1."""

    project_id: str


@dataclass
class SetupProjectOutput:
    """Output for setup step 1."""

    project_id: str
    domain: str
    final_url: str
    redirect_chain: list[str]
    domain_accessible: bool
    robots_allowed: bool
    primary_language: str
    primary_locale: str
    site_maturity: str
    maturity_signals: dict[str, Any]
    sitemap_url: str | None
    sitemap_page_count: int
    compliance_suggestions: list[str]


class Step01SetupProjectService(BaseStepService[SetupProjectInput, SetupProjectOutput]):
    """Step 1: Validate and initialize project setup metadata."""

    step_number = 1
    step_name = "setup_project"
    is_optional = False

    async def _validate_preconditions(self, input_data: SetupProjectInput) -> None:
        """No preconditions for setup step 1."""
        return None

    async def _execute(self, input_data: SetupProjectInput) -> SetupProjectOutput:
        """Execute project setup validation."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()
        domain = project.domain

        await self._update_progress(10, "Validating domain accessibility...")

        if not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"

        access_result = await self._check_domain_accessible(domain)
        domain_accessible = access_result["accessible"]
        html_content = access_result["html"]
        final_url = access_result["final_url"]
        redirect_chain = access_result["redirect_chain"]
        logger.info(
            "Domain check complete",
            extra={
                "domain": domain,
                "accessible": domain_accessible,
                "final_url": final_url,
                "redirect_count": len(redirect_chain) - 1,
            },
        )

        await self._update_progress(30, "Detecting language and locale...")
        primary_language, primary_locale = self._detect_language(html_content, domain)

        await self._update_progress(50, "Checking sitemap...")
        sitemap_url, sitemap_page_count = await self._check_sitemap(final_url)

        await self._update_progress(70, "Checking robots.txt...")
        robots_allowed = await self._check_robots_allowed(final_url)

        await self._update_progress(85, "Estimating site maturity...")
        site_maturity, maturity_signals = self._estimate_maturity(
            sitemap_page_count,
            html_content,
        )
        compliance_suggestions = self._infer_compliance_suggestions(html_content)

        await self._update_progress(100, "Setup checks complete")

        return SetupProjectOutput(
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

    async def _persist_results(self, result: SetupProjectOutput) -> None:
        """Update project with setup step results."""
        db_result = await self.session.execute(
            select(Project).where(Project.id == result.project_id)
        )
        project = db_result.scalar_one()

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
        project.current_step = max(project.current_step, self.step_number)
        project.status = "running"

        self.set_result_summary(
            {
                "domain_accessible": result.domain_accessible,
                "final_url": result.final_url,
                "redirect_count": len(result.redirect_chain) - 1,
                "robots_allowed": result.robots_allowed,
                "primary_language": result.primary_language,
                "primary_locale": result.primary_locale,
                "site_maturity": result.site_maturity,
                "sitemap_page_count": result.sitemap_page_count,
                "compliance_suggestions": result.compliance_suggestions,
            }
        )

        await self.session.commit()

    async def _check_domain_accessible(self, domain: str) -> dict[str, Any]:
        """Check if domain is accessible and track redirects."""
        redirect_chain = [domain]
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                max_redirects=10,
            ) as client:
                response = await client.get(domain)

                for resp in response.history:
                    if str(resp.url) not in redirect_chain:
                        redirect_chain.append(str(resp.url))

                final_url = str(response.url)
                if final_url not in redirect_chain:
                    redirect_chain.append(final_url)

                html = response.text[:50000] if response.text else ""

                return {
                    "accessible": response.status_code == 200,
                    "html": html,
                    "final_url": final_url,
                    "redirect_chain": redirect_chain,
                }
        except httpx.HTTPError:
            logger.warning("Domain not accessible", extra={"domain": domain})
            return {
                "accessible": False,
                "html": "",
                "final_url": domain,
                "redirect_chain": redirect_chain,
            }

    def _detect_language(self, html: str, domain: str) -> tuple[str, str]:
        """Detect language and locale from HTML content."""
        _ = domain
        language = "en"
        locale = "en-US"

        if not html:
            return language, locale

        lang_match = re.search(r'<html[^>]*\\slang=["\']([^"\']+)["\']', html, re.I)
        if lang_match:
            lang_value = lang_match.group(1)
            if "-" in lang_value:
                parts = lang_value.split("-")
                language = parts[0].lower()
                locale = f"{parts[0].lower()}-{parts[1].upper()}"
            else:
                language = lang_value.lower()
                locale = f"{language}-{language.upper()}"

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
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            urls = root.findall(".//sm:url", ns)
            if urls:
                return len(urls)

            urls = root.findall(".//url")
            if urls:
                return len(urls)

            sitemaps = root.findall(".//sm:sitemap", ns) or root.findall(".//sitemap")
            return len(sitemaps) * 100

        except ET.ParseError:
            return 0

    async def _check_robots_allowed(self, domain: str) -> bool:
        """Check whether robots.txt permits crawling."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"{domain}/robots.txt")
                if response.status_code != 200:
                    return True

                robots_content = response.text.lower()
                if "disallow: /" in robots_content:
                    lines = robots_content.split("\n")
                    in_star_section = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith("user-agent:"):
                            in_star_section = "*" in line
                        elif in_star_section and line == "disallow: /":
                            return False

                return True
        except httpx.HTTPError:
            return True

    def _infer_compliance_suggestions(self, html: str) -> list[str]:
        """Infer compliance suggestion flags from page content."""
        if not html:
            return []

        suggestions = []
        html_lower = html.lower()

        medical_terms = [
            "medical",
            "health",
            "doctor",
            "patient",
            "diagnosis",
            "treatment",
            "symptom",
            "disease",
            "medication",
            "clinical",
            "hospital",
            "healthcare",
            "medicine",
        ]
        if any(term in html_lower for term in medical_terms):
            suggestions.append("medical_ymyl")

        finance_terms = [
            "investment",
            "financial",
            "banking",
            "insurance",
            "mortgage",
            "loan",
            "credit",
            "tax",
            "retirement",
            "stock",
            "trading",
        ]
        if any(term in html_lower for term in finance_terms):
            suggestions.append("finance_ymyl")

        legal_terms = [
            "legal",
            "attorney",
            "lawyer",
            "law firm",
            "litigation",
            "lawsuit",
            "court",
            "legal advice",
        ]
        if any(term in html_lower for term in legal_terms):
            suggestions.append("legal_ymyl")

        ecommerce_terms = [
            "add to cart",
            "checkout",
            "buy now",
            "shopping cart",
            "payment",
            "shipping",
            "order",
        ]
        if any(term in html_lower for term in ecommerce_terms):
            suggestions.append("ecommerce")

        return suggestions

    def _estimate_maturity(
        self,
        sitemap_count: int,
        html: str,
    ) -> tuple[str, dict[str, Any]]:
        """Estimate site maturity based on simple site-wide signals."""
        signals = {
            "sitemap_pages": sitemap_count,
            "has_blog": False,
            "has_pricing": False,
            "has_about": False,
        }

        if html:
            html_lower = html.lower()
            signals["has_blog"] = "/blog" in html_lower or "blog" in html_lower
            signals["has_pricing"] = "/pricing" in html_lower
            signals["has_about"] = "/about" in html_lower

        score = 0
        if sitemap_count > 500:
            score += 3
        elif sitemap_count > 100:
            score += 2
        elif sitemap_count > 20:
            score += 1

        if signals["has_blog"]:
            score += 1
        if signals["has_pricing"]:
            score += 1
        if signals["has_about"]:
            score += 1

        if score >= 5:
            maturity = "strong"
        elif score >= 2:
            maturity = "mid"
        else:
            maturity = "new"

        signals["maturity_score"] = score
        return maturity, signals
