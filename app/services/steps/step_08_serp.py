"""Step 8: SERP validation.

Validates Step 5 intent/page-type assumptions against live SERP evidence.
Uses deterministic heuristics only (no LLM calls).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.core.exceptions import APIKeyMissingError
from app.integrations.dataforseo import DataForSEOClient, get_location_code
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class SerpValidationInput:
    """Input for Step 8."""

    project_id: str


@dataclass
class SerpValidationOutput:
    """Output from Step 8."""

    keywords_candidate: int
    keywords_validated: int
    keywords_mismatched: int
    keywords_failed: int
    api_calls_made: int
    warnings: list[str] = field(default_factory=list)


class Step08SerpValidationService(
    BaseStepService[SerpValidationInput, SerpValidationOutput]
):
    """Step 8: Validate keyword intent/page type with live SERP signals."""

    step_number = 8
    step_name = "serp_validation"
    is_optional = True

    LOW_COHERENCE_THRESHOLD = 0.7
    SERP_BATCH_SIZE = 25
    SERP_DEPTH = 10
    SERP_DEVICE = "desktop"
    MAX_SAMPLE_PER_TOPIC = 3  # Max keywords to validate per flagged topic
    MAX_API_CALLS = 100  # Hard cap on total SERP API calls per run

    INTENT_PATTERNS: dict[str, list[str]] = {
        "transactional": [
            r"\b(pricing|price|cost|buy|purchase|order|trial|signup|sign up)\b",
            r"/(pricing|plans|checkout|buy|order)",
        ],
        "commercial": [
            r"\b(best|top|review|reviews|compare|comparison|vs|versus|alternatives?)\b",
            r"/(compare|comparison|reviews?)",
        ],
        "informational": [
            r"\b(how to|what is|what are|guide|tutorial|learn|examples?)\b",
            r"/(blog|guide|learn|docs|wiki|faq)",
        ],
        "navigational": [
            r"\b(login|sign in|dashboard|official site)\b",
            r"/(login|signin|dashboard)",
        ],
    }

    PAGE_TYPE_PATTERNS: dict[str, list[str]] = {
        "alternatives": [r"\balternatives?\b", r"/alternatives?"],
        "comparison": [r"\b(vs|versus|compare|comparison)\b", r"/(vs|compare|comparison)"],
        "landing": [
            r"\b(pricing|plans|buy|purchase|trial|demo|sign up)\b",
            r"/(pricing|plans|product|software|services?)",
        ],
        "guide": [r"\b(how to|guide|tutorial|step by step)\b", r"/(guide|tutorial|how-to)"],
        "glossary": [r"\b(what is|definition|meaning)\b", r"/(glossary|definition)"],
        "tool": [
            r"\b(tool|calculator|generator|template)\b",
            r"/(tool|calculator|generator|template)",
        ],
        "list": [r"\b(best|top\s+\d+|list of)\b"],
    }

    async def _validate_preconditions(self, input_data: SerpValidationInput) -> None:
        """Validate Step 7 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")
        if project.current_step < 7:
            raise ValueError("Step 7 (Prioritization) must be completed first")

    async def _execute(self, input_data: SerpValidationInput) -> SerpValidationOutput:
        """Execute SERP validation against selected keywords."""
        await self._update_progress(5, "Loading prioritized topics...")

        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

        topics_result = await self.session.execute(
            select(Topic).where(
                Topic.project_id == input_data.project_id,
                Topic.priority_rank.isnot(None),
            )
        )
        prioritized_topics = list(topics_result.scalars())
        if not prioritized_topics:
            return SerpValidationOutput(
                keywords_candidate=0,
                keywords_validated=0,
                keywords_mismatched=0,
                keywords_failed=0,
                api_calls_made=0,
                warnings=[],
            )

        # Only validate topics that actually need it (low coherence or flagged)
        flagged_topics = [
            topic for topic in prioritized_topics
            if self._topic_needs_serp_validation(topic)
        ]
        logger.info(
            "SERP validation scope",
            extra={
                "prioritized_topics": len(prioritized_topics),
                "flagged_topics": len(flagged_topics),
            },
        )

        # Always include primary keywords for all prioritized topics so discovery
        # gating has deterministic SERP evidence per candidate topic.
        prioritized_primary_ids = {
            topic.primary_keyword_id
            for topic in prioritized_topics
            if topic.primary_keyword_id
        }
        flagged_topic_ids = {str(topic.id) for topic in flagged_topics}

        primary_keywords: list[Keyword] = []
        if prioritized_primary_ids:
            primary_result = await self.session.execute(
                select(Keyword).where(
                    Keyword.id.in_([uuid.UUID(kid) for kid in prioritized_primary_ids])
                )
            )
            primary_keywords = list(primary_result.scalars())

        # For flagged topics, sample top keywords by volume instead of all
        sampled_keywords: list[Keyword] = []
        if flagged_topic_ids:
            flagged_result = await self.session.execute(
                select(Keyword).where(
                    Keyword.topic_id.in_([uuid.UUID(tid) for tid in flagged_topic_ids]),
                    Keyword.status == "active",
                )
            )
            all_flagged = list(flagged_result.scalars())

            # Group by topic and take top N by search volume
            by_topic: dict[str, list[Keyword]] = defaultdict(list)
            for kw in all_flagged:
                by_topic[str(kw.topic_id)].append(kw)
            for topic_id, kws in by_topic.items():
                kws.sort(key=lambda k: k.search_volume or 0, reverse=True)
                sampled_keywords.extend(kws[: self.MAX_SAMPLE_PER_TOPIC])

        candidate_keywords = self._merge_keyword_candidates(primary_keywords, sampled_keywords)

        # Skip keywords that already have SERP data from a previous run
        already_validated = [kw for kw in candidate_keywords if kw.serp_top_results is not None]
        candidate_keywords = [kw for kw in candidate_keywords if kw.serp_top_results is None]
        if already_validated:
            logger.info(
                "Skipping keywords with existing SERP data",
                extra={"skipped": len(already_validated), "remaining": len(candidate_keywords)},
            )

        # Enforce API call cap
        if len(candidate_keywords) > self.MAX_API_CALLS:
            logger.warning(
                "Capping SERP validation to budget limit",
                extra={"candidates": len(candidate_keywords), "cap": self.MAX_API_CALLS},
            )
            candidate_keywords = candidate_keywords[: self.MAX_API_CALLS]
        if not candidate_keywords:
            return SerpValidationOutput(
                keywords_candidate=0,
                keywords_validated=0,
                keywords_mismatched=0,
                keywords_failed=0,
                api_calls_made=0,
                warnings=[],
            )

        await self._update_progress(
            20,
            f"Fetching SERPs for {len(candidate_keywords)} keywords...",
        )

        location_code = get_location_code(project.primary_locale or "en-US")
        language_code = project.primary_language or "en"

        serp_payloads, api_calls_made, warnings = await self._fetch_serp_payloads(
            candidate_keywords,
            location_code=location_code,
            language_code=language_code,
        )

        await self._update_progress(75, "Applying deterministic validation rules...")

        keywords_validated = 0
        keywords_mismatched = 0
        keywords_failed = 0

        for keyword_model, serp_payload in zip(candidate_keywords, serp_payloads):
            fetch_failed = bool(serp_payload.get("error"))
            organic_results = (
                serp_payload.get("organic_results", [])
                if isinstance(serp_payload.get("organic_results"), list)
                else []
            )
            serp_features = self._normalize_serp_features(serp_payload.get("serp_features"))

            if fetch_failed:
                keywords_failed += 1
                keyword_model.serp_top_results = []
                keyword_model.serp_features = []
                keyword_model.validated_intent = None
                keyword_model.validated_page_type = None
                keyword_model.format_requirements = []
                keyword_model.serp_mismatch_flags = self._build_mismatch_flags(
                    keyword_model,
                    validated_intent=None,
                    validated_page_type=None,
                    fetch_failed=True,
                    no_organic=False,
                )
                continue

            validated_intent = self._infer_validated_intent(
                keyword_text=keyword_model.keyword,
                organic_results=organic_results,
                serp_features=serp_features,
            )
            validated_page_type = self._infer_validated_page_type(
                keyword_text=keyword_model.keyword,
                organic_results=organic_results,
                serp_features=serp_features,
                validated_intent=validated_intent,
            )
            format_requirements = self._derive_format_requirements(
                validated_page_type=validated_page_type,
                organic_results=organic_results,
                serp_features=serp_features,
            )
            mismatch_flags = self._build_mismatch_flags(
                keyword_model,
                validated_intent=validated_intent,
                validated_page_type=validated_page_type,
                fetch_failed=False,
                no_organic=len(organic_results) == 0,
            )

            keyword_model.serp_top_results = organic_results[: self.SERP_DEPTH]
            keyword_model.serp_features = serp_features
            keyword_model.validated_intent = validated_intent
            keyword_model.validated_page_type = validated_page_type
            keyword_model.format_requirements = format_requirements
            keyword_model.serp_mismatch_flags = mismatch_flags

            keywords_validated += 1
            if any(flag in {"intent_mismatch", "page_type_mismatch"} for flag in mismatch_flags):
                keywords_mismatched += 1

        logger.info(
            "SERP validation complete",
            extra={
                "project_id": input_data.project_id,
                "keywords_candidate": len(candidate_keywords),
                "keywords_validated": keywords_validated,
                "keywords_mismatched": keywords_mismatched,
                "keywords_failed": keywords_failed,
                "warnings": len(warnings),
            },
        )

        await self._update_progress(100, "SERP validation complete")

        return SerpValidationOutput(
            keywords_candidate=len(candidate_keywords),
            keywords_validated=keywords_validated,
            keywords_mismatched=keywords_mismatched,
            keywords_failed=keywords_failed,
            api_calls_made=api_calls_made,
            warnings=warnings,
        )

    def _topic_needs_serp_validation(self, topic: Topic) -> bool:
        """Return True when topic should validate all cluster keywords."""
        notes = (topic.cluster_notes or "").lower()
        flagged_by_note = "[needs_serp_validation]" in notes
        coherence = topic.cluster_coherence if topic.cluster_coherence is not None else 1.0
        flagged_by_coherence = coherence < self.LOW_COHERENCE_THRESHOLD
        return flagged_by_note or flagged_by_coherence

    def _merge_keyword_candidates(
        self,
        primary_keywords: list[Keyword],
        flagged_keywords: list[Keyword],
    ) -> list[Keyword]:
        """Merge candidate keyword sets with deterministic de-duplication."""
        deduped: dict[str, Keyword] = {}
        for keyword_model in primary_keywords + flagged_keywords:
            deduped[str(keyword_model.id)] = keyword_model
        return list(deduped.values())

    async def _fetch_serp_payloads(
        self,
        keyword_models: list[Keyword],
        location_code: int,
        language_code: str,
    ) -> tuple[list[dict[str, Any]], int, list[str]]:
        """Fetch SERP payloads in batches with best-effort error handling."""
        default_payloads = [
            self._build_failed_serp_payload(
                keyword=keyword_model.keyword,
                reason="SERP fetch unavailable",
            )
            for keyword_model in keyword_models
        ]
        warnings: list[str] = []
        api_calls_made = 0

        try:
            async with DataForSEOClient() as client:
                total_batches = (
                    len(keyword_models) + self.SERP_BATCH_SIZE - 1
                ) // self.SERP_BATCH_SIZE

                for batch_index in range(total_batches):
                    start = batch_index * self.SERP_BATCH_SIZE
                    end = min(start + self.SERP_BATCH_SIZE, len(keyword_models))
                    batch_models = keyword_models[start:end]

                    try:
                        batch_payloads = await client.get_serp_batch(
                            keywords=[model.keyword for model in batch_models],
                            location_code=location_code,
                            language_code=language_code,
                            device=self.SERP_DEVICE,
                            depth=self.SERP_DEPTH,
                        )
                        api_calls_made += 1
                    except Exception as exc:
                        warnings.append(
                            f"SERP batch {batch_index + 1}/{total_batches} failed: {exc}"
                        )
                        for item_index, model in enumerate(batch_models):
                            default_payloads[start + item_index] = self._build_failed_serp_payload(
                                keyword=model.keyword,
                                reason="SERP fetch failed",
                            )
                        continue

                    for item_index, model in enumerate(batch_models):
                        if item_index >= len(batch_payloads):
                            default_payloads[start + item_index] = self._build_failed_serp_payload(
                                keyword=model.keyword,
                                reason="SERP payload missing",
                            )
                            continue

                        payload = batch_payloads[item_index]
                        if not isinstance(payload, dict):
                            default_payloads[start + item_index] = self._build_failed_serp_payload(
                                keyword=model.keyword,
                                reason="SERP payload invalid",
                            )
                            continue

                        payload.setdefault("keyword", model.keyword)
                        payload.setdefault("organic_results", [])
                        payload.setdefault("serp_features", [])
                        default_payloads[start + item_index] = payload
        except APIKeyMissingError:
            warnings.append("SERP validation skipped: DataForSEO credentials are missing.")
        except Exception as exc:
            warnings.append(f"SERP validation degraded: failed to initialize client ({exc}).")

        return default_payloads, api_calls_made, warnings

    def _build_failed_serp_payload(self, keyword: str, reason: str) -> dict[str, Any]:
        """Build normalized failed SERP payload."""
        return {
            "keyword": keyword,
            "organic_results": [],
            "serp_features": [],
            "error": reason,
        }

    def _infer_validated_intent(
        self,
        keyword_text: str,
        organic_results: list[dict[str, Any]],
        serp_features: list[str],
    ) -> str:
        """Infer intent from keyword + top SERP artifacts."""
        scores: dict[str, float] = {
            "informational": 0.0,
            "commercial": 0.0,
            "transactional": 0.0,
            "navigational": 0.0,
        }

        self._score_intent_text(scores, keyword_text, weight=1.2)

        for rank, result in enumerate(organic_results[:5]):
            combined = " ".join(
                [
                    str(result.get("title") or ""),
                    str(result.get("snippet") or ""),
                    str(result.get("url") or ""),
                    str(result.get("domain") or ""),
                ]
            )
            weight = 1.0 if rank < 3 else 0.6
            self._score_intent_text(scores, combined, weight=weight)

        if "shopping" in serp_features:
            scores["transactional"] += 2.0
        if "featured_snippet" in serp_features:
            scores["informational"] += 1.2
        if "paa" in serp_features:
            scores["informational"] += 1.2
        if "local" in serp_features:
            scores["navigational"] += 0.6

        return max(
            ("transactional", "commercial", "informational", "navigational"),
            key=lambda intent: scores.get(intent, 0.0),
        )

    def _score_intent_text(
        self,
        scores: dict[str, float],
        text: str,
        weight: float,
    ) -> None:
        """Add weighted intent scores based on regex pattern matches."""
        normalized = text.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, normalized):
                    scores[intent] += weight

    def _infer_validated_page_type(
        self,
        keyword_text: str,
        organic_results: list[dict[str, Any]],
        serp_features: list[str],
        validated_intent: str,
    ) -> str:
        """Infer page type from keyword + SERP artifact patterns."""
        keyword_page_type = self._first_matching_page_type(keyword_text)
        if keyword_page_type is not None:
            return keyword_page_type

        scores: dict[str, float] = {
            "guide": 0.0,
            "comparison": 0.0,
            "alternatives": 0.0,
            "list": 0.0,
            "landing": 0.0,
            "glossary": 0.0,
            "tool": 0.0,
        }

        for rank, result in enumerate(organic_results[:5]):
            combined = " ".join(
                [
                    str(result.get("title") or ""),
                    str(result.get("url") or ""),
                    str(result.get("snippet") or ""),
                ]
            )
            match = self._first_matching_page_type(combined)
            if match:
                scores[match] += 1.0 if rank < 3 else 0.6

        if "shopping" in serp_features:
            scores["landing"] += 1.5
        if "featured_snippet" in serp_features:
            scores["guide"] += 0.8
        if "paa" in serp_features:
            scores["guide"] += 0.8

        winner = max(scores, key=scores.get)
        if scores[winner] > 0:
            return winner

        # Intent-aligned fallback
        if validated_intent == "transactional":
            return "landing"
        if validated_intent == "commercial":
            return "comparison"
        return "guide"

    def _first_matching_page_type(self, text: str) -> str | None:
        """Return first matching page type for a text block."""
        normalized = text.lower()
        precedence = [
            "alternatives",
            "comparison",
            "landing",
            "tool",
            "guide",
            "glossary",
            "list",
        ]
        for page_type in precedence:
            for pattern in self.PAGE_TYPE_PATTERNS.get(page_type, []):
                if re.search(pattern, normalized):
                    return page_type
        return None

    def _derive_format_requirements(
        self,
        validated_page_type: str,
        organic_results: list[dict[str, Any]],
        serp_features: list[str],
    ) -> list[str]:
        """Derive competitor content format hints from SERP artifacts."""
        hints: set[str] = set()

        hints_by_page_type = {
            "comparison": {"comparison_content", "comparison_table"},
            "alternatives": {"alternatives_roundup", "pros_cons_sections"},
            "list": {"ranked_list", "quick_comparison_table"},
            "guide": {"step_by_step_sections", "faq_section"},
            "landing": {"pricing_blocks", "strong_cta_sections"},
            "glossary": {"definition_first_paragraph", "term_examples"},
            "tool": {"usage_instructions", "interactive_examples"},
        }
        hints.update(hints_by_page_type.get(validated_page_type, set()))

        for result in organic_results[:5]:
            url = str(result.get("url") or "").lower()
            domain = str(result.get("domain") or "").lower()
            title = str(result.get("title") or "").lower()

            if "youtube.com" in domain or "video" in title:
                hints.add("embedded_video_support")
            if "/blog/" in url:
                hints.add("editorial_blog_format")
            if "/pricing" in url:
                hints.add("pricing_comparison")
            if re.search(r"\b(faq|questions)\b", title):
                hints.add("faq_section")

        if "paa" in serp_features:
            hints.add("faq_section")
        if "featured_snippet" in serp_features:
            hints.add("snippet_focused_summary")
        if "shopping" in serp_features:
            hints.add("product_cards_or_pricing")
        if "images" in serp_features:
            hints.add("visual_examples")
        if "news" in serp_features:
            hints.add("freshness_signals")

        return sorted(hints)

    def _build_mismatch_flags(
        self,
        keyword_model: Keyword,
        validated_intent: str | None,
        validated_page_type: str | None,
        fetch_failed: bool,
        no_organic: bool,
    ) -> list[str]:
        """Build mismatch flags without mutating Step 5 source fields."""
        flags: list[str] = []
        if fetch_failed:
            flags.append("serp_fetch_failed")
        if no_organic:
            flags.append("no_organic_results")

        if (
            validated_intent
            and keyword_model.intent
            and validated_intent != keyword_model.intent
        ):
            flags.append("intent_mismatch")
        if (
            validated_page_type
            and keyword_model.recommended_page_type
            and validated_page_type != keyword_model.recommended_page_type
        ):
            flags.append("page_type_mismatch")

        return flags

    def _normalize_serp_features(self, value: Any) -> list[str]:
        """Normalize SERP feature payload into deterministic lower-case strings."""
        if not isinstance(value, list):
            return []
        normalized: set[str] = set()
        for item in value:
            if item is None:
                continue
            feature = str(item).strip().lower()
            if feature:
                normalized.add(feature)
        return sorted(normalized)

    async def _persist_results(self, result: SerpValidationOutput) -> None:
        """Persist Step 8 result summary and project step state."""
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, 8)

        self.set_result_summary({
            "keywords_candidate": result.keywords_candidate,
            "keywords_validated": result.keywords_validated,
            "keywords_mismatched": result.keywords_mismatched,
            "keywords_failed": result.keywords_failed,
            "api_calls_made": result.api_calls_made,
            "keywords_successfully_fetched": max(
                0,
                result.keywords_validated - result.keywords_failed,
            ),
            "warnings": result.warnings,
        })
        await self.session.commit()
