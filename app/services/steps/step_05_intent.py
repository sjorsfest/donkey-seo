"""Step 5: Intent Labeling + Page Type Recommendation.

Classifies search intent and recommends content format for each keyword.
Uses deterministic rules first, then LLM for ambiguous cases.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.agents.intent_classifier import IntentClassifierAgent, IntentClassifierInput
from app.models.brand import BrandProfile
from app.models.keyword import Keyword
from app.models.project import Project
from app.services.discovery_capabilities import CAPABILITY_INTENT_CLASSIFICATION
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class IntentInput:
    """Input for Step 5."""

    project_id: str


@dataclass
class IntentOutput:
    """Output from Step 5."""

    keywords_labeled: int
    keywords_by_rule: int
    keywords_by_llm: int
    keywords: list[dict[str, Any]] = field(default_factory=list)


# Deterministic intent rules (applied before LLM)
INTENT_RULES = {
    # Commercial intent patterns
    "commercial": [
        r"\bbest\b",
        r"\balternatives?\b",
        r"\bvs\b",
        r"\bversus\b",
        r"\bcompar",
        r"\breview",
        r"\btop\s+\d+",
        r"\bfor\s+(small\s+)?business",
    ],
    # Transactional intent patterns
    "transactional": [
        r"\bbuy\b",
        r"\bprice",
        r"\bpricing\b",
        r"\bcost\b",
        r"\bpurchase\b",
        r"\border\b",
        r"\bdiscount\b",
        r"\bcoupon\b",
        r"\bdeal\b",
        r"\bfree\s+trial\b",
        r"\bsign\s*up\b",
    ],
    # Informational intent patterns
    "informational": [
        r"\bhow\s+to\b",
        r"\bwhat\s+is\b",
        r"\bwhat\s+are\b",
        r"\bwhy\b",
        r"\bwhen\b",
        r"\bguide\b",
        r"\btutorial\b",
        r"\bexample",
        r"\btemplate\b",
        r"\bchecklist\b",
        r"\btips\b",
        r"\blearn\b",
    ],
    # Navigational intent patterns
    "navigational": [
        r"\blogin\b",
        r"\bsign\s*in\b",
        r"\bdashboard\b",
        r"\bdownload\b",
        r"\bapp\b",
    ],
}

# Page type recommendations based on patterns
PAGE_TYPE_RULES = {
    "comparison": [r"\bvs\b", r"\bversus\b", r"\bcompar"],
    "alternatives": [r"\balternatives?\b"],
    "list": [r"\btop\s+\d+", r"\bbest\b", r"\b\d+\s+(best|top|ways|tips)"],
    "guide": [r"\bhow\s+to\b", r"\bguide\b", r"\btutorial\b"],
    "glossary": [r"\bwhat\s+is\b", r"\bwhat\s+are\b", r"\bdefin"],
    "landing": [r"\bpric", r"\bbuy\b", r"\bfree\s+trial"],
    "tool": [r"\bcalculat", r"\btool\b", r"\bgenerator\b", r"\btemplate\b"],
}

# Risk flag patterns
RISK_PATTERNS = {
    "local": [r"\bnear\s+me\b", r"\bin\s+\w+\s+(city|town)", r"\blocal\b"],
    "ugc_dominated": [r"\breddit\b", r"\bquora\b", r"\bforum\b"],
    "seasonal": [r"\b(christmas|halloween|black\s*friday|cyber\s*monday)\b"],
}
WORKFLOW_INTENT_TERMS = {
    "integrate",
    "integration",
    "webhook",
    "api",
    "sync",
    "connect",
    "automation",
    "workflow",
    "route",
    "notify",
}
COMPARISON_TERMS = {
    "alternative",
    "alternatives",
    "replace",
    "replacement",
    "vs",
    "versus",
    "instead",
}


class Step05IntentService(BaseStepService[IntentInput, IntentOutput]):
    """Step 5: Intent Labeling + Page Type Recommendation.

    Classification approach:
    1. Apply deterministic rules first (fast, consistent, no API cost)
    2. Batch remaining keywords to LLM for classification
    3. Assign funnel stage based on intent
    """

    step_number = 5
    step_name = "intent_labeling"
    capability_key = CAPABILITY_INTENT_CLASSIFICATION
    is_optional = False

    LLM_BATCH_SIZE = 30  # Keywords per LLM call
    LLM_MAX_CONCURRENT_BATCHES = 3

    async def _validate_preconditions(self, input_data: IntentInput) -> None:
        """Validate Step 4 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 4:
            raise ValueError("Step 4 (Metrics) must be completed first")

    async def _execute(self, input_data: IntentInput) -> IntentOutput:
        """Execute intent classification."""
        # Load brand profile for context
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        brand_context = ""
        if brand:
            products = [
                p.get("name", "")
                for p in (brand.products_services or [])[:5]
            ]
            brand_context = (
                f"Brand: {brand.company_name}. "
                f"Products: {', '.join(products)}"
            )
        learning_context = await self.build_learning_context(
            self.capability_key,
            "IntentClassifierAgent",
        )
        if learning_context:
            brand_context = (
                f"{brand_context}\n\n{learning_context}"
                if brand_context
                else learning_context
            )

        await self._update_progress(5, "Loading keywords...")

        # Load all active keywords
        keywords_result = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == input_data.project_id,
                Keyword.status == "active",
            )
        )
        all_keywords = list(keywords_result.scalars())

        await self._update_progress(10, f"Classifying {len(all_keywords)} keywords...")

        # Apply deterministic rules first
        rule_classified: list[tuple[Keyword, dict]] = []
        needs_llm: list[Keyword] = []

        for kw in all_keywords:
            classification = self._apply_rules(kw.keyword)
            if classification:
                rule_classified.append((kw, classification))
            else:
                needs_llm.append(kw)

        logger.info(
            "Intent rule classification done",
            extra={
                "project_id": input_data.project_id,
                "rule_classified": len(rule_classified),
                "needs_llm": len(needs_llm),
                "total_keywords": len(all_keywords),
            },
        )

        await self._update_progress(
            30,
            f"Rules classified {len(rule_classified)}, "
            f"LLM needed for {len(needs_llm)}"
        )

        # Process remaining with LLM
        llm_classified: list[tuple[Keyword, dict]] = []

        if needs_llm:
            total_batches = (len(needs_llm) + self.LLM_BATCH_SIZE - 1) // self.LLM_BATCH_SIZE
            keyword_positions = {id(kw): idx for idx, kw in enumerate(needs_llm)}
            max_concurrent = min(self.LLM_MAX_CONCURRENT_BATCHES, total_batches)
            semaphore = asyncio.Semaphore(max_concurrent)

            logger.info(
                "Intent LLM batching",
                extra={
                    "project_id": input_data.project_id,
                    "total_batches": total_batches,
                    "batch_size": self.LLM_BATCH_SIZE,
                    "max_concurrent_batches": max_concurrent,
                },
            )

            await self._update_progress(
                35,
                f"Running {total_batches} LLM batches (max {max_concurrent} concurrent)...",
            )

            async def _process_batch(
                batch_num: int,
                batch: list[Keyword],
            ) -> tuple[int, list[tuple[Keyword, dict]]]:
                batch_size = len(batch)
                async with semaphore:
                    logger.info(
                        "Intent LLM batch started",
                        extra={
                            "batch_num": batch_num + 1,
                            "total_batches": total_batches,
                            "batch_size": batch_size,
                        },
                    )
                    agent = IntentClassifierAgent()
                    batch_results: list[tuple[Keyword, dict]] = []
                    used_fallback = False
                    try:
                        agent_input = IntentClassifierInput(
                            keywords=[kw.keyword for kw in batch],
                            context=brand_context,
                        )
                        output = await agent.run(agent_input)

                        # Match classifications to keywords
                        kw_map = {kw.keyword.lower(): kw for kw in batch}
                        for classification in output.classifications:
                            kw = kw_map.get(classification.keyword.lower())
                            if kw:
                                batch_results.append((kw, {
                                    "intent": classification.intent_label,
                                    "intent_confidence": classification.intent_confidence,
                                    "page_type": classification.recommended_page_type,
                                    "funnel_stage": classification.funnel_stage,
                                    "rationale": classification.intent_rationale,
                                    "risk_flags": classification.risk_flags,
                                }))
                    except Exception:
                        logger.warning(
                            "Intent LLM batch failed, using fallback",
                            extra={
                                "batch_num": batch_num + 1,
                                "total_batches": total_batches,
                                "batch_size": batch_size,
                            },
                        )
                        used_fallback = True
                        # Fallback: mark as informational with low confidence
                        for kw in batch:
                            batch_results.append((kw, {
                                "intent": "informational",
                                "intent_confidence": 0.3,
                                "page_type": "guide",
                                "funnel_stage": "tofu",
                                "rationale": "Fallback classification",
                                "risk_flags": ["ambiguous"],
                            }))

                    logger.info(
                        "Intent LLM batch completed",
                        extra={
                            "batch_num": batch_num + 1,
                            "total_batches": total_batches,
                            "result_count": len(batch_results),
                            "fallback_used": used_fallback,
                        },
                    )

                    return batch_num, batch_results

            tasks = []
            for batch_num in range(total_batches):
                start_idx = batch_num * self.LLM_BATCH_SIZE
                end_idx = min(start_idx + self.LLM_BATCH_SIZE, len(needs_llm))
                batch = needs_llm[start_idx:end_idx]
                tasks.append(asyncio.create_task(_process_batch(batch_num, batch)))

            completed = 0
            for completed_task in asyncio.as_completed(tasks):
                _, batch_results = await completed_task
                llm_classified.extend(batch_results)
                completed += 1
                progress = 35 + int((completed / total_batches) * 55)
                await self._update_progress(
                    progress,
                    f"LLM batches complete: {completed}/{total_batches}",
                )

            # Keep output stable to match keyword load order
            llm_classified.sort(key=lambda item: keyword_positions.get(id(item[0]), len(needs_llm)))

        await self._update_progress(95, "Finalizing classifications...")

        # Combine results
        all_classified = rule_classified + llm_classified
        result_keywords = []

        for kw, classification in all_classified:
            # Update keyword model
            discovery_signals = (
                kw.discovery_signals
                if isinstance(kw.discovery_signals, dict)
                else {}
            )
            intent_layer = self._derive_intent_layer(
                keyword=kw.keyword,
                intent=classification["intent"],
                discovery_signals=discovery_signals,
            )
            intent_score = self._calculate_intent_score(
                intent_layer=intent_layer,
                intent_confidence=classification.get("intent_confidence", 0.8),
                discovery_signals=discovery_signals,
            )
            kw.intent = classification["intent"]
            kw.intent_layer = intent_layer
            kw.intent_score = intent_score
            kw.intent_confidence = classification.get("intent_confidence", 0.8)
            kw.recommended_page_type = classification["page_type"]
            kw.page_type_rationale = classification.get("rationale")
            kw.funnel_stage = classification["funnel_stage"]
            kw.risk_flags = classification.get("risk_flags", [])

            result_keywords.append({
                "keyword_id": str(kw.id),
                "keyword": kw.keyword,
                "intent_label": classification["intent"],
                "intent_layer": intent_layer,
                "intent_score": intent_score,
                "intent_confidence": classification.get("intent_confidence", 0.8),
                "recommended_page_type": classification["page_type"],
                "funnel_stage": classification["funnel_stage"],
                "risk_flags": classification.get("risk_flags", []),
            })

        logger.info(
            "Intent classification complete",
            extra={
                "total_labeled": len(all_classified),
                "by_rule": len(rule_classified),
                "by_llm": len(llm_classified),
            },
        )

        await self._update_progress(100, "Intent classification complete")

        return IntentOutput(
            keywords_labeled=len(all_classified),
            keywords_by_rule=len(rule_classified),
            keywords_by_llm=len(llm_classified),
            keywords=result_keywords,
        )

    def _apply_rules(self, keyword: str) -> dict | None:
        """Apply deterministic rules to classify a keyword.

        Returns classification dict if rules match, None if LLM needed.
        """
        kw_lower = keyword.lower()

        # Check intent patterns
        matched_intent = None
        for intent, patterns in INTENT_RULES.items():
            for pattern in patterns:
                if re.search(pattern, kw_lower):
                    matched_intent = intent
                    break
            if matched_intent:
                break

        if not matched_intent:
            return None  # Need LLM

        # Determine page type
        page_type = "guide"  # Default
        for ptype, patterns in PAGE_TYPE_RULES.items():
            for pattern in patterns:
                if re.search(pattern, kw_lower):
                    page_type = ptype
                    break

        # Determine funnel stage
        funnel_map = {
            "informational": "tofu",
            "commercial": "mofu",
            "transactional": "bofu",
            "navigational": "bofu",
        }
        funnel_stage = funnel_map.get(matched_intent, "tofu")

        # Check for risk flags
        risk_flags = []
        for flag, patterns in RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, kw_lower):
                    risk_flags.append(flag)
                    break

        return {
            "intent": matched_intent,
            "intent_confidence": 0.9,  # High confidence for rule-based
            "page_type": page_type,
            "funnel_stage": funnel_stage,
            "rationale": "Classified by deterministic rules",
            "risk_flags": risk_flags,
        }

    def _derive_intent_layer(
        self,
        *,
        keyword: str,
        intent: str,
        discovery_signals: dict[str, Any],
    ) -> str:
        """Derive secondary intent taxonomy for market-aware discovery."""
        keyword_lower = keyword.lower()
        is_comparison = bool(discovery_signals.get("is_comparison")) or any(
            re.search(rf"\b{re.escape(term)}\b", keyword_lower) for term in COMPARISON_TERMS
        )
        if is_comparison:
            return "comparison_replacement"

        is_workflow = (
            bool(discovery_signals.get("has_action_verb"))
            or bool(discovery_signals.get("has_integration_term"))
            or bool(discovery_signals.get("has_two_entities"))
            or any(re.search(rf"\b{re.escape(term)}\b", keyword_lower) for term in WORKFLOW_INTENT_TERMS)
        )
        if is_workflow:
            return "workflow_integration"

        word_count = int(discovery_signals.get("word_count") or len(keyword_lower.split()))
        if intent == "informational" and word_count <= 2:
            return "category"

        return "solution_feature"

    def _calculate_intent_score(
        self,
        *,
        intent_layer: str,
        intent_confidence: float,
        discovery_signals: dict[str, Any],
    ) -> float:
        """Calculate normalized intent strength score (0-1)."""
        base_by_layer = {
            "category": 0.45,
            "solution_feature": 0.62,
            "workflow_integration": 0.80,
            "comparison_replacement": 0.78,
        }
        score = base_by_layer.get(intent_layer, 0.6)
        if discovery_signals.get("has_action_verb"):
            score += 0.07
        if discovery_signals.get("has_integration_term"):
            score += 0.06
        if discovery_signals.get("has_two_entities"):
            score += 0.05
        if discovery_signals.get("is_comparison"):
            score += 0.05

        confidence = max(0.0, min(1.0, float(intent_confidence)))
        blended = (score * 0.8) + (confidence * 0.2)
        return round(max(0.0, min(1.0, blended)), 4)

    async def _persist_results(self, result: IntentOutput) -> None:
        """Save intent classifications (already updated during execution)."""
        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 5

        # Set result summary
        self.set_result_summary({
            "keywords_labeled": result.keywords_labeled,
            "keywords_by_rule": result.keywords_by_rule,
            "keywords_by_llm": result.keywords_by_llm,
            "rule_classification_rate": (
                result.keywords_by_rule / result.keywords_labeled * 100
                if result.keywords_labeled > 0 else 0
            ),
        })

        await self.session.commit()
