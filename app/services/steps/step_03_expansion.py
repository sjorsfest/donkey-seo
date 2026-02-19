"""Step 3: Keyword Expansion.

Expands seed topics into keyword candidates using DataForSEO API.
Includes budget control to prevent API cost explosion.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.integrations.dataforseo import DataForSEOClient, get_location_code
from app.models.brand import BrandProfile
from app.models.generated_dtos import KeywordCreateDTO
from app.models.keyword import Keyword, SeedTopic
from app.models.project import Project
from app.services.market_diagnosis import (
    collect_known_entities,
    diagnose_market_mode,
    extract_keyword_discovery_signals,
)
from app.services.discovery_capabilities import CAPABILITY_KEYWORD_EXPANSION
from app.services.run_strategy import RunStrategy
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class ExpansionInput:
    """Input for Step 3."""

    project_id: str


@dataclass
class ExpansionOutput:
    """Output from Step 3."""

    keywords_generated: int
    api_calls_made: int
    budget_remaining: int
    seeds_processed: int
    seeds_skipped: int
    keywords_excluded: int
    market_mode: str
    keywords: list[dict[str, Any]] = field(default_factory=list)


class Step03ExpansionService(BaseStepService[ExpansionInput, ExpansionOutput]):
    """Step 3: Keyword Expansion.

    Uses DataForSEO API to expand seed topics into keyword candidates.
    Implements budget control to prevent API cost explosion.

    Budget Control Settings (from Project.settings):
    - expansion_budget: Max keywords to generate (default 500)
    - seeds_per_pillar: Limit seeds expanded per pillar (default 5)
    - api_calls_per_step: Max API calls per step run (default 50)
    """

    step_number = 3
    step_name = "keyword_expansion"
    capability_key = CAPABILITY_KEYWORD_EXPANSION
    is_optional = False

    # Default budget settings
    DEFAULT_EXPANSION_BUDGET = 500
    DEFAULT_SEEDS_PER_PILLAR = 5
    DEFAULT_API_CALLS_PER_STEP = 50
    DEFAULT_SUGGESTIONS_PER_SEED = 80
    DEFAULT_QUESTIONS_PER_SEED = 30
    COMPARISON_MODIFIERS = (
        " vs ",
        " versus ",
        " alternative",
        " alternatives",
        " comparison",
        " compare ",
    )
    STRATEGIC_TOKEN_STOPWORDS = {
        "and",
        "for",
        "with",
        "without",
        "from",
        "into",
        "that",
        "this",
        "your",
        "their",
        "the",
        "a",
        "an",
        "support",
        "customer",
        "service",
        "services",
        "software",
        "tool",
        "tools",
        "platform",
        "system",
        "systems",
        "solution",
        "solutions",
        "team",
        "teams",
    }

    async def _validate_preconditions(self, input_data: ExpansionInput) -> None:
        """Validate Step 2 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 2:
            raise ValueError("Step 2 (Seed Topics) must be completed first")

        # Check seed topics exist
        seeds_result = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == input_data.project_id)
        )
        if not seeds_result.scalars().first():
            raise ValueError("No seed topics found. Run Step 2 first.")

    async def _execute(self, input_data: ExpansionInput) -> ExpansionOutput:
        """Execute keyword expansion with budget control."""
        # Load project and settings
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

        # Load brand profile for exclusion filters
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()
        strategy = await self.get_run_strategy()

        # Get budget settings from project
        settings = project.api_budget_caps or {}
        expansion_budget = settings.get("expansion_budget", self.DEFAULT_EXPANSION_BUDGET)
        seeds_per_pillar = settings.get("seeds_per_pillar", self.DEFAULT_SEEDS_PER_PILLAR)
        api_calls_limit = settings.get("api_calls_per_step", self.DEFAULT_API_CALLS_PER_STEP)
        logger.info("Starting keyword expansion", extra={"project_id": input_data.project_id, "expansion_budget": expansion_budget, "seeds_per_pillar": seeds_per_pillar, "api_calls_limit": api_calls_limit})

        # Get locale for API calls
        locale = project.primary_locale or "en-US"
        location_code = get_location_code(locale)
        language_code = project.primary_language or "en"

        await self._update_progress(5, "Loading seed topics...")

        # Load seed topics, grouped by pillar
        seeds_result = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == input_data.project_id)
        )
        all_seeds = list(seeds_result.scalars())

        # Group seeds by pillar and select top N per pillar
        pillar_seeds: dict[str, list[SeedTopic]] = {}
        for seed in all_seeds:
            pillar = seed.pillar_type or "default"
            if pillar not in pillar_seeds:
                pillar_seeds[pillar] = []
            pillar_seeds[pillar].append(seed)

        # Select top seeds per pillar (by relevance score)
        selected_seeds = []
        for pillar, seeds in pillar_seeds.items():
            sorted_seeds = sorted(
                seeds,
                key=lambda s: s.relevance_score or 0,
                reverse=True
            )
            selected_seeds.extend(sorted_seeds[:seeds_per_pillar])

        seeds_skipped = len(all_seeds) - len(selected_seeds)
        logger.info("Seeds selected for expansion", extra={"selected_count": len(selected_seeds), "skipped_count": seeds_skipped})

        await self._update_progress(10, f"Selected {len(selected_seeds)} seeds for expansion...")

        # Prepare exclusion filters
        out_of_scope: set[str] = set(t.lower() for t in strategy.exclude_topics)
        own_brand_terms, competitor_terms = self._extract_brand_terms(brand)
        competitor_terms.update(self._extract_competitor_terms_from_seeds(selected_seeds))
        strategic_terms = self._collect_strategic_terms(
            brand=brand,
            strategy=strategy,
            selected_seeds=selected_seeds,
        )

        # Collect all seed phrases for batched API calls
        seed_phrases = [seed.name for seed in selected_seeds]
        known_entities = collect_known_entities(
            brand=brand,
            seed_terms=seed_phrases + list(strategy.include_topics),
        )

        all_keywords: list[dict[str, Any]] = []
        seen_keywords: set[str] = set()
        api_calls_made = 0
        excluded_count = 0
        active_by_seed: dict[str, int] = {str(seed.id): 0 for seed in selected_seeds}
        per_seed_caps = self._build_seed_caps(selected_seeds, expansion_budget)

        suggestions_per_seed = max(
            10,
            int(settings.get("suggestions_per_seed", self.DEFAULT_SUGGESTIONS_PER_SEED)),
        )
        questions_per_seed = max(
            5,
            int(settings.get("questions_per_seed", self.DEFAULT_QUESTIONS_PER_SEED)),
        )

        async with DataForSEOClient() as client:
            seed_count = len(selected_seeds)

            # Pass 1: suggestions per seed with even per-seed caps.
            await self._update_progress(20, f"Fetching suggestions for {seed_count} seeds...")
            for index, seed in enumerate(selected_seeds):
                if sum(active_by_seed.values()) >= expansion_budget:
                    break
                if api_calls_made >= api_calls_limit:
                    break

                progress = 20 + int((index / max(seed_count, 1)) * 30)
                await self._update_progress(
                    progress,
                    f"Suggestions for seed {index + 1}/{seed_count}: {seed.name}",
                )

                try:
                    suggestions = await client.get_keyword_suggestions(
                        seeds=[seed.name],
                        location_code=location_code,
                        language_code=language_code,
                        limit=suggestions_per_seed,
                    )
                    api_calls_made += 1
                except Exception:
                    logger.warning(
                        "Keyword suggestion API error",
                        extra={"seed": seed.name, "method": "suggestion"},
                    )
                    continue

                excluded_count += self._ingest_seed_keyword_rows(
                    rows=suggestions,
                    seed=seed,
                    source_method="suggestion",
                    known_entities=known_entities,
                    seen_keywords=seen_keywords,
                    out_of_scope=out_of_scope,
                    strategy=strategy,
                    own_brand_terms=own_brand_terms,
                    competitor_terms=competitor_terms,
                    strategic_terms=strategic_terms,
                    all_keywords=all_keywords,
                    active_by_seed=active_by_seed,
                    seed_active_cap=per_seed_caps[str(seed.id)],
                )

            # Pass 2: question variants for seeds still under cap.
            if (
                sum(active_by_seed.values()) < expansion_budget
                and api_calls_made < api_calls_limit
            ):
                await self._update_progress(55, f"Fetching question variants for up to {seed_count} seeds...")
                for index, seed in enumerate(selected_seeds):
                    if sum(active_by_seed.values()) >= expansion_budget:
                        break
                    if api_calls_made >= api_calls_limit:
                        break
                    if active_by_seed.get(str(seed.id), 0) >= per_seed_caps[str(seed.id)]:
                        continue

                    progress = 55 + int((index / max(seed_count, 1)) * 25)
                    await self._update_progress(
                        progress,
                        f"Questions for seed {index + 1}/{seed_count}: {seed.name}",
                    )

                    try:
                        questions = await client.get_keyword_questions(
                            keywords=[seed.name],
                            location_code=location_code,
                            language_code=language_code,
                            limit=questions_per_seed,
                        )
                        api_calls_made += 1
                    except Exception:
                        logger.warning(
                            "Question keywords API error",
                            extra={"seed": seed.name, "method": "questions"},
                        )
                        continue

                    excluded_count += self._ingest_seed_keyword_rows(
                        rows=questions,
                        seed=seed,
                        source_method="questions",
                        known_entities=known_entities,
                        seen_keywords=seen_keywords,
                        out_of_scope=out_of_scope,
                        strategy=strategy,
                        own_brand_terms=own_brand_terms,
                        competitor_terms=competitor_terms,
                        strategic_terms=strategic_terms,
                        all_keywords=all_keywords,
                        active_by_seed=active_by_seed,
                        seed_active_cap=per_seed_caps[str(seed.id)],
                    )

            # Pass 3: overflow fill (no per-seed cap) when budget still has room.
            if (
                sum(active_by_seed.values()) < expansion_budget
                and api_calls_made < api_calls_limit
            ):
                await self._update_progress(82, "Filling remaining budget with overflow suggestions...")
                for seed in sorted(selected_seeds, key=lambda s: s.relevance_score or 0, reverse=True):
                    if sum(active_by_seed.values()) >= expansion_budget:
                        break
                    if api_calls_made >= api_calls_limit:
                        break

                    try:
                        overflow_suggestions = await client.get_keyword_suggestions(
                            seeds=[seed.name],
                            location_code=location_code,
                            language_code=language_code,
                            limit=max(20, suggestions_per_seed // 2),
                        )
                        api_calls_made += 1
                    except Exception:
                        logger.warning(
                            "Overflow suggestion API error",
                            extra={"seed": seed.name, "method": "suggestion_overflow"},
                        )
                        continue

                    excluded_count += self._ingest_seed_keyword_rows(
                        rows=overflow_suggestions,
                        seed=seed,
                        source_method="suggestion",
                        known_entities=known_entities,
                        seen_keywords=seen_keywords,
                        out_of_scope=out_of_scope,
                        strategy=strategy,
                        own_brand_terms=own_brand_terms,
                        competitor_terms=competitor_terms,
                        strategic_terms=strategic_terms,
                        all_keywords=all_keywords,
                        active_by_seed=active_by_seed,
                        seed_active_cap=None,
                    )

        await self._update_progress(95, f"Processed {len(all_keywords)} keywords...")

        active_keywords = sum(active_by_seed.values())
        budget_remaining = max(0, expansion_budget - active_keywords)
        refreshed_diagnosis = diagnose_market_mode(
            source="step3_refresh",
            override=getattr(strategy, "market_mode_override", "auto"),
            seed_terms=seed_phrases,
            keyword_rows=all_keywords,
        )
        await self.set_market_diagnosis(refreshed_diagnosis.to_dict())
        logger.info(
            "Keyword expansion complete",
            extra={
                "keywords_generated": active_keywords,
                "keywords_excluded": excluded_count,
                "api_calls_made": api_calls_made,
                "budget_remaining": budget_remaining,
                "market_mode": refreshed_diagnosis.mode,
            },
        )

        await self._update_progress(100, "Keyword expansion complete")

        return ExpansionOutput(
            keywords_generated=active_keywords,
            api_calls_made=api_calls_made,
            budget_remaining=budget_remaining,
            seeds_processed=len(selected_seeds),
            seeds_skipped=seeds_skipped,
            keywords_excluded=excluded_count,
            market_mode=refreshed_diagnosis.mode,
            keywords=all_keywords,
        )

    async def _validate_output(self, result: ExpansionOutput, input_data: ExpansionInput) -> None:
        """Ensure output can be consumed by Step 4."""
        if result.seeds_processed <= 0:
            raise ValueError(
                "Step 3 processed 0 seeds. Step 4 requires expanded keywords sourced from seeds."
            )

        if result.keywords_generated <= 0:
            raise ValueError(
                "Step 3 generated 0 keywords. Step 4 requires at least one active keyword."
            )

    def _evaluate_keyword_policy(
        self,
        keyword_normalized: str,
        seen: set[str],
        out_of_scope: set[str],
        strategy: RunStrategy,
        own_brand_terms: set[str],
        competitor_terms: set[str],
        strategic_terms: set[str] | None = None,
    ) -> tuple[bool, str | None]:
        """Check if keyword should be included and return exclusion reason when blocked."""
        # Skip if already seen
        if keyword_normalized in seen:
            return False, "duplicate_keyword"

        # Skip if empty
        if not keyword_normalized or len(keyword_normalized) < 2:
            return False, "empty_or_too_short"

        words = [w for w in keyword_normalized.split() if w]
        if len(words) > 12 or keyword_normalized.count("?") > 1:
            return False, "low_relevance_pattern"
        if re.search(r"\b(lorem ipsum|test keyword|dummy)\b", keyword_normalized):
            return False, "low_relevance_pattern"

        # Skip if out of scope
        for exclusion in out_of_scope:
            if exclusion in keyword_normalized:
                return False, f"out_of_scope:{exclusion}"

        # Skip obvious spam/adult content patterns
        spam_patterns = [
            "xxx", "porn", "nude", "naked", "sex",
            "casino", "gambling", "slot machine",
        ]
        for pattern in spam_patterns:
            if pattern in keyword_normalized:
                return False, f"spam_pattern:{pattern}"

        has_own_brand = self._contains_any_term(keyword_normalized, own_brand_terms)
        has_competitor_brand = self._contains_any_term(keyword_normalized, competitor_terms)
        is_comparison_query = any(marker in f" {keyword_normalized} " for marker in self.COMPARISON_MODIFIERS)
        keyword_tokens = self._tokenize(keyword_normalized)

        if strategic_terms and len(strategic_terms) >= 4:
            if (
                not (keyword_tokens & strategic_terms)
                and not has_own_brand
                and not is_comparison_query
            ):
                return False, "low_strategic_relevance"

        if has_competitor_brand and not has_own_brand:
            if strategy.branded_keyword_mode == "exclude_all":
                return False, "competitor_branded"
            if strategy.branded_keyword_mode == "comparisons_only" and not is_comparison_query:
                return False, "competitor_branded_non_comparison"

        return True, None

    def _extract_brand_terms(self, brand: BrandProfile | None) -> tuple[set[str], set[str]]:
        """Extract own-brand and competitor-brand terms for policy filtering."""
        own_brand_terms: set[str] = set()
        competitor_terms: set[str] = set()

        if not brand:
            return own_brand_terms, competitor_terms

        if brand.company_name:
            own_brand_terms.add(brand.company_name.lower())
        for product in brand.products_services or []:
            name = (product.get("name") or "").strip().lower()
            if name:
                own_brand_terms.add(name)

        for competitor in brand.competitor_positioning or []:
            name = (competitor.get("name") or competitor.get("brand") or "").strip().lower()
            if name:
                competitor_terms.add(name)

        return own_brand_terms, competitor_terms

    def _contains_any_term(self, keyword: str, terms: set[str]) -> bool:
        """Return True when a keyword contains any configured term."""
        for term in terms:
            if len(term) < 3:
                continue
            if term in keyword:
                return True
        return False

    def _pick_seed_topic(self, keyword_normalized: str, seeds: list[SeedTopic]) -> SeedTopic | None:
        """Assign keyword provenance to the most similar seed topic."""
        keyword_tokens = self._tokenize(keyword_normalized)
        if not keyword_tokens:
            return None

        best_seed: SeedTopic | None = None
        best_score = 0.0

        for seed in seeds:
            seed_tokens = self._tokenize(seed.name.lower())
            if not seed_tokens:
                continue
            overlap = len(keyword_tokens & seed_tokens)
            score = overlap / len(seed_tokens)
            if score > best_score:
                best_score = score
                best_seed = seed
            elif score == best_score and best_seed is not None:
                if (seed.relevance_score or 0) > (best_seed.relevance_score or 0):
                    best_seed = seed

        return best_seed

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into comparable terms."""
        return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}

    def _build_seed_caps(self, selected_seeds: list[SeedTopic], expansion_budget: int) -> dict[str, int]:
        """Build near-even per-seed active keyword caps that sum to the expansion budget."""
        if not selected_seeds:
            return {}
        count = len(selected_seeds)
        base = max(1, expansion_budget // count)
        remainder = max(0, expansion_budget - (base * count))
        caps: dict[str, int] = {}
        for index, seed in enumerate(selected_seeds):
            extra = 1 if index < remainder else 0
            caps[str(seed.id)] = base + extra
        return caps

    def _ingest_seed_keyword_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        seed: SeedTopic,
        source_method: str,
        known_entities: set[str],
        seen_keywords: set[str],
        out_of_scope: set[str],
        strategy: RunStrategy,
        own_brand_terms: set[str],
        competitor_terms: set[str],
        strategic_terms: set[str],
        all_keywords: list[dict[str, Any]],
        active_by_seed: dict[str, int],
        seed_active_cap: int | None,
    ) -> int:
        """Apply policy filters and append keyword rows for one seed."""
        seed_id = str(seed.id)
        excluded_count = 0
        for row in rows:
            kw_text = str(row.get("keyword") or "").strip()
            if not kw_text:
                continue
            kw_normalized = kw_text.lower()
            if kw_normalized in seen_keywords:
                continue

            include, exclusion_reason = self._evaluate_keyword_policy(
                keyword_normalized=kw_normalized,
                seen=seen_keywords,
                out_of_scope=out_of_scope,
                strategy=strategy,
                own_brand_terms=own_brand_terms,
                competitor_terms=competitor_terms,
                strategic_terms=strategic_terms,
            )
            if include and seed_active_cap is not None and active_by_seed.get(seed_id, 0) >= seed_active_cap:
                # Keep cap fair in the balanced pass; allow later overflow pass.
                continue

            seen_keywords.add(kw_normalized)
            if include:
                active_by_seed[seed_id] = active_by_seed.get(seed_id, 0) + 1
            else:
                excluded_count += 1

            discovery_signals = extract_keyword_discovery_signals(
                kw_text,
                known_entities=known_entities,
            )

            all_keywords.append({
                "keyword_text": kw_text,
                "keyword_normalized": kw_normalized,
                "source_seed_topic": seed.name,
                "seed_topic_id": seed_id,
                "source_method": source_method,
                "search_volume": row.get("search_volume"),
                "cpc": row.get("cpc"),
                "competition": row.get("competition"),
                "exclusion_flags": [] if include else ["policy_filtered"],
                "status": "active" if include else "excluded",
                "exclusion_reason": exclusion_reason,
                "discovery_signals": discovery_signals,
            })
        return excluded_count

    def _collect_strategic_terms(
        self,
        *,
        brand: BrandProfile | None,
        strategy: RunStrategy,
        selected_seeds: list[SeedTopic],
    ) -> set[str]:
        """Collect high-signal terms used to reject off-scope keyword ideas."""
        source_texts: list[str] = []
        source_texts.extend(strategy.include_topics)
        source_texts.extend(strategy.icp_roles)
        source_texts.extend(strategy.icp_industries)
        source_texts.extend(strategy.icp_pains)
        source_texts.extend(strategy.conversion_intents)

        if brand:
            source_texts.extend(brand.in_scope_topics or [])
            for product in brand.products_services or []:
                source_texts.append(str(product.get("name") or ""))
                source_texts.append(str(product.get("category") or ""))
                source_texts.extend([str(item) for item in (product.get("core_benefits") or [])])
            source_texts.extend(brand.unique_value_props or [])

        terms: set[str] = set()
        for text in source_texts:
            terms.update(
                token for token in self._tokenize(str(text))
                if token not in self.STRATEGIC_TOKEN_STOPWORDS
            )

        # Fallback: derive terms from selected seeds when brand/strategy hints are sparse.
        if len(terms) < 6:
            for seed in selected_seeds:
                terms.update(
                    token for token in self._tokenize(seed.name)
                    if token not in self.STRATEGIC_TOKEN_STOPWORDS
                )

        return terms

    def _extract_competitor_terms_from_seeds(self, seeds: list[SeedTopic]) -> set[str]:
        """Infer competitor brands from alternative/vs seeds when profile data is missing."""
        inferred: set[str] = set()
        for seed in seeds:
            keyword = re.sub(r"\s+", " ", (seed.name or "").strip().lower())
            if not keyword:
                continue

            alt_match = re.match(r"^(.+?)\s+alternatives?$", keyword)
            if alt_match:
                candidate = alt_match.group(1).strip()
                if candidate and " " not in candidate:
                    inferred.add(candidate)

            vs_match = re.match(r"^(.+?)\s+(?:vs|versus)\s+(.+)$", keyword)
            if vs_match:
                for candidate in (vs_match.group(1).strip(), vs_match.group(2).strip()):
                    if candidate and " " not in candidate:
                        inferred.add(candidate)

        return inferred

    async def _persist_results(self, result: ExpansionOutput) -> None:
        """Save expanded keywords to database."""
        # Delete existing expansion keywords for this project
        existing = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == self.project_id,
                Keyword.source == "expansion",
            )
        )
        for keyword in existing.scalars():
            await keyword.delete(self.session)

        # Create new keywords
        for kw_data in result.keywords:
            Keyword.create(
                self.session,
                KeywordCreateDTO(
                    project_id=self.project_id,
                    keyword=kw_data["keyword_text"],
                    keyword_normalized=kw_data["keyword_normalized"],
                    source="expansion",
                    seed_topic_id=kw_data.get("seed_topic_id"),
                    source_method=kw_data.get("source_method"),
                    search_volume=kw_data.get("search_volume"),
                    cpc=kw_data.get("cpc"),
                    competition=kw_data.get("competition"),
                    exclusion_flags=kw_data.get("exclusion_flags"),
                    discovery_signals=kw_data.get("discovery_signals"),
                    status=kw_data.get("status", "active"),
                    exclusion_reason=kw_data.get("exclusion_reason"),
                ),
            )

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 3

        # Set result summary
        self.set_result_summary({
            "keywords_generated": result.keywords_generated,
            "api_calls_made": result.api_calls_made,
            "budget_remaining": result.budget_remaining,
            "seeds_processed": result.seeds_processed,
            "seeds_skipped": result.seeds_skipped,
            "keywords_excluded": result.keywords_excluded,
            "market_mode": result.market_mode,
        })

        await self.session.commit()
