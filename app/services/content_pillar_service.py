"""Content pillar taxonomy + assignment service."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content_pillar import ContentBriefPillarAssignment, ContentPillar
from app.models.generated_dtos import (
    ContentBriefPillarAssignmentCreateDTO,
    ContentPillarCreateDTO,
    ContentPillarPatchDTO,
)
from app.models.topic import Topic

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "best",
    "blog",
    "by",
    "for",
    "from",
    "guide",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "vs",
    "what",
    "when",
    "with",
}


@dataclass(slots=True)
class TopicDoc:
    """Normalized topic text view for taxonomy matching."""

    topic_id: str
    text: str
    tokens: list[str]


@dataclass(slots=True)
class PlannedPillarAssignment:
    """Resolved assignment for one topic/brief."""

    topic_id: str
    primary_pillar_id: str
    secondary_pillar_ids: list[str]
    confidence_score: float
    assignment_method: str


class ContentPillarService:
    """Build and assign stable content pillars for project briefs."""

    MAX_ACTIVE_PILLARS = 5
    MIN_BOOTSTRAP_PILLARS = 3
    MIN_PRIMARY_MATCH_SCORE = 0.18
    SECONDARY_MATCH_SCORE = 0.15
    SECONDARY_DELTA = 0.08
    STRICT_NEW_PILLAR_MIN_UNCOVERED = 4
    HIGH_LEVEL_PILLARS: tuple[tuple[str, str], ...] = (
        (
            "Blog",
            "General informational and educational content for broad awareness topics.",
        ),
        (
            "Comparison",
            "Competitor alternatives, versus pages, and evaluation-focused content.",
        ),
        (
            "How To",
            "Step-by-step implementation, setup, and tutorial-style content.",
        ),
        (
            "Use Case",
            "Audience or workflow-specific scenarios and jobs-to-be-done content.",
        ),
        (
            "Commercial",
            "Pricing, reviews, best-of, and evaluation-ready purchase intent content.",
        ),
    )

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def plan_assignments(
        self,
        *,
        project_id: str,
        topics: list[Topic],
    ) -> dict[str, PlannedPillarAssignment]:
        """Plan topic->pillar assignments, creating taxonomy only when strict rules pass."""
        requested_docs = self._topic_docs_from_topics(topics)
        if not requested_docs:
            return {}

        corpus_docs = await self._load_project_topic_docs(project_id)
        if not corpus_docs:
            corpus_docs = requested_docs

        pillars = await self._load_active_pillars(project_id)
        if not pillars:
            pillars = await self._bootstrap_pillars(project_id=project_id, docs=corpus_docs)

        pillars = await self._ensure_high_level_taxonomy(
            project_id=project_id,
            pillars=pillars,
            docs=corpus_docs,
        )

        return self._assign_docs(requested_docs, pillars)

    async def persist_assignments(
        self,
        *,
        project_id: str,
        brief_id: str,
        assignment: PlannedPillarAssignment,
    ) -> None:
        """Persist assignment for a brief while preserving manual curation."""
        existing_result = await self.session.execute(
            select(ContentBriefPillarAssignment).where(
                ContentBriefPillarAssignment.project_id == project_id,
                ContentBriefPillarAssignment.brief_id == brief_id,
            )
        )
        existing = list(existing_result.scalars().all())
        if any((item.assignment_method or "").strip().lower() == "manual" for item in existing):
            return

        for item in existing:
            await item.delete(self.session)

        ContentBriefPillarAssignment.create(
            self.session,
            ContentBriefPillarAssignmentCreateDTO(
                project_id=project_id,
                brief_id=brief_id,
                pillar_id=assignment.primary_pillar_id,
                relationship_type="primary",
                confidence_score=assignment.confidence_score,
                assignment_method=assignment.assignment_method,
            ),
        )

        for pillar_id in assignment.secondary_pillar_ids:
            ContentBriefPillarAssignment.create(
                self.session,
                ContentBriefPillarAssignmentCreateDTO(
                    project_id=project_id,
                    brief_id=brief_id,
                    pillar_id=pillar_id,
                    relationship_type="secondary",
                    confidence_score=None,
                    assignment_method=assignment.assignment_method,
                ),
            )

    async def _load_active_pillars(self, project_id: str) -> list[ContentPillar]:
        result = await self.session.execute(
            select(ContentPillar)
            .where(
                ContentPillar.project_id == project_id,
                ContentPillar.status == "active",
            )
            .order_by(ContentPillar.created_at.asc())
        )
        return list(result.scalars().all())

    async def _load_project_topic_docs(self, project_id: str) -> list[TopicDoc]:
        result = await self.session.execute(
            select(Topic).where(
                Topic.project_id == project_id,
                Topic.fit_tier.in_(["primary", "secondary"]),
            )
        )
        topics = list(result.scalars().all())
        return self._topic_docs_from_topics(topics)

    async def _bootstrap_pillars(
        self,
        *,
        project_id: str,
        docs: list[TopicDoc],
    ) -> list[ContentPillar]:
        names = self._derive_bootstrap_names(docs)
        if not names:
            return []
        return await self._create_pillars(project_id=project_id, names=names, source="auto")

    async def _ensure_high_level_taxonomy(
        self,
        *,
        project_id: str,
        pillars: list[ContentPillar],
        docs: list[TopicDoc],
    ) -> list[ContentPillar]:
        """Keep auto-generated taxonomy fixed to high-level pillar buckets."""
        current = list(pillars)
        if any(
            bool(pillar.locked) or str(pillar.source or "").strip().lower() == "manual"
            for pillar in current
        ):
            return current

        target_names = self._derive_bootstrap_names(docs)
        target_canonicals = {
            canonical
            for canonical in (self._canonical_pillar_key(name) for name in target_names)
            if canonical is not None
        }
        if not target_canonicals:
            target_names = [name for name, _ in self.HIGH_LEVEL_PILLARS[: self.MIN_BOOTSTRAP_PILLARS]]
            target_canonicals = {
                canonical
                for canonical in (self._canonical_pillar_key(name) for name in target_names)
                if canonical is not None
            }

        keep_ids: set[str] = set()
        seen_canonical: set[str] = set()
        for pillar in current:
            canonical = self._canonical_pillar_key(pillar.name)
            if canonical is None or canonical in seen_canonical or canonical not in target_canonicals:
                continue
            keep_ids.add(str(pillar.id))
            seen_canonical.add(canonical)

        for pillar in current:
            if str(pillar.id) in keep_ids:
                continue
            if pillar.status != "archived":
                pillar.patch(
                    self.session,
                    ContentPillarPatchDTO.from_partial({"status": "archived"}),
                )
        await self.session.flush()
        current = await self._load_active_pillars(project_id)

        canonical_to_pillar: dict[str, ContentPillar] = {}
        for pillar in current:
            canonical = self._canonical_pillar_key(pillar.name)
            if canonical:
                canonical_to_pillar[canonical] = pillar

        for pillar_name in target_names:
            canonical = self._canonical_pillar_key(pillar_name)
            if canonical is None:
                continue
            pillar_description = self._high_level_pillar_description(pillar_name)
            existing = canonical_to_pillar.get(canonical)
            if existing is None:
                created = await self._create_pillars(
                    project_id=project_id,
                    names=[pillar_name],
                    source="auto",
                )
                if created:
                    existing = created[0]
                    canonical_to_pillar[canonical] = existing
                    current.extend(created)
            if existing is None:
                continue

            patch_payload: dict[str, str] = {}
            if existing.name != pillar_name:
                patch_payload["name"] = pillar_name
            if pillar_description is not None and existing.description != pillar_description:
                patch_payload["description"] = pillar_description
            if existing.status != "active":
                patch_payload["status"] = "active"

            desired_slug = self._slugify(pillar_name) or "pillar"
            if existing.slug != desired_slug:
                used_slugs = {
                    pillar.slug
                    for pillar in current
                    if pillar.status == "active" and str(pillar.id) != str(existing.id)
                }
                patch_payload["slug"] = self._build_unique_slug(pillar_name, used_slugs)

            if patch_payload:
                existing.patch(
                    self.session,
                    ContentPillarPatchDTO.from_partial(patch_payload),
                )

        await self.session.flush()
        return await self._load_active_pillars(project_id)

    def _collect_uncovered_docs(self, docs: list[TopicDoc], pillars: list[ContentPillar]) -> list[TopicDoc]:
        uncovered: list[TopicDoc] = []
        pillar_token_map = {
            str(pillar.id): self._tokenize(f"{pillar.name} {pillar.description or ''}")
            for pillar in pillars
        }
        for doc in docs:
            best = 0.0
            for tokens in pillar_token_map.values():
                best = max(best, self._match_score(doc.tokens, tokens))
            if best < self.MIN_PRIMARY_MATCH_SCORE:
                uncovered.append(doc)
        return uncovered

    async def _create_pillars(
        self,
        *,
        project_id: str,
        names: list[str],
        source: str,
    ) -> list[ContentPillar]:
        if not names:
            return []

        existing = await self._load_active_pillars(project_id)
        existing_slugs = {pillar.slug for pillar in existing}
        created: list[ContentPillar] = []
        for name in names:
            if len(existing) + len(created) >= self.MAX_ACTIVE_PILLARS:
                break
            clean_name = self._clean_label(name)
            if not clean_name:
                continue
            slug = self._build_unique_slug(clean_name, existing_slugs)
            existing_slugs.add(slug)
            description = self._high_level_pillar_description(clean_name)
            created.append(
                ContentPillar.create(
                    self.session,
                    ContentPillarCreateDTO(
                        project_id=project_id,
                        name=clean_name,
                        slug=slug,
                        description=description,
                        status="active",
                        source=source,
                        locked=False,
                    ),
                )
            )

        if created:
            await self.session.flush()
        return created

    def _assign_docs(
        self,
        docs: list[TopicDoc],
        pillars: list[ContentPillar],
    ) -> dict[str, PlannedPillarAssignment]:
        if not pillars:
            return {}

        high_level_ids = self._high_level_pillar_ids(pillars)
        if high_level_ids:
            return self._assign_docs_with_high_level_taxonomy(
                docs=docs,
                high_level_ids=high_level_ids,
            )

        scored_pillars = [
            (
                pillar,
                self._tokenize(f"{pillar.name} {pillar.description or ''}"),
            )
            for pillar in pillars
        ]
        assignments: dict[str, PlannedPillarAssignment] = {}

        for doc in docs:
            scores: list[tuple[str, float]] = []
            for pillar, tokens in scored_pillars:
                score = self._match_score(doc.tokens, tokens)
                scores.append((str(pillar.id), score))
            scores.sort(key=lambda item: item[1], reverse=True)
            if not scores:
                continue

            primary_id, best_score = scores[0]
            assignment_method = "auto" if best_score >= self.MIN_PRIMARY_MATCH_SCORE else "auto_forced"
            confidence = round(min(1.0, max(0.0, best_score * 2.5)), 4)

            secondary_ids: list[str] = []
            for pillar_id, score in scores[1:]:
                if len(secondary_ids) >= 2:
                    break
                if score < self.SECONDARY_MATCH_SCORE:
                    continue
                if (best_score - score) > self.SECONDARY_DELTA:
                    continue
                secondary_ids.append(pillar_id)

            assignments[doc.topic_id] = PlannedPillarAssignment(
                topic_id=doc.topic_id,
                primary_pillar_id=primary_id,
                secondary_pillar_ids=secondary_ids,
                confidence_score=confidence,
                assignment_method=assignment_method,
            )

        return assignments

    def _assign_docs_with_high_level_taxonomy(
        self,
        *,
        docs: list[TopicDoc],
        high_level_ids: dict[str, str],
    ) -> dict[str, PlannedPillarAssignment]:
        assignments: dict[str, PlannedPillarAssignment] = {}
        for doc in docs:
            labels = self._classify_doc_labels(doc.text)
            primary_label = next((label for label in labels if label in high_level_ids), None)
            if primary_label is None:
                continue
            primary_id = high_level_ids[primary_label]
            secondary_ids: list[str] = []
            for label in labels:
                if label == primary_label:
                    continue
                secondary_id = high_level_ids.get(label)
                if secondary_id is None or secondary_id == primary_id:
                    continue
                secondary_ids.append(secondary_id)
                if len(secondary_ids) >= 2:
                    break
            confidence = 0.9 if primary_label in {"comparison", "how_to"} else 0.78
            assignments[doc.topic_id] = PlannedPillarAssignment(
                topic_id=doc.topic_id,
                primary_pillar_id=primary_id,
                secondary_pillar_ids=secondary_ids,
                confidence_score=confidence,
                assignment_method="auto_high_level",
            )
        return assignments

    def _topic_docs_from_topics(self, topics: list[Topic]) -> list[TopicDoc]:
        docs: list[TopicDoc] = []
        for topic in topics:
            text = " ".join(
                part
                for part in [
                    str(topic.name or "").strip(),
                    str(topic.description or "").strip(),
                    str(topic.cluster_notes or "").strip(),
                ]
                if part
            )
            tokens = self._tokenize(text)
            if not tokens:
                continue
            docs.append(
                TopicDoc(
                    topic_id=str(topic.id),
                    text=text,
                    tokens=tokens,
                )
            )
        return docs

    def _derive_bootstrap_names(self, docs: list[TopicDoc]) -> list[str]:
        selected: list[str] = ["Blog"]
        if any(self._is_comparison_text(doc.text) for doc in docs):
            selected.append("Comparison")
        if any(self._is_how_to_text(doc.text) for doc in docs):
            selected.append("How To")
        if any(self._is_use_case_text(doc.text) for doc in docs):
            selected.append("Use Case")
        if any(self._is_commercial_text(doc.text) for doc in docs):
            selected.append("Commercial")

        defaults = [name for name, _ in self.HIGH_LEVEL_PILLARS]
        for name in defaults:
            if len(selected) >= self.MIN_BOOTSTRAP_PILLARS:
                break
            if name in selected:
                continue
            selected.append(name)

        deduped: list[str] = []
        seen: set[str] = set()
        for name in selected:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(name)
            if len(deduped) >= self.MAX_ACTIVE_PILLARS:
                break
        return deduped

    def _fallback_from_text(self, text: str) -> str | None:
        tokens = self._tokenize(text)
        if not tokens:
            return None
        return self._clean_label(" ".join(tokens[:3]).title())

    def _match_score(self, topic_tokens: list[str], pillar_tokens: list[str]) -> float:
        if not topic_tokens or not pillar_tokens:
            return 0.0
        topic_set = set(topic_tokens)
        pillar_set = set(pillar_tokens)
        intersection = len(topic_set & pillar_set)
        if intersection == 0:
            return 0.0
        union = len(topic_set | pillar_set)
        jaccard = intersection / max(union, 1)
        coverage = intersection / max(len(topic_set), 1)
        return max(0.0, min(1.0, (jaccard * 0.6) + (coverage * 0.4)))

    def _build_unique_slug(self, label: str, existing: set[str]) -> str:
        base = self._slugify(label) or "pillar"
        slug = base
        suffix = 2
        while slug in existing:
            slug = f"{base}-{suffix}"
            suffix += 1
        return slug

    def _slugify(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return normalized[:80]

    def _clean_label(self, label: str) -> str:
        compact = re.sub(r"\s+", " ", str(label).strip())
        if not compact:
            return ""
        return compact[:80]

    def _high_level_pillar_description(self, name: str) -> str | None:
        canonical = self._canonical_pillar_key(name)
        if canonical is None:
            return None
        for pillar_name, pillar_description in self.HIGH_LEVEL_PILLARS:
            if self._canonical_pillar_key(pillar_name) == canonical:
                return pillar_description
        return None

    def _high_level_pillar_ids(self, pillars: list[ContentPillar]) -> dict[str, str]:
        ids: dict[str, str] = {}
        for pillar in pillars:
            canonical = self._canonical_pillar_key(pillar.name)
            if canonical is None:
                continue
            ids[canonical] = str(pillar.id)
        return ids

    def _canonical_pillar_key(self, label: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(label or "").strip().lower()).strip()
        if not normalized:
            return None
        if normalized in {"blog", "blogs"}:
            return "blog"
        if normalized in {"comparison", "comparisons"}:
            return "comparison"
        if normalized in {"how to", "howto", "how to guides", "guide", "guides"}:
            return "how_to"
        if normalized in {"use case", "use cases", "workflow", "workflows"}:
            return "use_case"
        if normalized in {"commercial", "purchase intent", "buyer intent"}:
            return "commercial"
        return None

    def _classify_doc_labels(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        labels: list[str] = []
        if self._is_comparison_text(normalized):
            labels.append("comparison")
        if self._is_how_to_text(normalized):
            labels.append("how_to")
        if self._is_commercial_text(normalized):
            labels.append("commercial")
        if self._is_use_case_text(normalized):
            labels.append("use_case")
        labels.append("blog")
        return labels

    def _is_comparison_text(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(vs|versus|compare|comparison|alternative|alternatives|replace|replacement)\b",
                text,
            )
        )

    def _is_how_to_text(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(how to|guide|tutorial|walkthrough|step by step|setup|set up|implement)\b",
                text,
            )
        )

    def _is_use_case_text(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(use case|workflow|workflows|scenario|for teams|for startups|for agencies)\b",
                text,
            )
        )

    def _is_commercial_text(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(pricing|price|cost|review|reviews|best|top|buy|purchase|demo)\b",
                text,
            )
        )

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return [
            token
            for token in tokens
            if len(token) > 2 and token not in _STOPWORDS
        ]
