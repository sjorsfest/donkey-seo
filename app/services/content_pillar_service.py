"""Content pillar taxonomy + assignment service."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content_pillar import ContentBriefPillarAssignment, ContentPillar
from app.models.generated_dtos import (
    ContentBriefPillarAssignmentCreateDTO,
    ContentPillarCreateDTO,
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

    MAX_ACTIVE_PILLARS = 7
    MIN_BOOTSTRAP_PILLARS = 3
    MIN_PRIMARY_MATCH_SCORE = 0.18
    SECONDARY_MATCH_SCORE = 0.15
    SECONDARY_DELTA = 0.08
    STRICT_NEW_PILLAR_MIN_UNCOVERED = 4

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

        if not pillars:
            # Last-resort single pillar keeps downstream assignment deterministic.
            fallback_name = self._fallback_pillar_name(corpus_docs)
            pillars = await self._create_pillars(
                project_id=project_id,
                names=[fallback_name],
                source="auto",
            )

        pillars = await self._strictly_expand_taxonomy(
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

    async def _strictly_expand_taxonomy(
        self,
        *,
        project_id: str,
        pillars: list[ContentPillar],
        docs: list[TopicDoc],
    ) -> list[ContentPillar]:
        """Add new pillars only when unmatched topic evidence is strong."""
        current = list(pillars)

        while len(current) < self.MAX_ACTIVE_PILLARS:
            uncovered = self._collect_uncovered_docs(docs, current)
            if len(uncovered) < self.STRICT_NEW_PILLAR_MIN_UNCOVERED:
                break

            candidate_name = self._derive_single_name(uncovered)
            if not candidate_name:
                break

            normalized_candidate = candidate_name.strip().lower()
            existing_names = {pillar.name.strip().lower() for pillar in current}
            if normalized_candidate in existing_names:
                break

            created = await self._create_pillars(
                project_id=project_id,
                names=[candidate_name],
                source="auto",
            )
            if not created:
                break
            current.extend(created)

        return current

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
            created.append(
                ContentPillar.create(
                    self.session,
                    ContentPillarCreateDTO(
                        project_id=project_id,
                        name=clean_name,
                        slug=slug,
                        description=None,
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
        target_count = min(
            self.MAX_ACTIVE_PILLARS,
            max(self.MIN_BOOTSTRAP_PILLARS, min(len(docs), 5)),
        )

        token_counts: Counter[str] = Counter()
        for doc in docs:
            token_counts.update(set(doc.tokens))

        names: list[str] = []
        seen: set[str] = set()

        for token, count in token_counts.most_common():
            if count < 2:
                continue
            label = self._clean_label(token.title())
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            names.append(label)
            seen.add(key)
            if len(names) >= target_count:
                return names

        for doc in docs:
            label = self._fallback_from_text(doc.text)
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            names.append(label)
            seen.add(key)
            if len(names) >= target_count:
                break

        if not names:
            names.append(self._fallback_pillar_name(docs))

        return names[:target_count]

    def _derive_single_name(self, docs: list[TopicDoc]) -> str | None:
        token_counts: Counter[str] = Counter()
        for doc in docs:
            token_counts.update(set(doc.tokens))
        for token, count in token_counts.most_common():
            if count < 2:
                continue
            label = self._clean_label(token.title())
            if label:
                return label
        return self._fallback_from_text(docs[0].text) if docs else None

    def _fallback_pillar_name(self, docs: list[TopicDoc]) -> str:
        if not docs:
            return "General"
        fallback = self._fallback_from_text(docs[0].text)
        return fallback or "General"

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

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return [
            token
            for token in tokens
            if len(token) > 2 and token not in _STOPWORDS
        ]
