"""Semantic interlinking service for generating high-quality internal link recommendations."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.embeddings import EmbeddingsClient
from app.integrations.sitemap_fetcher import SitemapFetcher, SitemapPage
from app.models.content import ContentBrief
from app.models.topic import Topic

logger = logging.getLogger(__name__)


@dataclass
class LinkCandidate:
    """Represents a potential internal link."""

    target_type: str  # sitemap_page | batch_brief | pillar_page | money_page
    target_url: str | None
    target_brief_id: str | None
    anchor_text: str
    placement_section: str
    relevance_score: float
    intent_alignment: str
    funnel_relationship: str


@dataclass
class InterlinkingResult:
    """Result of interlinking analysis."""

    briefs_enhanced: int
    sitemap_links_added: int
    batch_links_added: int
    sitemap_pages_found: int
    low_relevance_skipped: int


class InterlinkingService:
    """Service for intelligent internal linking using semantic analysis.

    Supports:
    - Sitemap-based linking to existing content
    - Topic-based cross-linking within batch
    - Semantic relevance scoring (0.5 cosine + 0.2 intent + 0.15 funnel + 0.15 keyword)
    - Contextual anchor text generation
    """

    # Scoring weights
    WEIGHT_COSINE = 0.50
    WEIGHT_INTENT = 0.20
    WEIGHT_FUNNEL = 0.15
    WEIGHT_KEYWORD = 0.15

    # Intent compatibility scores
    INTENT_SCORES = {
        ("informational", "informational"): 1.0,
        ("commercial", "commercial"): 1.0,
        ("navigational", "navigational"): 1.0,
        ("transactional", "transactional"): 1.0,
        # Complementary intents
        ("informational", "commercial"): 0.8,
        ("commercial", "transactional"): 0.8,
        ("informational", "navigational"): 0.7,
        # Weak relationships
        ("commercial", "navigational"): 0.5,
        ("informational", "transactional"): 0.5,
        ("navigational", "transactional"): 0.3,
    }

    # Funnel stage compatibility (source -> target)
    FUNNEL_SCORES = {
        # Natural progression (TOFU -> MOFU -> BOFU)
        ("tofu", "mofu"): 1.0,
        ("mofu", "bofu"): 1.0,
        ("tofu", "bofu"): 0.9,
        # Same stage cross-links
        ("tofu", "tofu"): 0.8,
        ("mofu", "mofu"): 0.8,
        ("bofu", "bofu"): 0.8,
        # Reverse flow (less desirable but not forbidden)
        ("mofu", "tofu"): 0.5,
        ("bofu", "mofu"): 0.5,
        ("bofu", "tofu"): 0.4,
    }

    def __init__(
        self,
        session: AsyncSession,
        embeddings_client: EmbeddingsClient,
        min_relevance_score: float = 0.65,
        max_links_per_brief: int = 8,
        max_sitemap_links: int = 5,
        max_batch_links: int = 3,
    ) -> None:
        """Initialize interlinking service.

        Args:
            session: Database session
            embeddings_client: Client for generating embeddings
            min_relevance_score: Minimum score to include a link (0.0-1.0)
            max_links_per_brief: Maximum total links to add per brief
            max_sitemap_links: Maximum links to sitemap pages per brief
            max_batch_links: Maximum links to other briefs in batch per brief
        """
        self.session = session
        self.embeddings_client = embeddings_client
        self.min_relevance_score = min_relevance_score
        self.max_links_per_brief = max_links_per_brief
        self.max_sitemap_links = max_sitemap_links
        self.max_batch_links = max_batch_links

    async def analyze_and_enrich_links(
        self,
        briefs: list[ContentBrief],
        topics: dict[str, Topic],
        sitemap_pages: list[SitemapPage] | None = None,
    ) -> InterlinkingResult:
        """Analyze briefs and generate enhanced internal link recommendations.

        Args:
            briefs: List of content briefs to enrich
            topics: Dictionary mapping brief.topic_id to Topic objects
            sitemap_pages: Optional list of sitemap pages to link to

        Returns:
            InterlinkingResult with statistics
        """
        if not briefs:
            logger.info("No briefs to process for interlinking")
            return InterlinkingResult(
                briefs_enhanced=0,
                sitemap_links_added=0,
                batch_links_added=0,
                sitemap_pages_found=0,
                low_relevance_skipped=0,
            )

        logger.info(
            "Starting interlinking analysis",
            extra={
                "briefs_count": len(briefs),
                "sitemap_pages_count": len(sitemap_pages) if sitemap_pages else 0,
            },
        )

        # Generate embeddings for briefs
        brief_texts = [self._brief_to_text(brief) for brief in briefs]
        brief_embeddings = await self.embeddings_client.get_embeddings(brief_texts)

        # Generate embeddings for sitemap pages if available
        sitemap_embeddings = []
        if sitemap_pages:
            sitemap_texts = [self._sitemap_page_to_text(page) for page in sitemap_pages]
            sitemap_embeddings = await self.embeddings_client.get_embeddings(sitemap_texts)

        # Convert to numpy arrays for efficient computation
        brief_emb_array = np.array(brief_embeddings)
        sitemap_emb_array = np.array(sitemap_embeddings) if sitemap_embeddings else np.array([])

        # Statistics
        briefs_enhanced = 0
        sitemap_links_added = 0
        batch_links_added = 0
        low_relevance_skipped = 0

        # Process each brief
        for i, brief in enumerate(briefs):
            topic = topics.get(brief.topic_id)
            if not topic:
                logger.warning(
                    "Topic not found for brief",
                    extra={"brief_id": str(brief.id), "topic_id": brief.topic_id},
                )
                continue

            # Get current links (preserve existing pillar/money page links)
            existing_links = brief.internal_links_out or []

            # Generate new link candidates
            candidates = []

            # 1. Add sitemap-based links
            if sitemap_pages and len(sitemap_emb_array) > 0:
                sitemap_candidates = self._generate_sitemap_links(
                    brief=brief,
                    topic=topic,
                    brief_embedding=brief_emb_array[i],
                    sitemap_pages=sitemap_pages,
                    sitemap_embeddings=sitemap_emb_array,
                )
                candidates.extend(sitemap_candidates)

            # 2. Add batch cross-links
            batch_candidates = self._generate_batch_links(
                source_brief=brief,
                source_topic=topic,
                source_embedding=brief_emb_array[i],
                all_briefs=briefs,
                all_topics=topics,
                all_embeddings=brief_emb_array,
            )
            candidates.extend(batch_candidates)

            # Filter by minimum relevance score
            qualified_candidates = [
                c for c in candidates if c.relevance_score >= self.min_relevance_score
            ]
            low_relevance_skipped += len(candidates) - len(qualified_candidates)

            # Sort by relevance score (highest first)
            qualified_candidates.sort(key=lambda c: c.relevance_score, reverse=True)

            # Limit links per category
            sitemap_links = [c for c in qualified_candidates if c.target_type == "sitemap_page"][
                :self.max_sitemap_links
            ]
            batch_links = [c for c in qualified_candidates if c.target_type == "batch_brief"][
                :self.max_batch_links
            ]

            # Combine and limit total
            new_links = (sitemap_links + batch_links)[:self.max_links_per_brief]

            # Convert to dict format for storage
            new_link_dicts = [self._candidate_to_dict(c) for c in new_links]

            # Update brief's internal_links_out (preserve existing + add new)
            brief.internal_links_out = existing_links + new_link_dicts

            # Update bi-directional links (internal_links_in) for target briefs
            for link in new_links:
                if link.target_type == "batch_brief" and link.target_brief_id:
                    self._add_incoming_link(
                        target_brief_id=link.target_brief_id,
                        source_brief=brief,
                        anchor_text=link.anchor_text,
                        briefs=briefs,
                    )

            # Update statistics
            if new_links:
                briefs_enhanced += 1
                sitemap_links_added += len(sitemap_links)
                batch_links_added += len(batch_links)

        result = InterlinkingResult(
            briefs_enhanced=briefs_enhanced,
            sitemap_links_added=sitemap_links_added,
            batch_links_added=batch_links_added,
            sitemap_pages_found=len(sitemap_pages) if sitemap_pages else 0,
            low_relevance_skipped=low_relevance_skipped,
        )

        logger.info(
            "Interlinking analysis complete",
            extra={
                "briefs_enhanced": result.briefs_enhanced,
                "sitemap_links": result.sitemap_links_added,
                "batch_links": result.batch_links_added,
                "low_relevance_skipped": result.low_relevance_skipped,
            },
        )

        return result

    def _generate_sitemap_links(
        self,
        brief: ContentBrief,
        topic: Topic,
        brief_embedding: np.ndarray,
        sitemap_pages: list[SitemapPage],
        sitemap_embeddings: np.ndarray,
    ) -> list[LinkCandidate]:
        """Generate link candidates from sitemap pages.

        Args:
            brief: Source brief
            topic: Source topic
            brief_embedding: Embedding vector for the brief
            sitemap_pages: List of sitemap pages
            sitemap_embeddings: Embedding vectors for sitemap pages

        Returns:
            List of link candidates
        """
        candidates = []

        # Calculate cosine similarities
        cosine_scores = self._cosine_similarity_batch(brief_embedding, sitemap_embeddings)

        for i, (page, cosine_score) in enumerate(zip(sitemap_pages, cosine_scores)):
            # Early exit if cosine similarity too low
            if cosine_score < 0.5:
                continue

            # For sitemap pages, we don't have intent/funnel data
            # Use default moderate scores
            intent_score = 0.7  # Assume complementary intent
            funnel_score = 0.7  # Assume reasonable funnel progression

            # Keyword overlap score (based on URL and brief keywords)
            keyword_score = self._calculate_keyword_overlap_from_url(
                url=page.url,
                brief=brief,
            )

            # Calculate composite relevance score
            relevance_score = (
                self.WEIGHT_COSINE * cosine_score
                + self.WEIGHT_INTENT * intent_score
                + self.WEIGHT_FUNNEL * funnel_score
                + self.WEIGHT_KEYWORD * keyword_score
            )

            # Generate anchor text from URL
            anchor_text = self._generate_anchor_text(
                target_url=page.url,
                source_brief=brief,
            )

            # Select placement section from outline
            placement_section = self._select_placement_section(brief)

            candidates.append(
                LinkCandidate(
                    target_type="sitemap_page",
                    target_url=page.url,
                    target_brief_id=None,
                    anchor_text=anchor_text,
                    placement_section=placement_section,
                    relevance_score=relevance_score,
                    intent_alignment="assumed_complementary",
                    funnel_relationship="assumed_progression",
                )
            )

        return candidates

    def _generate_batch_links(
        self,
        source_brief: ContentBrief,
        source_topic: Topic,
        source_embedding: np.ndarray,
        all_briefs: list[ContentBrief],
        all_topics: dict[str, Topic],
        all_embeddings: np.ndarray,
    ) -> list[LinkCandidate]:
        """Generate link candidates to other briefs in the batch.

        Args:
            source_brief: Source brief
            source_topic: Source topic
            source_embedding: Embedding vector for source brief
            all_briefs: All briefs in batch
            all_topics: Dictionary of all topics
            all_embeddings: Embedding vectors for all briefs

        Returns:
            List of link candidates
        """
        candidates = []

        # Calculate cosine similarities
        cosine_scores = self._cosine_similarity_batch(source_embedding, all_embeddings)

        for i, (target_brief, cosine_score) in enumerate(zip(all_briefs, cosine_scores)):
            # Skip self-links
            if target_brief.id == source_brief.id:
                continue

            # Only link to content scheduled before this brief.
            if not self._is_valid_publication_predecessor(source_brief, target_brief):
                continue

            # Early exit if cosine similarity too low
            if cosine_score < 0.5:
                continue

            target_topic = all_topics.get(target_brief.topic_id)
            if not target_topic:
                continue

            # Check for cannibalization conflicts
            if self._is_cannibalization_conflict(source_topic, target_topic):
                logger.debug(
                    "Skipping link due to cannibalization conflict",
                    extra={
                        "source_topic": source_topic.name,
                        "target_topic": target_topic.name,
                    },
                )
                continue

            # Calculate intent alignment score
            intent_score, intent_label = self._calculate_intent_alignment(
                source_intent=source_topic.dominant_intent,
                target_intent=target_topic.dominant_intent,
            )

            # Calculate funnel stage compatibility
            funnel_score, funnel_label = self._calculate_funnel_compatibility(
                source_funnel=source_topic.funnel_stage,
                target_funnel=target_topic.funnel_stage,
            )

            # Calculate keyword overlap
            keyword_score = self._calculate_keyword_overlap(
                source_brief=source_brief,
                target_brief=target_brief,
            )

            # Calculate composite relevance score
            relevance_score = (
                self.WEIGHT_COSINE * cosine_score
                + self.WEIGHT_INTENT * intent_score
                + self.WEIGHT_FUNNEL * funnel_score
                + self.WEIGHT_KEYWORD * keyword_score
            )

            # Generate anchor text
            anchor_text = self._generate_anchor_text(
                target_keyword=target_brief.primary_keyword,
                target_page_type=target_topic.dominant_page_type,
                source_brief=source_brief,
            )

            # Select placement section
            placement_section = self._select_placement_section(source_brief)

            candidates.append(
                LinkCandidate(
                    target_type="batch_brief",
                    target_url=None,  # Resolved later when/if target gets published_url
                    target_brief_id=str(target_brief.id),
                    anchor_text=anchor_text,
                    placement_section=placement_section,
                    relevance_score=relevance_score,
                    intent_alignment=intent_label,
                    funnel_relationship=funnel_label,
                )
            )

        return candidates

    def _is_valid_publication_predecessor(
        self,
        source_brief: ContentBrief,
        target_brief: ContentBrief,
    ) -> bool:
        """Return True when target is scheduled before source.

        If source has a proposed date and target does not, skip the target.
        If source lacks a proposed date, keep legacy behavior and allow target.
        """
        source_date = source_brief.proposed_publication_date
        target_date = target_brief.proposed_publication_date

        if source_date is None:
            return True
        if target_date is None:
            return False
        return target_date < source_date

    def _cosine_similarity_batch(
        self,
        source_embedding: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarity between source and multiple targets.

        Args:
            source_embedding: Single embedding vector
            target_embeddings: Array of embedding vectors

        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        source_norm = source_embedding / np.linalg.norm(source_embedding)
        target_norms = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )

        # Calculate dot products (cosine similarity)
        similarities = np.dot(target_norms, source_norm)

        return similarities

    def _calculate_intent_alignment(
        self,
        source_intent: str | None,
        target_intent: str | None,
    ) -> tuple[float, str]:
        """Calculate intent alignment score and label.

        Args:
            source_intent: Source topic intent
            target_intent: Target topic intent

        Returns:
            Tuple of (score, label)
        """
        if not source_intent or not target_intent:
            return 0.7, "unknown"

        source = source_intent.lower()
        target = target_intent.lower()

        # Check direct match
        score = self.INTENT_SCORES.get((source, target))
        if score is not None:
            if source == target:
                return score, "same"
            elif score >= 0.7:
                return score, "complementary"
            else:
                return score, "weak"

        # Try reverse order
        score = self.INTENT_SCORES.get((target, source))
        if score is not None:
            if score >= 0.7:
                return score, "complementary"
            else:
                return score, "weak"

        # Default for unknown combinations
        return 0.5, "unknown"

    def _calculate_funnel_compatibility(
        self,
        source_funnel: str | None,
        target_funnel: str | None,
    ) -> tuple[float, str]:
        """Calculate funnel stage compatibility score and label.

        Args:
            source_funnel: Source topic funnel stage
            target_funnel: Target topic funnel stage

        Returns:
            Tuple of (score, label)
        """
        if not source_funnel or not target_funnel:
            return 0.7, "unknown"

        source = source_funnel.lower()
        target = target_funnel.lower()

        score = self.FUNNEL_SCORES.get((source, target))
        if score is not None:
            if source == target:
                return score, "same_stage"
            elif score >= 0.9:
                return score, "natural_progression"
            elif score >= 0.5:
                return score, "reverse_flow"
            else:
                return score, "weak_flow"

        # Default for unknown combinations
        return 0.7, "unknown"

    def _calculate_keyword_overlap(
        self,
        source_brief: ContentBrief,
        target_brief: ContentBrief,
    ) -> float:
        """Calculate keyword overlap using Jaccard similarity.

        Args:
            source_brief: Source brief
            target_brief: Target brief

        Returns:
            Jaccard similarity score (0.0-1.0)
        """
        # Get supporting keywords
        source_keywords = set(source_brief.supporting_keywords or [])
        target_keywords = set(target_brief.supporting_keywords or [])

        # Add primary keywords
        if source_brief.primary_keyword:
            source_keywords.add(source_brief.primary_keyword.lower())
        if target_brief.primary_keyword:
            target_keywords.add(target_brief.primary_keyword.lower())

        if not source_keywords or not target_keywords:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(source_keywords & target_keywords)
        union = len(source_keywords | target_keywords)

        return intersection / union if union > 0 else 0.0

    def _calculate_keyword_overlap_from_url(
        self,
        url: str,
        brief: ContentBrief,
    ) -> float:
        """Calculate keyword overlap between URL and brief keywords.

        Args:
            url: Target URL
            brief: Source brief

        Returns:
            Overlap score (0.0-1.0)
        """
        # Extract words from URL path
        url_words = set()
        url_path = url.split("?")[0]  # Remove query params
        # Extract words (alphanumeric sequences)
        words = re.findall(r"[a-z0-9]+", url_path.lower())
        url_words.update(words)

        # Get brief keywords
        brief_keywords = set()
        if brief.primary_keyword:
            brief_keywords.update(brief.primary_keyword.lower().split())
        if brief.supporting_keywords:
            for kw in brief.supporting_keywords:
                brief_keywords.update(kw.lower().split())

        if not url_words or not brief_keywords:
            return 0.0

        # Calculate overlap
        intersection = len(url_words & brief_keywords)
        union = len(url_words | brief_keywords)

        return intersection / union if union > 0 else 0.0

    def _is_cannibalization_conflict(
        self,
        source_topic: Topic,
        target_topic: Topic,
    ) -> bool:
        """Check if topics have cannibalization conflict.

        Args:
            source_topic: Source topic
            target_topic: Target topic

        Returns:
            True if conflict exists
        """
        # Check if target is in source's overlapping topics
        if source_topic.overlapping_topic_ids:
            if str(target_topic.id) in source_topic.overlapping_topic_ids:
                return True

        # Check reverse
        if target_topic.overlapping_topic_ids:
            if str(source_topic.id) in target_topic.overlapping_topic_ids:
                return True

        return False

    def _generate_anchor_text(
        self,
        target_keyword: str | None = None,
        target_page_type: str | None = None,
        target_url: str | None = None,
        source_brief: ContentBrief | None = None,
    ) -> str:
        """Generate natural anchor text for a link.

        Args:
            target_keyword: Primary keyword of target
            target_page_type: Page type of target
            target_url: URL of target (for sitemap links)
            source_brief: Source brief (for context)

        Returns:
            Natural anchor text (5-8 words)
        """
        # If we have a target keyword, use it
        if target_keyword:
            keyword_clean = target_keyword.lower().strip()

            # Template-based generation
            templates = [
                f"learn more about {keyword_clean}",
                f"explore {keyword_clean}",
                f"{keyword_clean} best practices",
                f"guide to {keyword_clean}",
                f"everything about {keyword_clean}",
            ]

            # Pick template based on page type
            if target_page_type:
                page_type = target_page_type.lower()
                if "guide" in page_type or "tutorial" in page_type:
                    return f"complete guide to {keyword_clean}"
                elif "comparison" in page_type or "versus" in page_type:
                    return f"compare {keyword_clean} options"
                elif "list" in page_type or "listicle" in page_type:
                    return f"top {keyword_clean} solutions"

            return templates[0]

        # If we have a URL, extract topic from it
        if target_url:
            # Extract last path segment
            path = target_url.split("?")[0].rstrip("/")
            slug = path.split("/")[-1]
            # Convert dashes/underscores to spaces
            topic = re.sub(r"[-_]", " ", slug)
            topic = re.sub(r"\s+", " ", topic).strip()

            if topic:
                return f"learn about {topic}"

        # Fallback
        return "explore related content"

    def _select_placement_section(self, brief: ContentBrief) -> str:
        """Select appropriate section from outline for link placement.

        Args:
            brief: Content brief

        Returns:
            Section heading or default placement
        """
        outline = brief.outline
        if not outline or not isinstance(outline, dict):
            return "In relevant section"

        sections = outline.get("sections", [])
        if not sections:
            return "In relevant section"

        # Prefer middle sections (avoid intro/conclusion)
        if len(sections) >= 3:
            middle_section = sections[len(sections) // 2]
            if isinstance(middle_section, dict) and "heading" in middle_section:
                return middle_section["heading"]

        # Use first section
        if sections and isinstance(sections[0], dict) and "heading" in sections[0]:
            return sections[0]["heading"]

        return "In relevant section"

    def _brief_to_text(self, brief: ContentBrief) -> str:
        """Convert brief to text for embedding generation.

        Args:
            brief: Content brief

        Returns:
            Text representation
        """
        parts = []

        if brief.primary_keyword:
            parts.append(brief.primary_keyword)

        if brief.working_titles:
            parts.extend(brief.working_titles[:2])  # First 2 titles

        if brief.target_audience:
            parts.append(brief.target_audience)

        if brief.reader_job_to_be_done:
            parts.append(brief.reader_job_to_be_done)

        # Add outline headings
        if brief.outline and isinstance(brief.outline, dict):
            sections = brief.outline.get("sections", [])
            for section in sections[:5]:  # First 5 sections
                if isinstance(section, dict) and "heading" in section:
                    parts.append(section["heading"])

        return " ".join(parts)

    def _sitemap_page_to_text(self, page: SitemapPage) -> str:
        """Convert sitemap page to text for embedding generation.

        Args:
            page: Sitemap page

        Returns:
            Text representation
        """
        # Extract topic from URL
        path = page.url.split("?")[0].rstrip("/")
        slug = path.split("/")[-1]
        # Convert dashes/underscores to spaces
        topic = re.sub(r"[-_]", " ", slug)
        topic = re.sub(r"\s+", " ", topic).strip()

        return topic

    def _candidate_to_dict(self, candidate: LinkCandidate) -> dict:
        """Convert LinkCandidate to dictionary for storage.

        Args:
            candidate: Link candidate

        Returns:
            Dictionary representation
        """
        return {
            "target_type": candidate.target_type,
            "target_url": candidate.target_url,
            "target_brief_id": candidate.target_brief_id,
            "anchor_text": candidate.anchor_text,
            "placement_section": candidate.placement_section,
            "relevance_score": round(candidate.relevance_score, 3),
            "intent_alignment": candidate.intent_alignment,
            "funnel_relationship": candidate.funnel_relationship,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _add_incoming_link(
        self,
        target_brief_id: str,
        source_brief: ContentBrief,
        anchor_text: str,
        briefs: list[ContentBrief],
    ) -> None:
        """Add incoming link to target brief's internal_links_in.

        Args:
            target_brief_id: ID of target brief
            source_brief: Source brief
            anchor_text: Anchor text used
            briefs: All briefs (to find target)
        """
        # Find target brief
        target_brief = next(
            (b for b in briefs if str(b.id) == target_brief_id),
            None,
        )

        if not target_brief:
            return

        # Initialize internal_links_in if needed
        if not target_brief.internal_links_in:
            target_brief.internal_links_in = []

        # Add incoming link
        target_brief.internal_links_in.append({
            "source_type": "batch_brief",
            "source_url": None,  # Will be determined when published
            "source_brief_id": str(source_brief.id),
            "anchor_text": anchor_text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
