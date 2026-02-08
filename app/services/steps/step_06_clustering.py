"""Step 6: Clustering into Topics.

Groups keywords into topic clusters using two-stage clustering:
1. Coarse clustering by head term + intent
2. Refine with HDBSCAN embeddings
"""

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.cluster_agent import ClusterAgent, ClusterAgentInput
from app.integrations.embeddings import EmbeddingsClient
from app.models.brand import BrandProfile
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.steps.base_step import BaseStepService, StepResult


@dataclass
class ClusteringInput:
    """Input for Step 6."""

    project_id: str


@dataclass
class ClusteringOutput:
    """Output from Step 6."""

    clusters_created: int
    unclustered_count: int
    clusters: list[dict[str, Any]] = field(default_factory=list)
    unclustered_keywords: list[str] = field(default_factory=list)


class Step06ClusteringService(BaseStepService[ClusteringInput, ClusteringOutput]):
    """Step 6: Clustering into Topics.

    Two-stage clustering approach (avoids fixed threshold instability):

    Stage 1: Coarse clustering by head term + intent
    - Group by shared n-grams (2-3 word heads)
    - Separate by intent (don't cluster commercial + informational)

    Stage 2: Refine with HDBSCAN embeddings
    - Use HDBSCAN (density-aware, no fixed threshold)
    - Handles variable density across niches

    Constraints:
    - NEVER cluster across conflicting intents
    - Cap cluster size (max 20 keywords, then split)
    - Require primary keyword clarity
    """

    step_number = 6
    step_name = "clustering"
    is_optional = False

    MAX_CLUSTER_SIZE = 20
    MIN_CLUSTER_SIZE = 3
    LOW_COHERENCE_THRESHOLD = 0.7  # Triggers Step 8 SERP validation

    async def _validate_preconditions(self, input_data: ClusteringInput) -> None:
        """Validate Step 5 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 5:
            raise ValueError("Step 5 (Intent Labeling) must be completed first")

    async def _execute(self, input_data: ClusteringInput) -> ClusteringOutput:
        """Execute two-stage clustering."""
        # Load brand profile for context
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()
        brand_context = brand.company_name if brand else ""

        await self._update_progress(5, "Loading keywords with intent...")

        # Load all active keywords with intent
        keywords_result = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == input_data.project_id,
                Keyword.status == "active",
            )
        )
        all_keywords = list(keywords_result.scalars())

        if len(all_keywords) < self.MIN_CLUSTER_SIZE:
            return ClusteringOutput(
                clusters_created=0,
                unclustered_count=len(all_keywords),
                unclustered_keywords=[kw.keyword for kw in all_keywords],
            )

        await self._update_progress(10, "Stage 1: Coarse clustering by head term + intent...")

        # Stage 1: Coarse clustering
        coarse_clusters = self._stage1_coarse_cluster(all_keywords)

        await self._update_progress(30, f"Created {len(coarse_clusters)} coarse clusters")
        await self._update_progress(35, "Stage 2: Refining with embeddings...")

        # Stage 2: Refine with embeddings
        refined_clusters = await self._stage2_refine_with_embeddings(coarse_clusters)

        await self._update_progress(70, f"Refined to {len(refined_clusters)} clusters")
        await self._update_progress(75, "Validating and naming clusters...")

        # Validate and name clusters with LLM
        final_clusters = await self._validate_and_name_clusters(
            refined_clusters,
            brand_context,
        )

        await self._update_progress(95, "Finalizing clusters...")

        # Separate valid clusters from unclustered
        valid_clusters = [c for c in final_clusters if len(c["keywords"]) >= self.MIN_CLUSTER_SIZE]
        unclustered = []
        for c in final_clusters:
            if len(c["keywords"]) < self.MIN_CLUSTER_SIZE:
                unclustered.extend(c["keywords"])

        await self._update_progress(100, "Clustering complete")

        return ClusteringOutput(
            clusters_created=len(valid_clusters),
            unclustered_count=len(unclustered),
            clusters=valid_clusters,
            unclustered_keywords=[kw.keyword for kw in unclustered],
        )

    def _stage1_coarse_cluster(
        self,
        keywords: list[Keyword],
    ) -> dict[str, list[Keyword]]:
        """Stage 1: Coarse cluster by head term + intent.

        Groups keywords that share:
        - Same 2-3 word head term
        - Same intent category
        """
        clusters: dict[str, list[Keyword]] = defaultdict(list)

        for kw in keywords:
            # Extract head term (first 2-3 significant words)
            head_term = self._extract_head_term(kw.keyword)
            intent = kw.intent or "unknown"

            # Create cluster key combining head term + intent
            cluster_key = f"{intent}::{head_term}"
            clusters[cluster_key].append(kw)

        return dict(clusters)

    def _extract_head_term(self, keyword: str) -> str:
        """Extract the head term (main topic) from a keyword."""
        # Remove common modifiers and stopwords
        stopwords = {
            "how", "to", "what", "is", "are", "the", "a", "an", "for",
            "in", "on", "with", "and", "or", "of", "best", "top",
        }

        words = keyword.lower().split()
        significant_words = [w for w in words if w not in stopwords and len(w) > 2]

        # Return first 2-3 significant words as head term
        if len(significant_words) >= 2:
            return " ".join(significant_words[:3])
        elif significant_words:
            return significant_words[0]
        else:
            return keyword.lower()[:20]  # Fallback

    async def _stage2_refine_with_embeddings(
        self,
        coarse_clusters: dict[str, list[Keyword]],
    ) -> list[dict[str, Any]]:
        """Stage 2: Refine coarse clusters using HDBSCAN on embeddings."""
        refined_clusters: list[dict[str, Any]] = []

        async with EmbeddingsClient() as embed_client:
            for cluster_key, keywords in coarse_clusters.items():
                if len(keywords) < self.MIN_CLUSTER_SIZE:
                    # Too small to cluster, keep as single cluster
                    intent = cluster_key.split("::")[0]
                    refined_clusters.append({
                        "keywords": keywords,
                        "dominant_intent": intent,
                        "head_term": cluster_key.split("::")[-1],
                    })
                    continue

                if len(keywords) <= self.MAX_CLUSTER_SIZE:
                    # Within size limits, no need to sub-cluster
                    intent = cluster_key.split("::")[0]
                    refined_clusters.append({
                        "keywords": keywords,
                        "dominant_intent": intent,
                        "head_term": cluster_key.split("::")[-1],
                    })
                    continue

                # Large cluster - use HDBSCAN to sub-divide
                try:
                    # Get embeddings
                    kw_texts = [kw.keyword for kw in keywords]
                    embeddings = await embed_client.get_embeddings(kw_texts)

                    # Cluster with HDBSCAN
                    labels, probs = embed_client.cluster_hdbscan(
                        embeddings,
                        min_cluster_size=self.MIN_CLUSTER_SIZE,
                        min_samples=2,
                    )

                    # Group by cluster label
                    sub_clusters: dict[int, list[tuple[Keyword, float, list[float]]]] = defaultdict(list)
                    for kw, label, prob, emb in zip(keywords, labels, probs, embeddings):
                        sub_clusters[label].append((kw, prob, emb))

                    # Process each sub-cluster
                    intent = cluster_key.split("::")[0]
                    head_term = cluster_key.split("::")[-1]

                    for label, members in sub_clusters.items():
                        if label == -1:
                            # Noise points - add individually or to closest cluster
                            for kw, prob, emb in members:
                                refined_clusters.append({
                                    "keywords": [kw],
                                    "dominant_intent": intent,
                                    "head_term": head_term,
                                    "is_noise": True,
                                })
                        else:
                            cluster_keywords = [m[0] for m in members]
                            cluster_embeddings = [m[2] for m in members]

                            # Calculate coherence
                            coherence = embed_client.calculate_cluster_coherence(
                                cluster_embeddings
                            )

                            refined_clusters.append({
                                "keywords": cluster_keywords,
                                "dominant_intent": intent,
                                "head_term": head_term,
                                "coherence": coherence,
                                "embeddings": cluster_embeddings,
                            })

                except Exception:
                    # Fallback: keep as single cluster
                    intent = cluster_key.split("::")[0]
                    refined_clusters.append({
                        "keywords": keywords,
                        "dominant_intent": intent,
                        "head_term": cluster_key.split("::")[-1],
                    })

        return refined_clusters

    async def _validate_and_name_clusters(
        self,
        clusters: list[dict[str, Any]],
        brand_context: str,
    ) -> list[dict[str, Any]]:
        """Validate clusters and add names using LLM."""
        # Prepare cluster data for agent
        agent_clusters = []
        for i, cluster in enumerate(clusters):
            keywords = cluster.get("keywords", [])
            agent_clusters.append({
                "keywords": [kw.keyword for kw in keywords],
                "dominant_intent": cluster.get("dominant_intent"),
                "total_volume": sum(kw.search_volume or 0 for kw in keywords),
            })

        # Process in batches of 10 clusters
        validated = []
        batch_size = 10
        agent = ClusterAgent()

        for i in range(0, len(agent_clusters), batch_size):
            batch = agent_clusters[i:i + batch_size]

            try:
                agent_input = ClusterAgentInput(
                    clusters=batch,
                    context=brand_context,
                )
                output = await agent.run(agent_input)

                for j, validation in enumerate(output.validations):
                    cluster_idx = i + j
                    if cluster_idx < len(clusters):
                        original = clusters[cluster_idx]
                        keywords = original.get("keywords", [])

                        # Find primary keyword
                        primary_kw = next(
                            (kw for kw in keywords if kw.keyword.lower() == validation.primary_keyword.lower()),
                            keywords[0] if keywords else None
                        )

                        validated.append({
                            "name": validation.name,
                            "description": validation.description,
                            "keywords": keywords,
                            "primary_keyword": primary_kw,
                            "dominant_intent": original.get("dominant_intent"),
                            "coherence": validation.coherence_score,
                            "needs_serp_validation": validation.coherence_score < self.LOW_COHERENCE_THRESHOLD,
                            "action": validation.action,
                            "notes": validation.notes,
                            "total_volume": sum(kw.search_volume or 0 for kw in keywords),
                            "avg_difficulty": (
                                sum(kw.difficulty or 0 for kw in keywords) / len(keywords)
                                if keywords else 0
                            ),
                        })

            except Exception:
                # Fallback: create basic cluster info
                for j, cluster in enumerate(batch):
                    cluster_idx = i + j
                    if cluster_idx < len(clusters):
                        original = clusters[cluster_idx]
                        keywords = original.get("keywords", [])
                        validated.append({
                            "name": f"Cluster {cluster_idx + 1}",
                            "description": "",
                            "keywords": keywords,
                            "primary_keyword": keywords[0] if keywords else None,
                            "dominant_intent": original.get("dominant_intent"),
                            "coherence": original.get("coherence", 0.5),
                            "needs_serp_validation": True,
                            "action": "keep",
                            "notes": "Auto-generated cluster",
                            "total_volume": sum(kw.search_volume or 0 for kw in keywords),
                            "avg_difficulty": (
                                sum(kw.difficulty or 0 for kw in keywords) / len(keywords)
                                if keywords else 0
                            ),
                        })

        return validated

    async def _persist_results(self, result: ClusteringOutput) -> None:
        """Save clusters to database as Topics."""
        # Delete existing topics for this project
        existing = await self.session.execute(
            select(Topic).where(Topic.project_id == self.project_id)
        )
        for topic in existing.scalars():
            await self.session.delete(topic)

        # Create new topics
        for cluster in result.clusters:
            primary_kw = cluster.get("primary_keyword")
            topic = Topic(
                id=uuid.uuid4(),
                project_id=uuid.UUID(self.project_id),
                name=cluster["name"],
                description=cluster.get("description"),
                primary_keyword_id=primary_kw.id if primary_kw else None,
                dominant_intent=cluster.get("dominant_intent"),
                dominant_page_type=primary_kw.recommended_page_type if primary_kw else None,
                estimated_total_demand=cluster.get("total_volume", 0),
                avg_difficulty=cluster.get("avg_difficulty", 0),
                cluster_coherence=cluster.get("coherence", 0.5),
                needs_serp_validation=cluster.get("needs_serp_validation", False),
                cluster_notes=cluster.get("notes"),
            )
            self.session.add(topic)

            # Update keywords to reference this topic
            for kw in cluster.get("keywords", []):
                kw.topic_id = topic.id
                if kw == primary_kw:
                    # Mark as primary
                    pass  # Already set via primary_keyword_id

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 6

        # Set result summary
        self.set_result_summary({
            "clusters_created": result.clusters_created,
            "unclustered_count": result.unclustered_count,
            "avg_cluster_size": (
                sum(len(c.get("keywords", [])) for c in result.clusters) / result.clusters_created
                if result.clusters_created > 0 else 0
            ),
            "clusters_needing_validation": sum(
                1 for c in result.clusters if c.get("needs_serp_validation")
            ),
        })

        await self.session.commit()
