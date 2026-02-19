"""Step 6: Clustering into Topics.

Groups keywords into topic clusters using two-stage clustering:
1. Coarse clustering by head term + intent
2. Refine with HDBSCAN embeddings
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sqlalchemy import select

from app.agents.cluster_agent import ClusterAgent, ClusterAgentInput
from app.integrations.embeddings import EmbeddingsClient
from app.models.brand import BrandProfile
from app.models.generated_dtos import TopicCreateDTO
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.discovery_capabilities import CAPABILITY_CLUSTERING
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


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
    capability_key = CAPABILITY_CLUSTERING
    is_optional = False

    MAX_CLUSTER_SIZE = 20
    MIN_CLUSTER_SIZE = 3
    LOW_COHERENCE_THRESHOLD = 0.7  # Triggers Step 8 SERP validation
    NOISE_SIMILARITY_THRESHOLD = 0.3  # Min cosine sim to assign noise to nearest cluster

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
        strategy = await self.get_run_strategy()
        brand_context = self._build_clustering_context(brand, strategy)
        learning_context = await self.build_learning_context(
            self.capability_key,
            "ClusterAgent",
        )
        if learning_context:
            brand_context = (
                f"{brand_context}\n\n{learning_context}"
                if brand_context
                else learning_context
            )
        market_mode = await self.get_market_mode(default="mixed")
        workflowish_mode = market_mode in {"fragmented_workflow", "mixed"}

        await self._update_progress(5, "Loading keywords with intent...")

        # Load all active keywords with intent
        keywords_result = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == input_data.project_id,
                Keyword.status == "active",
            )
        )
        all_keywords = list(keywords_result.scalars())
        logger.info(
            "Clustering starting",
            extra={
                "project_id": input_data.project_id,
                "keyword_count": len(all_keywords),
            },
        )

        if len(all_keywords) < self.MIN_CLUSTER_SIZE:
            return ClusteringOutput(
                clusters_created=0,
                unclustered_count=len(all_keywords),
                unclustered_keywords=[kw.keyword for kw in all_keywords],
            )

        # Check for checkpoint to skip expensive stages
        if self._checkpoint and self._checkpoint.get("stage") == "stage2_complete":
            logger.info("Restoring refined clusters from checkpoint")
            refined_clusters = self._deserialize_clusters(
                self._checkpoint["refined_clusters"],
                all_keywords,
            )
            await self._update_progress(70, f"Restored {len(refined_clusters)} clusters from checkpoint")
        else:
            await self._update_progress(10, "Stage 1: Coarse clustering by head term + intent...")

            # Stage 1: Coarse clustering
            coarse_clusters = self._stage1_coarse_cluster(
                all_keywords,
                workflowish_mode=workflowish_mode,
            )
            logger.info("Coarse clustering done", extra={"cluster_count": len(coarse_clusters)})

            await self._update_progress(30, f"Created {len(coarse_clusters)} coarse clusters")
            await self._update_progress(35, "Stage 2: Refining with embeddings...")

            # Stage 2: Refine with embeddings
            refined_clusters = await self._stage2_refine_with_embeddings(coarse_clusters)
            logger.info("Refinement done", extra={"refined_count": len(refined_clusters)})

            await self._update_progress(65, f"Refined to {len(refined_clusters)} clusters")

            # Merge undersized clusters so they don't all get filtered out
            refined_clusters = self._merge_small_clusters(refined_clusters)
            logger.info("After merging small clusters", extra={"cluster_count": len(refined_clusters)})

            await self._update_progress(70, f"Merged to {len(refined_clusters)} clusters")

            # Checkpoint after Stage 2 (embeddings are the expensive part)
            await self._save_checkpoint({
                "stage": "stage2_complete",
                "refined_clusters": self._serialize_clusters(refined_clusters),
            })

        await self._update_progress(75, "Validating and naming clusters...")

            # Validate and name clusters with LLM
        final_clusters = await self._validate_and_name_clusters(
            refined_clusters,
            brand_context,
            workflowish_mode=workflowish_mode,
        )

        await self._update_progress(90, "Processing split/merge recommendations...")

        # Act on agent split/merge recommendations
        final_clusters = await self._process_agent_actions(final_clusters)

        await self._update_progress(95, "Finalizing clusters...")

        # Separate valid clusters from unclustered
        valid_clusters = [c for c in final_clusters if len(c["keywords"]) >= self.MIN_CLUSTER_SIZE]
        unclustered = []
        for c in final_clusters:
            if len(c["keywords"]) < self.MIN_CLUSTER_SIZE:
                unclustered.extend(c["keywords"])
        logger.info(
            "Cluster validation done",
            extra={
                "valid_clusters": len(valid_clusters),
                "unclustered": len(unclustered),
            },
        )

        await self._update_progress(100, "Clustering complete")

        return ClusteringOutput(
            clusters_created=len(valid_clusters),
            unclustered_count=len(unclustered),
            clusters=valid_clusters,
            unclustered_keywords=[kw.keyword for kw in unclustered],
        )

    def _build_clustering_context(self, brand: BrandProfile | None, strategy: Any) -> str:
        """Build richer context so cluster naming stays aligned with project strategy."""
        parts: list[str] = []
        if brand and brand.company_name:
            parts.append(f"Company: {brand.company_name}")
        if brand and brand.products_services:
            products = [p.get("name", "") for p in brand.products_services[:5] if p.get("name")]
            if products:
                parts.append(f"Products/Services: {', '.join(products)}")
        if strategy.icp_roles:
            parts.append(f"ICP Roles: {', '.join(strategy.icp_roles[:8])}")
        if strategy.icp_industries:
            parts.append(f"ICP Industries: {', '.join(strategy.icp_industries[:8])}")
        if strategy.icp_pains:
            parts.append(f"ICP Pains: {', '.join(strategy.icp_pains[:8])}")
        if strategy.include_topics:
            parts.append(f"In Scope Topics: {', '.join(strategy.include_topics[:12])}")
        if strategy.exclude_topics:
            parts.append(f"Out of Scope Topics: {', '.join(strategy.exclude_topics[:12])}")
        if strategy.conversion_intents:
            parts.append(f"Conversion Intents: {', '.join(strategy.conversion_intents[:8])}")
        return "\n".join(parts)

    async def _validate_output(self, result: ClusteringOutput, input_data: ClusteringInput) -> None:
        """Ensure output can be consumed by Step 7."""
        if result.clusters_created <= 0:
            raise ValueError(
                "Step 6 produced 0 valid clusters. Step 7 requires at least one cluster/topic."
            )

    def _stage1_coarse_cluster(
        self,
        keywords: list[Keyword],
        *,
        workflowish_mode: bool,
    ) -> dict[str, list[Keyword]]:
        """Stage 1: Coarse cluster by head term + intent.

        Groups keywords that share:
        - Same 3-4 word head term
        - Same intent category
        """
        clusters: dict[str, list[Keyword]] = defaultdict(list)

        for kw in keywords:
            # Extract head term (first 2-3 significant words)
            head_term = self._extract_head_term(kw.keyword)
            intent = kw.intent or "unknown"
            if workflowish_mode:
                signal_parts = self._workflow_cluster_signal_parts(kw)
                cluster_key = (
                    f"{intent}::{head_term}"
                    f"::pair={signal_parts['entity_pair']}"
                    f"::verb={signal_parts['workflow_verb']}"
                    f"::cmp={signal_parts['comparison_target']}"
                    f"::noun={signal_parts['core_noun_phrase']}"
                )
            else:
                # Create cluster key combining head term + intent
                cluster_key = f"{intent}::{head_term}"
            clusters[cluster_key].append(kw)

        return dict(clusters)

    def _extract_head_term(self, keyword: str) -> str:
        """Extract the head term (main topic) from a keyword.

        Uses 3-4 significant words and preserves intent-carrying modifiers
        like 'best' and 'top' which distinguish commercial from informational.
        """
        # True non-semantic words only -- NOT intent signals like best/top
        stopwords = {
            "how", "to", "what", "is", "are", "the", "a", "an", "for",
            "in", "on", "with", "and", "or", "of", "do", "does", "can",
            "should", "will", "would", "much", "many", "get", "vs",
            "versus", "you", "your", "their", "its", "this", "that",
            "which", "where", "when", "why", "from", "about", "into",
        }

        words = keyword.lower().split()
        significant_words = [w for w in words if w not in stopwords and len(w) > 1]

        # Use 3-4 words to reduce false groupings
        if len(significant_words) >= 3:
            return " ".join(significant_words[:4])
        elif significant_words:
            return " ".join(significant_words)
        else:
            return keyword.lower()[:30]  # Fallback

    def _workflow_cluster_signal_parts(self, keyword_model: Keyword) -> dict[str, str]:
        """Extract signal partitions used by workflow-ish clustering constraints."""
        signals = (
            keyword_model.discovery_signals
            if isinstance(keyword_model.discovery_signals, dict)
            else {}
        )

        matched_entities = sorted(
            str(entity).strip().lower()
            for entity in (signals.get("matched_entities") or [])
            if str(entity).strip()
        )
        entity_pair = "none"
        if len(matched_entities) >= 2:
            entity_pair = "|".join(sorted(matched_entities[:2]))

        workflow_verb = str(signals.get("workflow_verb") or "none").strip().lower() or "none"
        comparison_target = str(signals.get("comparison_target") or "none").strip().lower() or "none"
        core_noun_phrase = str(signals.get("core_noun_phrase") or "none").strip().lower() or "none"

        return {
            "entity_pair": entity_pair,
            "workflow_verb": workflow_verb,
            "comparison_target": comparison_target,
            "core_noun_phrase": core_noun_phrase,
        }

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
                    # Right-sized: compute embeddings + coherence but don't sub-cluster
                    intent = cluster_key.split("::")[0]
                    head_term = cluster_key.split("::")[-1]
                    try:
                        kw_texts = [kw.keyword for kw in keywords]
                        embeddings = await embed_client.get_embeddings(kw_texts)
                        coherence = embed_client.calculate_cluster_coherence(embeddings)
                        refined_clusters.append({
                            "keywords": keywords,
                            "dominant_intent": intent,
                            "head_term": head_term,
                            "coherence": coherence,
                            "embeddings": embeddings,
                        })
                    except Exception:
                        logger.warning(
                            "Embedding coherence failed for right-sized cluster",
                            extra={"cluster_key": cluster_key, "keyword_count": len(keywords)},
                        )
                        refined_clusters.append({
                            "keywords": keywords,
                            "dominant_intent": intent,
                            "head_term": head_term,
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
                    sub_clusters: dict[
                        int,
                        list[tuple[Keyword, float, list[float]]],
                    ] = defaultdict(list)
                    for kw, label, prob, emb in zip(keywords, labels, probs, embeddings):
                        sub_clusters[label].append((kw, prob, emb))

                    # Process each sub-cluster
                    intent = cluster_key.split("::")[0]
                    head_term = cluster_key.split("::")[-1]

                    # Separate noise from real sub-clusters
                    noise_members = sub_clusters.pop(-1, [])
                    real_sub: list[dict[str, Any]] = []

                    for label, members in sub_clusters.items():
                        cluster_keywords = [m[0] for m in members]
                        cluster_embeddings = [m[2] for m in members]
                        coherence = embed_client.calculate_cluster_coherence(
                            cluster_embeddings
                        )
                        real_sub.append({
                            "keywords": cluster_keywords,
                            "dominant_intent": intent,
                            "head_term": head_term,
                            "coherence": coherence,
                            "embeddings": cluster_embeddings,
                        })

                    # Assign noise points to nearest sub-cluster by centroid similarity
                    if noise_members and real_sub:
                        centroids = [
                            np.mean(sc["embeddings"], axis=0).tolist()
                            for sc in real_sub
                        ]
                        for kw, prob, emb in noise_members:
                            best_sim = -1.0
                            best_idx = -1
                            for idx, centroid in enumerate(centroids):
                                sim = EmbeddingsClient.cosine_similarity(emb, centroid)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_idx = idx
                            if best_sim >= self.NOISE_SIMILARITY_THRESHOLD and best_idx >= 0:
                                real_sub[best_idx]["keywords"].append(kw)
                                real_sub[best_idx]["embeddings"].append(emb)
                            else:
                                refined_clusters.append({
                                    "keywords": [kw],
                                    "dominant_intent": intent,
                                    "head_term": head_term,
                                    "is_noise": True,
                                })
                    elif noise_members:
                        for kw, prob, emb in noise_members:
                            refined_clusters.append({
                                "keywords": [kw],
                                "dominant_intent": intent,
                                "head_term": head_term,
                                "is_noise": True,
                            })

                    # Recalculate coherence for sub-clusters that absorbed noise
                    for sc in real_sub:
                        sc["coherence"] = embed_client.calculate_cluster_coherence(
                            sc["embeddings"]
                        )
                        refined_clusters.append(sc)

                except Exception:
                    logger.warning(
                        "HDBSCAN clustering failed for cluster, using fallback",
                        extra={
                            "cluster_key": cluster_key,
                            "keyword_count": len(keywords),
                        },
                    )
                    # Fallback: keep as single cluster
                    intent = cluster_key.split("::")[0]
                    refined_clusters.append({
                        "keywords": keywords,
                        "dominant_intent": intent,
                        "head_term": cluster_key.split("::")[-1],
                    })

        return refined_clusters

    def _merge_small_clusters(
        self,
        clusters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge undersized clusters into same-intent neighbors.

        Clusters with fewer than MIN_CLUSTER_SIZE keywords get absorbed into
        the closest same-intent cluster (matched by shared head-term words).
        Remaining orphans with the same intent are grouped together.
        """
        large: list[dict[str, Any]] = []
        small: list[dict[str, Any]] = []

        for c in clusters:
            if len(c.get("keywords", [])) >= self.MIN_CLUSTER_SIZE:
                large.append(c)
            else:
                small.append(c)

        if not small:
            return clusters

        logger.info(
            "Merging small clusters",
            extra={"large": len(large), "small": len(small)},
        )

        # Try to merge each small cluster into a same-intent large cluster
        # that shares at least one head-term word
        unmerged: list[dict[str, Any]] = []
        for sc in small:
            sc_intent = sc.get("dominant_intent", "unknown")
            sc_head_words = set(sc.get("head_term", "").split())

            best_match: dict[str, Any] | None = None
            best_overlap = 0
            for lc in large:
                if lc.get("dominant_intent") != sc_intent:
                    continue
                lc_head_words = set(lc.get("head_term", "").split())
                overlap = len(sc_head_words & lc_head_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = lc

            if best_match is not None and best_overlap > 0:
                best_match["keywords"].extend(sc.get("keywords", []))
            else:
                unmerged.append(sc)

        # Group remaining orphans by intent
        orphan_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sc in unmerged:
            orphan_groups[sc.get("dominant_intent", "unknown")].append(sc)

        for intent, group in orphan_groups.items():
            merged_keywords = []
            head_terms = []
            for sc in group:
                merged_keywords.extend(sc.get("keywords", []))
                head_terms.append(sc.get("head_term", ""))

            # Split into chunks of MAX_CLUSTER_SIZE if too many
            for i in range(0, len(merged_keywords), self.MAX_CLUSTER_SIZE):
                chunk = merged_keywords[i : i + self.MAX_CLUSTER_SIZE]
                large.append({
                    "keywords": chunk,
                    "dominant_intent": intent,
                    "head_term": head_terms[0] if head_terms else "",
                })

        return large

    async def _validate_and_name_clusters(
        self,
        clusters: list[dict[str, Any]],
        brand_context: str,
        *,
        workflowish_mode: bool,
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

        # Process in batches of 25 clusters, run up to 5 batches concurrently
        batch_size = 25
        max_concurrent = 5
        agent = ClusterAgent()
        semaphore = asyncio.Semaphore(max_concurrent)
        total_batches = (len(agent_clusters) + batch_size - 1) // batch_size
        logger.info(
            "Cluster validation batching",
            extra={
                "total_clusters": len(agent_clusters),
                "batch_size": batch_size,
                "total_batches": total_batches,
                "max_concurrent_batches": max_concurrent,
            },
        )

        async def _process_batch(
            batch_index: int,
            batch_start: int,
            batch: list[dict],
        ) -> list[dict]:
            async with semaphore:
                logger.info(
                    "Cluster validation batch started",
                    extra={
                        "batch_index": batch_index + 1,
                        "total_batches": total_batches,
                        "cluster_count": len(batch),
                    },
                )
                try:
                    agent_input = ClusterAgentInput(
                        clusters=batch,
                        context=brand_context,
                    )
                    output = await agent.run(agent_input)

                    results = []
                    for j, validation in enumerate(output.validations):
                        cluster_idx = batch_start + j
                        if cluster_idx < len(clusters):
                            original = clusters[cluster_idx]
                            keywords = original.get("keywords", [])

                            # Find primary keyword: prefer embedding-based
                            # selection for established markets; use clarity-first
                            # selection for workflow-ish markets.
                            primary_kw = None
                            cluster_embeddings = original.get("embeddings")

                            if workflowish_mode:
                                primary_kw = self._select_primary_keyword_workflow(
                                    keywords=keywords,
                                    cluster_embeddings=cluster_embeddings,
                                )
                            elif cluster_embeddings and len(cluster_embeddings) == len(keywords):
                                items_for_scoring = [
                                    {
                                        "search_volume": kw.search_volume,
                                        "difficulty": kw.difficulty,
                                    }
                                    for kw in keywords
                                ]
                                best_idx = EmbeddingsClient.find_primary_in_cluster(
                                    items_for_scoring,
                                    cluster_embeddings,
                                )
                                primary_kw = keywords[best_idx]
                            else:
                                primary_kw = next(
                                    (
                                        kw
                                        for kw in keywords
                                        if kw.keyword.lower()
                                        == validation.primary_keyword.lower()
                                    ),
                                    keywords[0] if keywords else None,
                                )

                            results.append({
                                "name": validation.name,
                                "description": validation.description,
                                "keywords": keywords,
                                "primary_keyword": primary_kw,
                                "dominant_intent": original.get("dominant_intent"),
                                "coherence": validation.coherence_score,
                                "needs_serp_validation": (
                                    validation.coherence_score
                                    < self.LOW_COHERENCE_THRESHOLD
                                ),
                                "action": validation.action,
                                "notes": validation.notes,
                                "total_volume": sum(kw.search_volume or 0 for kw in keywords),
                                "adjusted_volume_sum": sum(
                                    kw.adjusted_volume
                                    if kw.adjusted_volume is not None
                                    else (kw.search_volume or 0)
                                    for kw in keywords
                                ),
                                "avg_difficulty": (
                                    sum(kw.difficulty or 0 for kw in keywords) / len(keywords)
                                    if keywords else 0
                                ),
                                "embeddings": cluster_embeddings,
                            })
                    logger.info(
                        "Cluster validation batch completed",
                        extra={
                            "batch_index": batch_index + 1,
                            "total_batches": total_batches,
                            "result_count": len(results),
                            "fallback_used": False,
                        },
                    )
                    return results

                except Exception:
                    logger.warning(
                        "Cluster validation batch failed, using fallback",
                        extra={
                            "batch_index": batch_index + 1,
                            "total_batches": total_batches,
                            "cluster_count": len(batch),
                        },
                    )
                    # Fallback: create basic cluster info
                    results = []
                    for j, cluster in enumerate(batch):
                        cluster_idx = batch_start + j
                        if cluster_idx < len(clusters):
                            original = clusters[cluster_idx]
                            keywords = original.get("keywords", [])
                            results.append({
                                "name": f"Cluster {cluster_idx + 1}",
                                "description": "",
                                "keywords": keywords,
                                "primary_keyword": (
                                    self._select_primary_keyword_workflow(
                                        keywords=keywords,
                                        cluster_embeddings=original.get("embeddings"),
                                    )
                                    if workflowish_mode
                                    else (keywords[0] if keywords else None)
                                ),
                                "dominant_intent": original.get("dominant_intent"),
                                "coherence": original.get("coherence", 0.5),
                                "needs_serp_validation": True,
                                "action": "keep",
                                "notes": "Auto-generated cluster",
                                "total_volume": sum(kw.search_volume or 0 for kw in keywords),
                                "adjusted_volume_sum": sum(
                                    kw.adjusted_volume
                                    if kw.adjusted_volume is not None
                                    else (kw.search_volume or 0)
                                    for kw in keywords
                                ),
                                "avg_difficulty": (
                                    sum(kw.difficulty or 0 for kw in keywords) / len(keywords)
                                    if keywords else 0
                                ),
                            })
                    logger.info(
                        "Cluster validation batch completed",
                        extra={
                            "batch_index": batch_index + 1,
                            "total_batches": total_batches,
                            "result_count": len(results),
                            "fallback_used": True,
                        },
                    )
                    return results

        # Launch all batches concurrently
        tasks = []
        for batch_index, i in enumerate(range(0, len(agent_clusters), batch_size)):
            batch = agent_clusters[i:i + batch_size]
            tasks.append(_process_batch(batch_index, i, batch))

        batch_results = await asyncio.gather(*tasks)

        # Flatten results in order
        validated = []
        for batch_result in batch_results:
            validated.extend(batch_result)

        return validated

    async def _process_agent_actions(
        self,
        clusters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process split/merge actions recommended by ClusterAgent.

        - split: Re-cluster by sub-intent
        - merge: Flag as recommendation in cluster_notes for manual review
        """
        processed: list[dict[str, Any]] = []

        for cluster in clusters:
            action = cluster.get("action", "keep")

            if action == "split" and len(cluster.get("keywords", [])) >= self.MIN_CLUSTER_SIZE * 2:
                sub_clusters = self._split_cluster(cluster)
                if len(sub_clusters) > 1:
                    logger.info(
                        "Split cluster per agent recommendation",
                        extra={
                            "original_name": cluster.get("name", ""),
                            "original_size": len(cluster.get("keywords", [])),
                            "sub_cluster_count": len(sub_clusters),
                        },
                    )
                    processed.extend(sub_clusters)
                    continue

            if action == "merge":
                notes = cluster.get("notes", "") or ""
                cluster["notes"] = f"[merge_recommended] {notes}".strip()
                logger.info(
                    "Cluster flagged for merge review",
                    extra={
                        "cluster_name": cluster.get("name", ""),
                        "size": len(cluster.get("keywords", [])),
                    },
                )

            processed.append(cluster)

        return processed

    def _split_cluster(self, cluster: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a cluster by sub-intent grouping.

        Groups keywords by their specific intent, then checks if splitting
        produces at least 2 groups meeting MIN_CLUSTER_SIZE.
        """
        keywords = cluster.get("keywords", [])
        embeddings = cluster.get("embeddings")

        # Split by sub-intent (keywords have intent from Step 5)
        sub_groups: dict[str, list[tuple[Any, list[float] | None]]] = defaultdict(list)
        for i, kw in enumerate(keywords):
            sub_key = kw.intent or "unknown"
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            sub_groups[sub_key].append((kw, emb))

        if len(sub_groups) <= 1:
            return [cluster]

        result = []
        for sub_intent, members in sub_groups.items():
            sub_kws = [m[0] for m in members]
            sub_embs = [m[1] for m in members if m[1] is not None]
            coherence = cluster.get("coherence", 0.5)
            if sub_embs and len(sub_embs) == len(sub_kws):
                coherence = EmbeddingsClient.calculate_cluster_coherence(sub_embs)
            result.append({
                "name": cluster.get("name", "") + f" ({sub_intent})",
                "description": cluster.get("description", ""),
                "keywords": sub_kws,
                "primary_keyword": sub_kws[0],
                "dominant_intent": sub_intent,
                "coherence": coherence,
                "needs_serp_validation": True,
                "action": "keep",
                "notes": "Split from parent cluster by sub-intent",
                "total_volume": sum(kw.search_volume or 0 for kw in sub_kws),
                "adjusted_volume_sum": sum(
                    kw.adjusted_volume if kw.adjusted_volume is not None else (kw.search_volume or 0)
                    for kw in sub_kws
                ),
                "avg_difficulty": (
                    sum(kw.difficulty or 0 for kw in sub_kws) / len(sub_kws)
                    if sub_kws else 0
                ),
                "embeddings": sub_embs if len(sub_embs) == len(sub_kws) else None,
            })

        # Only accept if at least 2 groups meet MIN_CLUSTER_SIZE
        valid = [r for r in result if len(r["keywords"]) >= self.MIN_CLUSTER_SIZE]
        if len(valid) >= 2:
            return result
        return [cluster]

    def _select_primary_keyword_workflow(
        self,
        *,
        keywords: list[Keyword],
        cluster_embeddings: list[list[float]] | None,
    ) -> Keyword | None:
        """Select workflow primary keyword by clarity + intent + feasible difficulty."""
        if not keywords:
            return None
        if len(keywords) == 1:
            return keywords[0]

        embedding_scores = self._embedding_representativeness(
            embeddings=cluster_embeddings,
            count=len(keywords),
        )
        best_index = 0
        best_score = -1.0

        for idx, keyword in enumerate(keywords):
            signals = keyword.discovery_signals if isinstance(keyword.discovery_signals, dict) else {}
            word_count = int(signals.get("word_count") or len(keyword.keyword.split()))
            clarity = self._keyword_clarity_score(
                word_count=word_count,
                has_action_verb=bool(signals.get("has_action_verb")),
                has_integration_term=bool(signals.get("has_integration_term")),
                has_two_entities=bool(signals.get("has_two_entities")),
                is_comparison=bool(signals.get("is_comparison")),
            )
            intent_score = float(keyword.intent_score or 0.5)
            difficulty = float(keyword.difficulty if keyword.difficulty is not None else 55.0)
            difficulty_ease = max(0.0, min(1.0, 1.0 - (difficulty / 100.0)))
            representativeness = embedding_scores[idx] if idx < len(embedding_scores) else 0.5

            composite = (
                (clarity * 0.35)
                + (intent_score * 0.30)
                + (difficulty_ease * 0.20)
                + (representativeness * 0.15)
            )
            if composite > best_score:
                best_score = composite
                best_index = idx

        return keywords[best_index]

    def _keyword_clarity_score(
        self,
        *,
        word_count: int,
        has_action_verb: bool,
        has_integration_term: bool,
        has_two_entities: bool,
        is_comparison: bool,
    ) -> float:
        """Estimate how clear and representative a workflow keyword is."""
        if word_count <= 0:
            base = 0.2
        elif 2 <= word_count <= 6:
            base = 0.75
        elif word_count <= 8:
            base = 0.6
        else:
            base = 0.35

        if has_action_verb:
            base += 0.08
        if has_integration_term:
            base += 0.07
        if has_two_entities:
            base += 0.06
        if is_comparison:
            base += 0.04

        return max(0.0, min(1.0, base))

    def _embedding_representativeness(
        self,
        *,
        embeddings: list[list[float]] | None,
        count: int,
    ) -> list[float]:
        """Return centroid similarity per keyword, fallback to neutral values."""
        if not embeddings or len(embeddings) != count:
            return [0.5 for _ in range(count)]

        centroid = np.mean(embeddings, axis=0).tolist()
        scores: list[float] = []
        for emb in embeddings:
            scores.append(max(0.0, min(1.0, EmbeddingsClient.cosine_similarity(emb, centroid))))
        return scores

    def _serialize_clusters(self, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serialize clusters for checkpoint storage.

        Converts Keyword ORM objects to their IDs for JSON serialization.
        Drops embeddings to keep checkpoint size manageable.
        """
        serialized = []
        for cluster in clusters:
            sc: dict[str, Any] = {
                "keyword_ids": [kw.id for kw in cluster.get("keywords", [])],
                "dominant_intent": cluster.get("dominant_intent"),
                "head_term": cluster.get("head_term"),
                "coherence": cluster.get("coherence"),
                "is_noise": cluster.get("is_noise", False),
            }
            serialized.append(sc)
        return serialized

    def _deserialize_clusters(
        self,
        serialized: list[dict[str, Any]],
        all_keywords: list[Keyword],
    ) -> list[dict[str, Any]]:
        """Deserialize clusters from checkpoint, re-linking Keyword objects."""
        kw_by_id = {kw.id: kw for kw in all_keywords}
        clusters = []
        for sc in serialized:
            keywords = [kw_by_id[kid] for kid in sc["keyword_ids"] if kid in kw_by_id]
            cluster: dict[str, Any] = {
                "keywords": keywords,
                "dominant_intent": sc.get("dominant_intent"),
                "head_term": sc.get("head_term"),
            }
            if sc.get("coherence") is not None:
                cluster["coherence"] = sc["coherence"]
            if sc.get("is_noise"):
                cluster["is_noise"] = True
            clusters.append(cluster)
        return clusters

    def _resolve_pillar_seed(self, keywords: list[Keyword]) -> str | None:
        """Find the most common seed_topic_id among the cluster's keywords."""
        counts: dict[str, int] = {}
        for kw in keywords:
            sid = kw.seed_topic_id
            if sid:
                sid_str = str(sid)
                counts[sid_str] = counts.get(sid_str, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    async def _persist_results(self, result: ClusteringOutput) -> None:
        """Save clusters to database as Topics."""
        # Delete existing topics for this project
        existing = await self.session.execute(
            select(Topic).where(Topic.project_id == self.project_id)
        )
        for topic in existing.scalars():
            await topic.delete(self.session)

        # Create new topics
        for cluster in result.clusters:
            primary_kw = cluster.get("primary_keyword")
            keywords = cluster.get("keywords", [])
            total_volume = cluster.get("total_volume", 0)
            adjusted_volume_sum = cluster.get("adjusted_volume_sum", total_volume)
            notes = cluster.get("notes") or ""
            if cluster.get("needs_serp_validation"):
                notes = "[needs_serp_validation] " + notes if notes else "[needs_serp_validation]"

            # Resolve pillar seed: pick the most common seed_topic_id
            # among the cluster's keywords (provenance set in Step 3).
            pillar_seed_id = self._resolve_pillar_seed(keywords)

            topic_data = TopicCreateDTO(
                project_id=self.project_id,
                name=cluster["name"],
                description=cluster.get("description"),
                primary_keyword_id=primary_kw.id if primary_kw else None,
                pillar_seed_topic_id=pillar_seed_id,
                cluster_method="two_stage_hdbscan_llm",
                dominant_intent=cluster.get("dominant_intent"),
                dominant_page_type=primary_kw.recommended_page_type if primary_kw else None,
                funnel_stage=primary_kw.funnel_stage if primary_kw else None,
                total_volume=total_volume,
                adjusted_volume_sum=adjusted_volume_sum,
                keyword_count=len(keywords),
                estimated_demand=total_volume,
                avg_difficulty=cluster.get("avg_difficulty", 0),
                cluster_coherence=cluster.get("coherence", 0.5),
                cluster_notes=notes or None,
            )
            topic = Topic.create(self.session, topic_data)
            # `Topic.id` is generated on flush; flush before FK assignment so
            # keyword -> topic links are persisted correctly.
            await self.session.flush()

            # Update keywords to reference this topic
            for kw in keywords:
                kw.topic_id = topic.id
                if kw == primary_kw:
                    # Mark as primary
                    pass  # Already set via primary_keyword_id

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, 6)

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
