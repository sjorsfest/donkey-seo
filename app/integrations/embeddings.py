"""Embeddings integration for semantic clustering.

Uses OpenRouter embeddings (OpenAI-compatible API) with HDBSCAN for density-aware
clustering.
"""

import logging
from typing import Any

import httpx
import numpy as np
from hdbscan import HDBSCAN

from app.config import settings
from app.core.exceptions import APIKeyMissingError, ExternalAPIError

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Client for generating embeddings and clustering.

    Uses OpenRouter's OpenAI-compatible embeddings endpoint with
    qwen/qwen3-embedding-8b.
    Implements HDBSCAN for density-aware clustering (no fixed threshold).
    """

    EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
    EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.default_model = settings.embeddings_model or self.EMBEDDING_MODEL
        configured_provider = (settings.embeddings_provider or "").strip().lower()
        self.provider = configured_provider or None
        self.allow_fallbacks = settings.embeddings_allow_fallbacks
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

        if not self.api_key:
            raise APIKeyMissingError("OpenRouter (for embeddings)")

    async def __aenter__(self) -> "EmbeddingsClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client must be used as async context manager")
        return self._client

    async def get_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            model: Optional model override

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        resolved_model = model or self.default_model
        logger.info(
            "Generating embeddings",
            extra={
                "text_count": len(texts),
                "model": resolved_model,
                "provider": self.provider,
            },
        )
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]

            try:
                payload: dict[str, Any] = {
                    "model": resolved_model,
                    "input": batch,
                }
                if self.provider:
                    payload["provider"] = {
                        "order": [self.provider],
                        "allow_fallbacks": self.allow_fallbacks,
                    }

                response = await self.client.post(
                    self.EMBEDDING_URL,
                    json=payload,
                )

                if response.status_code != 200:
                    logger.warning("Embeddings API error", extra={"status": response.status_code})
                    raise ExternalAPIError(
                        "OpenRouter Embeddings",
                        f"API error: {response.status_code} - {response.text}"
                    )

                result = response.json()
                data = result.get("data", [])

                # Sort by index to maintain order
                sorted_data = sorted(data, key=lambda x: x.get("index", 0))
                batch_embeddings = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

            except httpx.HTTPError as e:
                logger.warning("Embeddings HTTP error", extra={"error": str(e)})
                raise ExternalAPIError("OpenRouter Embeddings", str(e)) from e

        return all_embeddings

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def cluster_hdbscan(
        embeddings: list[list[float]],
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.0,
    ) -> tuple[list[int], list[float]]:
        """Cluster embeddings using HDBSCAN (density-aware, no fixed threshold).

        HDBSCAN advantages over fixed-threshold cosine similarity:
        - Handles variable density across niches
        - Automatically determines cluster count
        - Identifies noise points (outliers)
        - No need to tune similarity threshold

        Args:
            embeddings: List of embedding vectors
            min_cluster_size: Minimum points to form a cluster
            min_samples: Minimum samples for core point
            cluster_selection_epsilon: Distance threshold for cluster selection

        Returns:
            Tuple of (cluster_labels, probabilities)
            - cluster_labels: Cluster ID for each point (-1 = noise/outlier)
            - probabilities: Probability of belonging to assigned cluster
        """
        if not embeddings or len(embeddings) < min_cluster_size:
            # Not enough points to cluster
            return [-1] * len(embeddings), [0.0] * len(embeddings)

        logger.info("Running HDBSCAN clustering", extra={"point_count": len(embeddings), "min_cluster_size": min_cluster_size})
        # Convert to numpy array
        X = np.array(embeddings)

        # Initialize HDBSCAN clusterer
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",  # Works well with normalized embeddings
            cluster_selection_method="eom",  # Excess of mass (handles varying densities)
        )

        # Fit and predict
        cluster_labels = clusterer.fit_predict(X)
        probabilities = clusterer.probabilities_

        n_clusters = len(set(cluster_labels.tolist())) - (1 if -1 in cluster_labels else 0)
        noise_count = cluster_labels.tolist().count(-1)
        logger.info("HDBSCAN complete", extra={"clusters_found": n_clusters, "noise_points": noise_count, "total_points": len(embeddings)})

        return cluster_labels.tolist(), probabilities.tolist()

    @staticmethod
    def group_by_cluster(
        items: list[Any],
        cluster_labels: list[int],
        probabilities: list[float],
    ) -> dict[int, list[tuple[Any, float]]]:
        """Group items by their cluster assignment.

        Args:
            items: Original items (keywords, etc.)
            cluster_labels: Cluster ID for each item
            probabilities: Probability of cluster membership

        Returns:
            Dict mapping cluster_id -> [(item, probability), ...]
            Cluster -1 contains noise/outlier points
        """
        clusters: dict[int, list[tuple[Any, float]]] = {}

        for item, label, prob in zip(items, cluster_labels, probabilities):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((item, prob))

        return clusters

    @staticmethod
    def calculate_cluster_coherence(
        embeddings: list[list[float]],
    ) -> float:
        """Calculate coherence score for a cluster (0-1).

        Measures how similar all items in a cluster are to each other.
        Higher score = more coherent cluster.

        Args:
            embeddings: List of embedding vectors for cluster members

        Returns:
            Coherence score 0-1 (1 = perfectly similar)
        """
        if len(embeddings) < 2:
            return 1.0

        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)

        # Calculate average cosine similarity to centroid
        similarities = []
        for emb in embeddings:
            sim = EmbeddingsClient.cosine_similarity(emb, centroid.tolist())
            similarities.append(sim)

        return float(np.mean(similarities))

    @staticmethod
    def find_primary_in_cluster(
        items: list[dict[str, Any]],
        embeddings: list[list[float]],
        volume_key: str = "search_volume",
        difficulty_key: str = "difficulty",
    ) -> int:
        """Find the best primary keyword in a cluster.

        Selection criteria:
        1. Highest search volume
        2. Medium difficulty (not hardest)
        3. Closest to cluster centroid (most representative)

        Args:
            items: List of keyword dicts with metrics
            embeddings: Corresponding embeddings
            volume_key: Key for search volume in item dict
            difficulty_key: Key for difficulty in item dict

        Returns:
            Index of the best primary keyword
        """
        if len(items) == 0:
            return 0

        if len(items) == 1:
            return 0

        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)

        # Score each item
        scores = []
        for i, (item, emb) in enumerate(zip(items, embeddings)):
            volume = item.get(volume_key) or 0
            difficulty = item.get(difficulty_key) or 50  # Default medium

            # Normalize scores
            volume_score = min(volume / 10000, 1.0)  # Cap at 10k
            difficulty_score = 1.0 - abs(difficulty - 40) / 100  # Prefer ~40 difficulty
            centroid_sim = EmbeddingsClient.cosine_similarity(emb, centroid.tolist())

            # Combined score (weighted)
            combined = (
                volume_score * 0.5 +
                difficulty_score * 0.2 +
                centroid_sim * 0.3
            )
            scores.append(combined)

        return int(np.argmax(scores))
