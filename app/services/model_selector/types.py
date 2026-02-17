"""Domain types for model selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ModelCandidate:
    """Normalized model candidate from upstream ranking sources."""

    model: str
    raw_model_id: str
    price_per_million: float
    openrouter_rank: int
    openrouter_rank_score: float
    openrouter_popularity: float | None = None
    arena_score: float = 0.0
    final_score: float = 0.0
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentSelection:
    """Resolved model selection for a single agent in one environment."""

    agent_class: str
    model: str
    max_price: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize selection to JSON-compatible dict."""
        return {
            "model": self.model,
            "max_price": self.max_price,
            "score_breakdown": self.score_breakdown,
            "source_metadata": self.source_metadata,
            "fallback_used": self.fallback_used,
        }

    @classmethod
    def from_dict(cls, agent_class: str, payload: dict[str, Any]) -> "AgentSelection":
        """Deserialize selection payload."""
        return cls(
            agent_class=agent_class,
            model=str(payload.get("model", "")),
            max_price=float(payload.get("max_price", 0.0)),
            score_breakdown=dict(payload.get("score_breakdown", {})),
            source_metadata=dict(payload.get("source_metadata", {})),
            fallback_used=bool(payload.get("fallback_used", False)),
        )


@dataclass(slots=True)
class SelectionSnapshot:
    """Full selector snapshot persisted to disk/redis."""

    version: str
    generated_at: str
    environments: dict[str, dict[str, Any]]

    @classmethod
    def empty(cls) -> "SelectionSnapshot":
        """Create an empty snapshot shell."""
        return cls(
            version="1",
            generated_at=datetime.now(timezone.utc).isoformat(),
            environments={},
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SelectionSnapshot":
        """Deserialize snapshot payload."""
        return cls(
            version=str(payload.get("version", "1")),
            generated_at=str(payload.get("generated_at", "")),
            environments=dict(payload.get("environments", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot payload."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "environments": self.environments,
        }

    def get_agent_model(self, environment: str, agent_class_name: str) -> str | None:
        """Read selected model for a specific agent/environment."""
        env_payload = self.environments.get(environment)
        if not isinstance(env_payload, dict):
            return None

        agents_payload = env_payload.get("agents")
        if not isinstance(agents_payload, dict):
            return None

        agent_payload = agents_payload.get(agent_class_name)
        if not isinstance(agent_payload, dict):
            return None

        model = agent_payload.get("model")
        if isinstance(model, str) and model:
            return model
        return None
