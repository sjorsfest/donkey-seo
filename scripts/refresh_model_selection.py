"""Refresh per-agent model selections from OpenRouter + Arena signals."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import settings
from app.services.model_selector.arena_client import ArenaClient
from app.services.model_selector.openrouter_client import OpenRouterClient
from app.services.model_selector.registry import write_snapshot_to_redis
from app.services.model_selector.scoring import select_best_model
from app.services.model_selector.types import SelectionSnapshot

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USECASE_PATH = ROOT / "app" / "agents" / "agent_usecases.yaml"


@dataclass(slots=True)
class AgentUseCase:
    """Agent-specific ranking mapping."""

    use_case: str
    openrouter_category: str
    arena_use_case_slug: str


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        action="append",
        choices=["development", "staging", "production"],
        help="Environment(s) to refresh (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and score but do not write snapshot",
    )
    parser.add_argument(
        "--write-redis",
        action="store_true",
        help="Mirror generated snapshot and per-agent picks to Redis",
    )
    parser.add_argument(
        "--order",
        default="top-weekly",
        help="OpenRouter ranking order (default: top-weekly)",
    )
    parser.add_argument(
        "--usecases-path",
        default=str(DEFAULT_USECASE_PATH),
        help="Path to agent use-case YAML",
    )
    return parser.parse_args()


def load_agent_usecases(path: Path) -> dict[str, AgentUseCase]:
    """Load agent use-case mapping file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("agent use-case config must be a mapping")

    agents_payload = payload.get("agents")
    if not isinstance(agents_payload, dict):
        raise ValueError("agent use-case config must include 'agents' mapping")

    mapping: dict[str, AgentUseCase] = {}
    for agent_class, config in agents_payload.items():
        if not isinstance(config, dict):
            raise ValueError(f"config for {agent_class} must be a mapping")

        use_case = str(config.get("use_case", "")).strip()
        openrouter_category = str(config.get("openrouter_category", "")).strip()
        arena_use_case_slug = str(config.get("arena_use_case_slug", "")).strip()
        if not use_case:
            raise ValueError(f"{agent_class}.use_case must be non-empty")
        if not openrouter_category:
            raise ValueError(f"{agent_class}.openrouter_category must be non-empty")

        mapping[str(agent_class)] = AgentUseCase(
            use_case=use_case,
            openrouter_category=openrouter_category,
            arena_use_case_slug=arena_use_case_slug,
        )

    return mapping


def get_environment_max_price(environment: str) -> float:
    """Resolve max-price cap by environment."""
    if environment == "development":
        return settings.model_selector_max_price_dev
    if environment == "staging":
        return settings.model_selector_max_price_staging
    return settings.model_selector_max_price_prod


def write_snapshot_atomically(snapshot_path: Path, snapshot: SelectionSnapshot) -> None:
    """Persist JSON snapshot atomically (write temp + rename)."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")

    payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
    tmp_path.write_text(payload + "\n", encoding="utf-8")
    tmp_path.replace(snapshot_path)


async def build_snapshot(
    *,
    environments: list[str],
    usecases: dict[str, AgentUseCase],
    order: str,
) -> tuple[SelectionSnapshot, dict[str, Any]]:
    """Fetch upstream data, score candidates, and build snapshot payload."""
    openrouter_client = OpenRouterClient(api_key=settings.openrouter_api_key)
    arena_client = ArenaClient()

    openrouter_cache: dict[tuple[str, float], tuple[list[Any], dict[str, Any]]] = {}
    arena_cache: dict[str, tuple[dict[str, float], dict[str, Any]]] = {}

    available_slugs, sitemap_meta = await arena_client.fetch_available_use_cases()

    env_payloads: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "fallbacks": 0,
        "arena_degraded": 0,
        "openrouter_failures": 0,
        "openrouter_failure_details": {},
        "arena_failure_details": {},
        "openrouter_category_fallbacks": {},
    }
    if not sitemap_meta.get("ok"):
        summary["arena_failure_details"]["sitemap"] = sitemap_meta.get(
            "error",
            "arena_sitemap_unavailable",
        )

    for environment in environments:
        max_price = get_environment_max_price(environment)
        env_agents: dict[str, Any] = {}

        for agent_class, mapping in usecases.items():
            or_cache_key = (mapping.openrouter_category, max_price)
            if or_cache_key not in openrouter_cache:
                openrouter_cache[or_cache_key] = await openrouter_client.fetch_models(
                    category=mapping.openrouter_category,
                    max_price=max_price,
                    order=order,
                )

            candidates, openrouter_meta = openrouter_cache[or_cache_key]
            if not openrouter_meta.get("ok"):
                failure_key = f"{mapping.openrouter_category}@{max_price}"
                summary["openrouter_failure_details"][failure_key] = openrouter_meta.get(
                    "error",
                    "unknown_openrouter_error",
                )
            elif openrouter_meta.get("category_fallback_used"):
                fallback_key = f"{mapping.openrouter_category}@{max_price}"
                summary["openrouter_category_fallbacks"][fallback_key] = openrouter_meta.get(
                    "category_error",
                    "category_retry_used",
                )

            arena_meta: dict[str, Any] = {
                "source": "arena",
                "ok": False,
                "skipped": True,
                "reason": "arena_use_case_not_configured",
            }
            raw_arena_scores: dict[str, float] = {}

            slug = mapping.arena_use_case_slug
            if slug:
                if available_slugs and slug not in available_slugs:
                    arena_meta = {
                        "source": "arena",
                        "ok": False,
                        "skipped": True,
                        "use_case": slug,
                        "reason": "use_case_not_in_sitemap",
                    }
                    summary["arena_degraded"] += 1
                else:
                    if slug not in arena_cache:
                        arena_cache[slug] = await arena_client.fetch_leaderboard_scores(slug)
                    raw_arena_scores, arena_meta = arena_cache[slug]
                    if not arena_meta.get("ok"):
                        summary["arena_degraded"] += 1
                        summary["arena_failure_details"][slug] = arena_meta.get(
                            "error",
                            "unknown_arena_error",
                        )

            selection = select_best_model(
                agent_class=agent_class,
                candidates=candidates,
                raw_arena_scores=raw_arena_scores,
                openrouter_weight=settings.model_selector_openrouter_weight,
                arena_weight=settings.model_selector_arena_weight,
                max_price=max_price,
                fallback_model=settings.model_selector_fallback_model,
                openrouter_meta=openrouter_meta,
                arena_meta=arena_meta,
            )
            if selection.fallback_used:
                summary["fallbacks"] += 1

            env_agents[agent_class] = selection.to_dict()

        env_payloads[environment] = {
            "max_price": max_price,
            "agents": env_agents,
            "source_health": {
                "arena_sitemap": sitemap_meta,
            },
        }

    snapshot = SelectionSnapshot(
        version="1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        environments=env_payloads,
    )
    summary["environments"] = len(environments)
    summary["agents"] = len(usecases)
    summary["openrouter_failures"] = len(summary["openrouter_failure_details"])
    summary["arena_degraded"] = len(summary["arena_failure_details"])
    return snapshot, summary


def print_failure_details(summary: dict[str, Any]) -> None:
    """Print compact source failure diagnostics for operator debugging."""
    openrouter_errors = summary.get("openrouter_failure_details", {})
    if isinstance(openrouter_errors, dict) and openrouter_errors:
        print("OpenRouter failures:", file=sys.stderr)
        for key, error in sorted(openrouter_errors.items()):
            print(f"  - {key}: {error}", file=sys.stderr)

    arena_errors = summary.get("arena_failure_details", {})
    if isinstance(arena_errors, dict) and arena_errors:
        print("Arena failures:", file=sys.stderr)
        for key, error in sorted(arena_errors.items()):
            print(f"  - {key}: {error}", file=sys.stderr)

    category_fallbacks = summary.get("openrouter_category_fallbacks", {})
    if isinstance(category_fallbacks, dict) and category_fallbacks:
        print("OpenRouter category fallbacks used:", file=sys.stderr)
        for key, reason in sorted(category_fallbacks.items()):
            print(f"  - {key}: {reason}", file=sys.stderr)


async def async_main() -> int:
    """Async entrypoint."""
    args = parse_args()
    environments = args.env or ["development", "staging", "production"]

    usecases_path = Path(args.usecases_path)
    if not usecases_path.is_absolute():
        usecases_path = Path.cwd() / usecases_path

    try:
        usecases = load_agent_usecases(usecases_path)
    except Exception as exc:
        print(f"Failed to load use-case mapping: {exc}", file=sys.stderr)
        return 1

    snapshot, summary = await build_snapshot(
        environments=environments,
        usecases=usecases,
        order=args.order,
    )

    snapshot_path = Path(settings.model_selector_snapshot_path)
    if not snapshot_path.is_absolute():
        snapshot_path = Path.cwd() / snapshot_path

    if args.dry_run:
        print(
            "DRY RUN model selector: "
            f"envs={summary['environments']} agents={summary['agents']} "
            f"fallbacks={summary['fallbacks']} arena_degraded={summary['arena_degraded']} "
            f"openrouter_failures={summary['openrouter_failures']}"
        )
        print_failure_details(summary)
        return 0

    write_snapshot_atomically(snapshot_path, snapshot)

    if args.write_redis:
        try:
            await write_snapshot_to_redis(snapshot)
        except Exception as exc:
            print(f"Snapshot written, but Redis mirror failed: {exc}", file=sys.stderr)
            return 1

    print(
        "Model selector refreshed: "
        f"path={snapshot_path} envs={summary['environments']} agents={summary['agents']} "
        f"fallbacks={summary['fallbacks']} arena_degraded={summary['arena_degraded']} "
        f"openrouter_failures={summary['openrouter_failures']}"
    )
    print_failure_details(summary)
    return 0


def main() -> int:
    """Sync wrapper."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
