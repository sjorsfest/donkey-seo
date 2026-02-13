"""Guardrail check for direct ORM constructor usage in service/api layers.

Default mode is warning-only. Use ``--strict`` to fail on violations.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MODEL_CONSTRUCTOR_PATTERN = re.compile(
    r"\b(BrandProfile|BriefDelta|ContentBrief|Keyword|OAuthAccount|PipelineRun|"
    r"Project|ProjectStyleGuide|SeedTopic|StepExecution|Topic|User|WriterInstructions)\("
)

DEFAULT_SCAN_ROOTS = ("app/services", "app/api")
IGNORE_PATH_PARTS = (
    "app/persistence/typed/adapters",
    "app/models",
)


def should_scan(path: Path) -> bool:
    """Return True for source files in scan roots, excluding known safe zones."""
    if path.suffix != ".py":
        return False
    path_str = str(path)
    if any(ignored in path_str for ignored in IGNORE_PATH_PARTS):
        return False
    return True


def find_violations(paths: list[Path]) -> list[tuple[Path, int, str]]:
    """Find potential direct ORM constructor write calls."""
    violations: list[tuple[Path, int, str]] = []

    for base in paths:
        if not base.exists():
            continue
        for file_path in base.rglob("*.py"):
            if not should_scan(file_path):
                continue

            for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if MODEL_CONSTRUCTOR_PATTERN.search(line):
                    violations.append((file_path, lineno, line.strip()))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on violations")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_SCAN_ROOTS),
        help="Root directories to scan",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    violations = find_violations(roots)

    if not violations:
        print("typed-writes check: no direct ORM constructor usage found in scan roots")
        return 0

    print("typed-writes check: found direct ORM constructor usage")
    for file_path, lineno, line in violations:
        print(f"  {file_path}:{lineno} -> {line}")

    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
