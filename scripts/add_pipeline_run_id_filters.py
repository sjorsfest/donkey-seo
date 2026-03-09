#!/usr/bin/env python3
"""Add pipeline_run_id filters to discovery step queries."""

import re
from pathlib import Path

# Files to update
files = [
    "app/services/steps/discovery/step_04_metrics.py",
    "app/services/steps/discovery/step_05_intent.py",
    "app/services/steps/discovery/step_06_clustering.py",
    "app/services/steps/discovery/step_07_prioritization.py",
    "app/services/steps/discovery/step_08_serp.py",
]

for file_path in files:
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping {file_path} - not found")
        continue

    content = path.read_text()
    original = content

    # Pattern 1: select(Keyword).where(Keyword.project_id ==
    content = re.sub(
        r'select\(Keyword\)\.where\(\s*Keyword\.project_id\s*==\s*([^)]+)\s*\)',
        r'select(Keyword).where(\n                Keyword.project_id == \1,\n                Keyword.pipeline_run_id == str(self.execution.pipeline_run_id),\n            )',
        content
    )

    # Pattern 2: select(Topic).where(Topic.project_id ==
    content = re.sub(
        r'select\(Topic\)\.where\(\s*Topic\.project_id\s*==\s*([^)]+)\s*\)',
        r'select(Topic).where(\n                Topic.project_id == \1,\n                Topic.pipeline_run_id == str(self.execution.pipeline_run_id),\n            )',
        content
    )

    if content != original:
        path.write_text(content)
        print(f"✓ Updated {file_path}")
    else:
        print(f"  No changes needed in {file_path}")

print("\nDone!")
