# DonkeySEO

Keyword research backend service with a 14-step pipeline for programmatic content planning.

## Features

- **14-step keyword research pipeline** for systematic content strategy
- **Split orchestration modes** for discovery loop and content production
- **Brand profile extraction** from website scraping using LLM
- **Keyword expansion** via DataForSEO API
- **Intent classification** and page type recommendations
- **SERP validation** to confirm intent/page type against live search results
- **Topic clustering** with priority scoring
- **Content brief generation** enriched with SERP validation signals
- **Publication date manager** with configurable cadence for content calendar planning
- **JWT authentication** for secure API access
- **Configurable LLM providers** (OpenAI & Anthropic)
- **Dynamic per-agent model selector** with max-price guardrails

## Tech Stack

- **FastAPI** - REST API framework
- **SQLAlchemy** - Async ORM with PostgreSQL
- **Alembic** - Database migrations
- **Pydantic AI** - LLM agents for content analysis
- **Redis** - Caching and rate limiting
- **DataForSEO** - Keyword data API

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Docker (optional, for database services)

### 1. Start Database Services

```bash
docker-compose up -d
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 4. Run Migrations

```bash
alembic upgrade head
```

### 5. Start the Server

```bash
uvicorn app.main:app --reload
```

API docs available at: http://localhost:8000/docs

## Pipeline Steps

| Step | Name | Description | Required |
|------|------|-------------|----------|
| 0 | Setup | Project configuration | Yes |
| 1 | Brand Profile | Extract brand info from website | Yes |
| 2 | Seed Topics | Generate initial topic pillars | Yes |
| 3 | Keyword Expansion | Expand keywords via API | Yes |
| 4 | Metrics Enrichment | Get volume, CPC, difficulty | Yes |
| 5 | Intent Labeling | Classify search intent | Yes |
| 6 | Clustering | Group keywords into topics | Yes |
| 7 | Prioritization | Rank topic backlog | Yes |
| 8 | SERP Validation | Validate intent/page type with live SERP data | Optional |
| 9 | Content Inventory | Index existing content | Optional |
| 10 | Cannibalization | Detect content overlap | Optional |
| 11 | Internal Linking | Plan internal links | Optional |
| 12 | Content Brief | Generate writer briefs | Yes |
| 13 | Writer Templates | Create QA checklists | Yes |
| 14 | Article Generation | Generate modular SEO articles + HTML | Yes |

## Pipeline Modes

The pipeline start endpoint supports three modes:

- `full`: Standard range execution (default behavior).
- `discovery_loop`: Runs an adaptive loop for topic opportunity discovery.
  - Executes Step `2 -> 8` repeatedly.
  - If brand setup is missing, runs Step `0 -> 1` once before looping.
  - Stops when enough topics pass fit + SERP gate, or pauses after max iterations.
  - Can auto-start `content_production` with accepted topics.
- `content_production`: Runs content generation only.
  - Executes Step `12 -> 14`.
  - Can use selected topic IDs + max briefs from discovery output.

### Discovery Acceptance Gate

In `discovery_loop`, a topic is accepted only when:

- Step 7 fit tier is `primary` or `secondary`.
- For `established_category` topics: primary-keyword SERP gate passes:
  - `difficulty <= max_keyword_difficulty`
  - `domain_diversity >= min_domain_diversity`
  - No `intent_mismatch` (if `require_intent_match=true`)
- For `fragmented_workflow` topics: cluster-level SERP gate passes:
  - Cluster SERP evidence exists (primary or fallback variant)
  - `difficulty <= max_keyword_difficulty`
  - Not saturated (`serp_servedness_score` + `serp_competitor_density`)
  - `serp_intent_confidence >= min_serp_intent_confidence` (if `require_intent_match=true`)

Per-topic accept/reject decisions are persisted per iteration as discovery snapshots.

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Create account
- `POST /api/v1/auth/login` - Login, get tokens
- `POST /api/v1/auth/refresh` - Refresh token
- `GET /api/v1/auth/me` - Get current user

### Projects
- `POST /api/v1/projects/` - Create project
- `GET /api/v1/projects/` - List projects
- `GET /api/v1/projects/{id}` - Get project
- `PUT /api/v1/projects/{id}` - Update project
- `DELETE /api/v1/projects/{id}` - Delete project

### Pipeline
- `POST /api/v1/pipeline/{project_id}/start` - Start pipeline
- `POST /api/v1/pipeline/{project_id}/pause` - Pause pipeline
- `POST /api/v1/pipeline/{project_id}/resume/{run_id}` - Resume pipeline
- `GET /api/v1/pipeline/{project_id}/runs` - List runs
- `GET /api/v1/pipeline/{project_id}/runs/{run_id}/progress` - Get progress
- `GET /api/v1/pipeline/{project_id}/runs/{run_id}/discovery-snapshots` - Get discovery loop topic decisions

### Pipeline Start Examples

Start discovery loop with adaptive retries and auto-handoff:

```json
{
  "mode": "discovery_loop",
  "strategy": {
    "fit_threshold_profile": "aggressive",
    "scope_mode": "strict",
    "market_mode_override": "auto",
    "include_topics": ["customer support automation"],
    "exclude_topics": ["medical advice"]
  },
  "discovery": {
    "max_iterations": 3,
    "min_eligible_topics": 8,
    "require_serp_gate": true,
    "max_keyword_difficulty": 65,
    "min_domain_diversity": 0.5,
    "require_intent_match": true,
    "max_serp_servedness": 0.75,
    "max_serp_competitor_density": 0.7,
    "min_serp_intent_confidence": 0.35,
    "auto_start_content": true
  },
  "content": {
    "max_briefs": 20,
    "posts_per_week": 3,
    "preferred_weekdays": [0, 2, 4],
    "min_lead_days": 7
  }
}
```

Start content-only pipeline:

```json
{
  "mode": "content_production",
  "content": {
    "max_briefs": 15,
    "posts_per_week": 2,
    "preferred_weekdays": [1, 3],
    "min_lead_days": 5
  }
}
```

### Keywords
- `GET /api/v1/keywords/{project_id}` - List keywords
- `GET /api/v1/keywords/{project_id}/{keyword_id}` - Get keyword
- `POST /api/v1/keywords/{project_id}` - Add keyword
- `PUT /api/v1/keywords/{project_id}/{keyword_id}` - Update keyword
- `POST /api/v1/keywords/{project_id}/bulk-update` - Bulk update

### Topics
- `GET /api/v1/topics/{project_id}` - List topics
- `GET /api/v1/topics/{project_id}/ranked` - Get prioritized backlog
- `GET /api/v1/topics/{project_id}/hierarchy` - Get topic tree
- `POST /api/v1/topics/{project_id}/merge` - Merge topics

### Content
- `GET /api/v1/content/{project_id}/briefs` - List briefs
- `GET /api/v1/content/{project_id}/briefs/{brief_id}` - Get brief
- `POST /api/v1/content/{project_id}/briefs` - Create brief
- `GET /api/v1/content/{project_id}/briefs/{brief_id}/instructions` - Get writer instructions
- `GET /api/v1/content/{project_id}/articles` - List generated articles
- `GET /api/v1/content/{project_id}/briefs/{brief_id}/article` - Get canonical article for brief
- `POST /api/v1/content/{project_id}/briefs/{brief_id}/article/regenerate` - Regenerate article version
- `GET /api/v1/content/{project_id}/articles/{article_id}/versions/{version_number}` - Get article version

## Configuration

Key environment variables:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/donkeyseo

# JWT
JWT_SECRET_KEY=your-secret-key

# LLM (choose default)
DEFAULT_LLM_MODEL=openai:gpt-4-turbo
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Dynamic model selector (optional)
MODEL_SELECTOR_ENABLED=false
MODEL_SELECTOR_SNAPSHOT_PATH=app/agents/model_selection_snapshot.json
# USD per 1M tokens
MODEL_SELECTOR_MAX_PRICE_DEV=0
MODEL_SELECTOR_MAX_PRICE_STAGING=0
MODEL_SELECTOR_MAX_PRICE_PROD=0
MODEL_SELECTOR_OPENROUTER_WEIGHT=0.75
MODEL_SELECTOR_ARENA_WEIGHT=0.25
MODEL_SELECTOR_FALLBACK_MODEL=openrouter:google/gemma-3-27b-it:free

# DataForSEO
DATAFORSEO_LOGIN=your-login
DATAFORSEO_PASSWORD=your-password
```

### Dynamic Model Selector

Model selection is refreshed from OpenRouter rankings (primary) and Arena leaderboard signals (secondary bonus), then persisted to:

- `app/agents/model_selection_snapshot.json` (durable snapshot)
- Redis keys:
  - `model_selector:snapshot:latest`
  - `model_selector:selected:<environment>:<AgentClassName>`

Run refresh manually:

```bash
uv run python scripts/refresh_model_selection.py --env development --env staging --env production
```

Mirror to Redis in the same run:

```bash
uv run python scripts/refresh_model_selection.py --env development --env staging --env production --write-redis
```

Dry run (fetch + score only):

```bash
uv run python scripts/refresh_model_selection.py --dry-run
```

Recommended operations setup: run `refresh_model_selection.py` daily (for example via CI scheduler or cron) and keep `MODEL_SELECTOR_ENABLED=false` until the snapshot quality looks good in development.

## Development

```bash
# Generate typed model DTOs (auto-generated)
python scripts/generate_model_dtos.py

# Run tests
pytest

# Type checking
ty check app

# Typed write guardrails (warning mode)
python scripts/check_typed_writes.py

# Refresh per-agent model selections
make refresh-models

# Linting
ruff check app

# Format
ruff format app
```

## License

MIT
