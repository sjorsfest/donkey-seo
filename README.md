# DonkeySEO

Keyword research backend service with a 14-step pipeline for programmatic content planning.

## Features

- **14-step keyword research pipeline** for systematic content strategy
- **Brand profile extraction** from website scraping using LLM
- **Keyword expansion** via DataForSEO API
- **Intent classification** and page type recommendations
- **Topic clustering** with priority scoring
- **Content brief generation** with writer instructions
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
| 8 | SERP Validation | Validate with SERP data | Optional |
| 9 | Content Inventory | Index existing content | Optional |
| 10 | Cannibalization | Detect content overlap | Optional |
| 11 | Internal Linking | Plan internal links | Optional |
| 12 | Content Brief | Generate writer briefs | Yes |
| 13 | Writer Templates | Create QA checklists | Yes |
| 14 | GSC/GA4 Audit | Import analytics data | Optional |

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
- `POST /api/v1/pipeline/{project_id}/resume` - Resume pipeline
- `GET /api/v1/pipeline/{project_id}/runs` - List runs
- `GET /api/v1/pipeline/{project_id}/runs/{run_id}/progress` - Get progress

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
