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

# DataForSEO
DATAFORSEO_LOGIN=your-login
DATAFORSEO_PASSWORD=your-password
```

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

# Linting
ruff check app

# Format
ruff format app
```

## License

MIT
