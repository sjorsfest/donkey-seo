# Pipeline Run ID Scoping Implementation Plan

## Goal
Scope Topics, Keywords, and SeedTopics by `pipeline_run_id` so each discovery run works with its own data without seeing or deleting data from other runs.

## Database Changes ✅

### Models Updated
1. ✅ **SeedTopic** - Added `pipeline_run_id` field (nullable, indexed, CASCADE delete)
2. ✅ **Keyword** - Added `pipeline_run_id` field (nullable, indexed, CASCADE delete)
3. ✅ **Topic** - Added `pipeline_run_id` field (nullable, indexed, CASCADE delete)

### Migration Created
✅ `alembic/versions/878310ff008b_add_pipeline_run_id_to_discovery_models.py`

## Code Changes Needed

### Step 2: Seeds (step_02_seeds.py)
- [ ] Update deletion query to filter by `pipeline_run_id`
- [ ] Add `pipeline_run_id` when creating SeedTopics

### Step 3: Expansion (step_03_expansion.py)
- [ ] Update deletion queries to filter by `pipeline_run_id` (Keywords and Topics)
- [ ] Add `pipeline_run_id` when creating Keywords
- [ ] Update queries that load existing data

### Step 6: Clustering (step_06_clustering.py)
- [ ] Update deletion query to filter by `pipeline_run_id`
- [ ] Add `pipeline_run_id` when creating Topics

### Step 7: Prioritization (step_07_prioritization.py)
- [ ] Update queries that load Topics to filter by `pipeline_run_id`

### Step 8: SERP Validation (step_08_serp.py)
- [ ] Update queries that load Topics to filter by `pipeline_run_id`

### Discovery Loop (loop.py)
- [ ] May need updates if it queries Topics/Keywords

## DTOs to Update
- [ ] SeedTopicCreateDTO - Add `pipeline_run_id` field
- [ ] KeywordCreateDTO - Add `pipeline_run_id` field
- [ ] TopicCreateDTO - Add `pipeline_run_id` field

## Testing
- [ ] Run migration: `uv run alembic upgrade head`
- [ ] Start a new discovery run and verify:
  - SeedTopics created with pipeline_run_id
  - Keywords created with pipeline_run_id
  - Topics created with pipeline_run_id
  - No data from previous runs is deleted
  - Within-run iterations still work correctly

## Benefits
✅ Historical data preserved across runs
✅ Each run works in isolation
✅ Database integrity maintained
✅ Easier to debug individual runs
✅ Can analyze/compare runs over time
