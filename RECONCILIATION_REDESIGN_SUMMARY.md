# Discovery Reconciliation Redesign

## Summary

The discovery reconciliation logic has been redesigned to **start new discovery runs** when future scheduled article counts are low, not just resume paused runs.

## What Changed

### Before
- Only checked for **paused** discovery runs with auto-halt reason
- Only **resumed** those paused runs when article count dropped below threshold
- Ignored projects with **completed** discovery runs

### After
- Checks **all paid projects** for low future scheduled article counts
- For each project needing more articles:
  - **Resumes** paused auto-halted run if one exists
  - **Starts NEW discovery run** if no paused run exists
  - Skips if already-running discovery pipeline exists
- Focuses on **future scheduled articles** (within configured window)

## Key Insight: Built-in Deduplication

The content pipeline automatically prevents duplicate articles:

- [step_12_brief.py](app/services/steps/content/step_12_brief.py) loads ALL existing ContentBriefs for the project
- Checks new topics against existing briefs using semantic similarity (82% threshold)
- Skips topics that are semantically similar to already-created briefs

**This means restarting discovery runs is safe** - only NEW, non-duplicate topics will get briefs created.

## Modified Files

### 1. [app/services/discovery_pipeline_halt.py](app/services/discovery_pipeline_halt.py)

**Function**: `reconcile_discovery_auto_halted_runs()`

**Changes**:
- Now queries ALL paid projects (not just paused runs)
- For each project:
  - Checks if scheduled articles < threshold
  - Skips if discovery already running
  - Resumes paused run OR starts new run
- Added comprehensive logging for both resume and start actions

**Metrics**:
- Updated payload to use `started_or_resumed_runs` (includes both counts)

### 2. [app/workers/discovery_reconciliation_worker.py](app/workers/discovery_reconciliation_worker.py)

**Function**: `run_reconciliation_sweep()`

**Changes**:
- Updated variable names from `resumed` to `started_or_resumed`
- Updated log messages to reflect both actions

## Configuration

The reconciliation uses existing settings from `app/config.py`:

```python
discovery_pipeline_halt_threshold: int = 10  # Min articles needed
discovery_pipeline_halt_window_days: int = 60  # Days to look ahead
```

## Behavior

### Daily Reconciliation Sweep

1. **On startup**: Runs immediately (ensures deployments trigger reconciliation)
2. **Scheduled**: Runs daily at configured hour (default: 00:00 UTC)
3. **For each paid project**:
   - Count scheduled articles in next N days
   - If count < threshold:
     - Resume paused auto-halted run, OR
     - Start NEW discovery run
   - Skip if discovery already running

### Example Scenario

**Project**: Has 1 article scheduled (threshold: 10)

**Before**: Nothing happened (no paused run exists)

**After**: New discovery run starts automatically
- Generates new topics
- Dispatches to content pipeline
- Content pipeline skips any semantically duplicate topics
- Only creates briefs for NEW topics

## Testing Notes

### Unit Tests Need Updates

[tests/unit/test_discovery_pipeline_halt.py](tests/unit/test_discovery_pipeline_halt.py) needs updating:

- `test_reconcile_discovery_auto_halted_runs_enqueues_resume` expects old query pattern
- New implementation queries projects first, then checks for paused runs
- Test mocks need to be restructured

### Manual Testing

1. Deploy the changes
2. Check logs for reconciliation sweep:
   ```bash
   sudo journalctl -u donkeyseo-reconciliation-worker -f
   ```
3. Verify projects with low article counts get discovery runs started
4. Check that duplicate articles are NOT created

## Deployment

The changes are backward-compatible and will deploy automatically via [deploy.sh](deploy.sh).

No database migrations needed.

## Rollback

If needed, revert [app/services/discovery_pipeline_halt.py](app/services/discovery_pipeline_halt.py) to only query paused runs (previous behavior).

The reconciliation worker will continue to function - it just won't start new runs, only resume paused ones.
