"""Constants for pipeline routes."""

DEFAULT_RUN_LIMIT = 10
MAX_RUN_LIMIT = 50

DISCOVERY_PIPELINE_ALREADY_RUNNING_DETAIL = "Discovery pipeline is already running for this project"
CONTENT_PIPELINE_ALREADY_RUNNING_DETAIL = "Content pipeline is already running for this project"
NO_RUNNING_PIPELINE_DETAIL = "No running pipeline found"
MULTIPLE_RUNNING_PIPELINES_DETAIL = (
    "Multiple pipelines are running. Use run-scoped pause endpoint."
)
NO_PAUSED_PIPELINE_DETAIL = "No paused pipeline found"
PIPELINE_RUN_NOT_FOUND_DETAIL = "Pipeline run not found"
PIPELINE_QUEUE_FULL_DETAIL = "Pipeline queue is full, try again shortly"
