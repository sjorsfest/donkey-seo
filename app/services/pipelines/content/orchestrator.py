"""Content pipeline module definitions."""

from __future__ import annotations

from app.services.steps.content.step_12_brief import BriefInput, Step12BriefService
from app.services.steps.content.step_13_templates import Step13TemplatesService, TemplatesInput
from app.services.steps.content.step_14_article_writer import (
    ArticleWriterInput,
    Step14ArticleWriterService,
)

CONTENT_LOCAL_STEP_NAMES: dict[int, str] = {
    1: "content_brief",
    2: "writer_templates",
    3: "article_generation",
}

CONTENT_LOCAL_STEP_DEPENDENCIES: dict[int, list[int]] = {
    1: [],
    2: [1],
    3: [2],
}

CONTENT_LOCAL_TO_SERVICE = {
    1: Step12BriefService,
    2: Step13TemplatesService,
    3: Step14ArticleWriterService,
}

CONTENT_LOCAL_TO_INPUT = {
    1: BriefInput,
    2: TemplatesInput,
    3: ArticleWriterInput,
}

CONTENT_DEFAULT_START_STEP = 1
CONTENT_DEFAULT_END_STEP = 3
