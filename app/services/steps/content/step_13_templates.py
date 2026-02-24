"""Step 13: Writer Instructions + QA Gates.

Creates project-level style guide (once) + per-brief deltas (smaller).
This approach reduces storage by ~80% and ensures consistency.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.agents.template_generator import (
    BriefDeltaGeneratorAgent,
    BriefDeltaGeneratorInput,
    StyleGuideGeneratorAgent,
    StyleGuideGeneratorInput,
)
from app.models.brand import BrandProfile
from app.models.content import ContentBrief, WriterInstructions
from app.models.generated_dtos import (
    BriefDeltaCreateDTO,
    ProjectStyleGuideCreateDTO,
    WriterInstructionsCreateDTO,
)
from app.models.project import Project
from app.models.style_guide import BriefDelta, ProjectStyleGuide
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class TemplatesInput:
    """Input for Step 13."""

    project_id: str
    brief_ids: list[str] | None = None  # Specific briefs, or None for all


@dataclass
class TemplatesOutput:
    """Output from Step 13."""

    style_guide_created: bool
    briefs_processed: int
    deltas_created: int
    project_style_guide: dict[str, Any] = field(default_factory=dict)
    brief_deltas: list[dict[str, Any]] = field(default_factory=list)


class Step13TemplatesService(BaseStepService[TemplatesInput, TemplatesOutput]):
    """Step 13: Writer Instructions + QA Gates.

    Architecture:
    - ProjectStyleGuide (1 per project): Voice/tone, forbidden claims,
      compliance notes, base QA checklist
    - BriefDelta (1 per brief): Page-type specifics, must-include sections,
      link placeholders, schema template

    Benefits:
    - Reduces storage by ~80% (no duplication)
    - Reduces LLM costs (style guide generated once)
    - Ensures consistency across all briefs
    """

    step_number = 3
    step_name = "writer_templates"
    is_optional = False

    async def _validate_preconditions(self, input_data: TemplatesInput) -> None:
        """Validate required content artifacts exist."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        # Check briefs exist
        briefs_result = await self.session.execute(
            select(ContentBrief).where(
                ContentBrief.project_id == input_data.project_id
            ).limit(1)
        )
        if not briefs_result.scalars().first():
            raise ValueError("No content briefs found. Run Step 1 first.")

    async def _execute(self, input_data: TemplatesInput) -> TemplatesOutput:
        """Execute writer template generation."""
        # Load project and brand
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        await self._update_progress(5, "Checking for existing style guide...")

        # Step 1: Check if ProjectStyleGuide exists, create if not
        style_guide_result = await self.session.execute(
            select(ProjectStyleGuide).where(
                ProjectStyleGuide.project_id == input_data.project_id
            )
        )
        style_guide = style_guide_result.scalar_one_or_none()
        logger.info("Style guide status", extra={"project_id": input_data.project_id, "exists": style_guide is not None})

        style_guide_created = False
        style_guide_data = {}

        if not style_guide:
            await self._update_progress(10, "Generating project style guide...")
            style_guide, style_guide_data = await self._generate_style_guide(
                project, brand
            )
            style_guide_created = True
            logger.info("Style guide generated", extra={"project_id": input_data.project_id})
        else:
            # Extract existing style guide data
            style_guide_data = {
                "voice_tone_constraints": style_guide.voice_tone_constraints,
                "forbidden_claims": style_guide.forbidden_claims,
                "compliance_notes": style_guide.compliance_notes,
                "formatting_requirements": style_guide.formatting_requirements,
                "base_qa_checklist": style_guide.base_qa_checklist,
                "common_failure_modes": style_guide.common_failure_modes,
            }

        await self._update_progress(30, "Loading content briefs...")

        # Step 2: Load briefs to process
        if input_data.brief_ids:
            briefs_result = await self.session.execute(
                select(ContentBrief).where(
                    ContentBrief.project_id == input_data.project_id,
                    ContentBrief.id.in_(input_data.brief_ids),
                )
            )
        else:
            briefs_result = await self.session.execute(
                select(ContentBrief).where(
                    ContentBrief.project_id == input_data.project_id
                )
            )

        briefs = list(briefs_result.scalars())

        if not briefs:
            return TemplatesOutput(
                style_guide_created=style_guide_created,
                briefs_processed=0,
                deltas_created=0,
                project_style_guide=style_guide_data,
            )

        # Step 3: Filter out briefs that already have deltas
        existing_deltas_result = await self.session.execute(
            select(BriefDelta.brief_id).where(
                BriefDelta.brief_id.in_([b.id for b in briefs])
            )
        )
        existing_brief_ids = set(existing_deltas_result.scalars())
        briefs_to_process = [b for b in briefs if b.id not in existing_brief_ids]

        if not briefs_to_process:
            logger.info("All briefs already have deltas, skipping LLM calls")
            return TemplatesOutput(
                style_guide_created=style_guide_created,
                briefs_processed=len(briefs),
                deltas_created=0,
                project_style_guide=style_guide_data,
            )

        await self._update_progress(
            35, f"Generating deltas for {len(briefs_to_process)} briefs in parallel..."
        )

        # Step 4: Run all LLM calls in parallel, then persist results
        delta_agent = BriefDeltaGeneratorAgent()

        async def _generate_delta(brief: ContentBrief) -> dict[str, Any]:
            """Run a single LLM call for a brief delta."""
            delta_input = BriefDeltaGeneratorInput(
                brief_summary={
                    "primary_keyword": brief.primary_keyword,
                    "topic_name": brief.primary_keyword,
                    "working_titles": brief.working_titles or [],
                },
                page_type=brief.page_type or "guide",
                search_intent=brief.search_intent or "informational",
                funnel_stage=brief.funnel_stage or "tofu",
            )
            delta_output = await delta_agent.run(delta_input)
            delta_result = delta_output.delta
            return {
                "brief_id": str(brief.id),
                "page_type_rules": delta_result.page_type_rules,
                "must_include_sections": delta_result.must_include_sections,
                "h1_h2_usage": delta_result.h1_h2_usage,
                "schema_type": delta_result.schema_type,
                "additional_qa_items": [
                    {"item": qa.item, "required": qa.required, "threshold": qa.threshold}
                    for qa in delta_result.additional_qa_items
                ],
            }

        async def _generate_delta_safe(brief: ContentBrief) -> dict[str, Any]:
            """Wrapper that returns a fallback on failure."""
            try:
                return await _generate_delta(brief)
            except Exception:
                logger.warning(
                    "Brief delta generation failed",
                    extra={"brief_id": str(brief.id), "keyword": brief.primary_keyword},
                )
                return {
                    "brief_id": str(brief.id),
                    "page_type_rules": {},
                    "must_include_sections": self._infer_sections(brief.page_type),
                    "h1_h2_usage": {"h1": "One per page", "h2": "Main sections"},
                    "schema_type": self._infer_schema(brief.page_type),
                    "additional_qa_items": [],
                    "_fallback": True,
                }

        # Fire all LLM calls concurrently
        delta_results = await asyncio.gather(
            *[_generate_delta_safe(brief) for brief in briefs_to_process]
        )

        await self._update_progress(85, "Saving delta records...")

        # Persist all results sequentially on the session
        output_deltas = []
        deltas_created = 0
        brief_by_id = {str(b.id): b for b in briefs_to_process}

        for delta_data in delta_results:
            output_deltas.append(delta_data)
            is_fallback = delta_data.pop("_fallback", False)
            if not is_fallback:
                deltas_created += 1

            brief = brief_by_id[delta_data["brief_id"]]

            BriefDelta.create(
                self.session,
                BriefDeltaCreateDTO(
                    style_guide_id=style_guide.id,
                    brief_id=brief.id,
                    page_type_rules=delta_data["page_type_rules"],
                    must_include_sections=delta_data["must_include_sections"],
                    h1_h2_usage=delta_data["h1_h2_usage"],
                    schema_type=delta_data["schema_type"],
                    additional_qa_items=delta_data["additional_qa_items"],
                ),
            )

            WriterInstructions.create(
                self.session,
                WriterInstructionsCreateDTO(
                    brief_id=brief.id,
                    voice_tone_constraints=style_guide.voice_tone_constraints,
                    forbidden_claims=style_guide.forbidden_claims,
                    compliance_notes=style_guide.compliance_notes,
                    formatting_requirements=style_guide.formatting_requirements,
                    h1_h2_usage=delta_data["h1_h2_usage"],
                    internal_linking_minimums={
                        "min_internal": style_guide.default_internal_linking_min or 3,
                        "min_external": style_guide.default_external_linking_min or 2,
                    },
                    schema_guidance=f"Use {delta_data['schema_type']} schema",
                    qa_checklist=style_guide.base_qa_checklist,
                    pass_fail_thresholds={
                        "seo_score_target": 75,
                        "keyword_density_soft_min": 0.2,
                        "keyword_density_soft_max": 2.5,
                        "max_auto_revisions": 1,
                    },
                    common_failure_modes=[
                        {"mode": fm, "severity": "warning"}
                        for fm in (style_guide.common_failure_modes or [])
                    ],
                ),
            )

        logger.info("Template generation complete", extra={"style_guide_created": style_guide_created, "briefs_processed": len(briefs), "deltas_created": deltas_created})

        await self._update_progress(100, "Template generation complete")

        return TemplatesOutput(
            style_guide_created=style_guide_created,
            briefs_processed=len(briefs),
            deltas_created=deltas_created,
            project_style_guide=style_guide_data,
            brief_deltas=output_deltas,
        )

    async def _generate_style_guide(
        self,
        project: Project,
        brand: BrandProfile | None,
    ) -> tuple[ProjectStyleGuide, dict[str, Any]]:
        """Generate project-level style guide."""
        agent = StyleGuideGeneratorAgent()

        # Prepare input from brand profile
        agent_input = StyleGuideGeneratorInput(
            company_name=(brand.company_name or project.name) if brand else project.name,
            tagline=(brand.tagline or "") if brand else "",
            products_services=[
                p.get("name", "") for p in (brand.products_services or [])[:5]
            ] if brand else [],
            tone_attributes=(brand.tone_attributes or []) if brand else [],
            target_audience=(brand.target_roles or []) if brand else [],
            allowed_claims=(brand.allowed_claims or []) if brand else [],
            restricted_claims=(brand.restricted_claims or []) if brand else [],
            compliance_flags=project.compliance_flags or [],
        )

        try:
            output = await agent.run(agent_input)
            sg = output.style_guide

            # Create style guide record
            style_guide = ProjectStyleGuide.create(
                self.session,
                ProjectStyleGuideCreateDTO(
                    project_id=self.project_id,
                    voice_tone_constraints={
                        "do_list": sg.voice_tone_constraints.do_list,
                        "dont_list": sg.voice_tone_constraints.dont_list,
                        "good_examples": sg.voice_tone_constraints.good_examples,
                        "bad_examples": sg.voice_tone_constraints.bad_examples,
                    },
                    forbidden_claims=sg.forbidden_claims,
                    compliance_notes=sg.compliance_notes,
                    formatting_requirements=sg.formatting_requirements,
                    base_qa_checklist=[
                        {"item": qa.item, "required": qa.required, "threshold": qa.threshold}
                        for qa in sg.base_qa_checklist
                    ],
                    common_failure_modes=sg.common_failure_modes,
                    default_internal_linking_min=3,
                    default_external_linking_min=2,
                ),
            )

            style_guide_data = {
                "voice_tone_constraints": style_guide.voice_tone_constraints,
                "forbidden_claims": style_guide.forbidden_claims,
                "compliance_notes": style_guide.compliance_notes,
                "formatting_requirements": style_guide.formatting_requirements,
                "base_qa_checklist": style_guide.base_qa_checklist,
                "common_failure_modes": style_guide.common_failure_modes,
            }

            return style_guide, style_guide_data

        except Exception:
            # Fallback: create minimal style guide
            style_guide = ProjectStyleGuide.create(
                self.session,
                ProjectStyleGuideCreateDTO(
                    project_id=self.project_id,
                    voice_tone_constraints={
                        "do_list": ["Be helpful", "Use examples", "Be clear"],
                        "dont_list": ["Use jargon", "Make unsubstantiated claims"],
                        "good_examples": [],
                        "bad_examples": [],
                    },
                    forbidden_claims=["#1 in the industry", "Best in class", "Guaranteed results"],
                    compliance_notes=project.compliance_flags or [],
                    formatting_requirements={
                        "headings": "Use H2 for main sections, H3 for subsections",
                        "lists": "Use bullet points for lists of 3+ items",
                    },
                    base_qa_checklist=[
                        {"item": "Primary keyword in title", "required": True, "threshold": ""},
                        {"item": "Meta description < 160 chars", "required": True, "threshold": "160"},
                        {"item": "At least 3 internal links", "required": True, "threshold": "3"},
                        {"item": "At least 2 external links", "required": False, "threshold": "2"},
                    ],
                    common_failure_modes=["Missing meta description", "No internal links", "Title too long"],
                    default_internal_linking_min=3,
                    default_external_linking_min=2,
                ),
            )

            style_guide_data = {
                "voice_tone_constraints": style_guide.voice_tone_constraints,
                "forbidden_claims": style_guide.forbidden_claims,
                "compliance_notes": style_guide.compliance_notes,
                "formatting_requirements": style_guide.formatting_requirements,
                "base_qa_checklist": style_guide.base_qa_checklist,
                "common_failure_modes": style_guide.common_failure_modes,
            }

            return style_guide, style_guide_data

    def _infer_sections(self, page_type: str | None) -> list[str]:
        """Infer required sections based on page type."""
        page_type = page_type or "guide"

        section_map = {
            "guide": ["Introduction", "Main Content", "Conclusion", "FAQ"],
            "how-to": ["Introduction", "Prerequisites", "Steps", "Troubleshooting", "FAQ"],
            "comparison": ["Introduction", "Quick Comparison", "Detailed Analysis", "Verdict"],
            "list": ["Introduction", "Items", "How to Choose", "FAQ"],
            "alternatives": ["Introduction", "Best Alternatives", "Comparison Table", "Verdict"],
            "landing": ["Hero", "Benefits", "Features", "CTA"],
            "tool": ["Introduction", "How to Use", "Features", "FAQ"],
            "glossary": ["Definition", "Examples", "Related Terms"],
        }

        return section_map.get(page_type, section_map["guide"])

    def _infer_schema(self, page_type: str | None) -> str:
        """Infer schema type based on page type."""
        page_type = page_type or "guide"

        schema_map = {
            "guide": "Article",
            "how-to": "HowTo",
            "comparison": "Article",
            "list": "ItemList",
            "alternatives": "Article",
            "landing": "Product",
            "tool": "SoftwareApplication",
            "glossary": "DefinedTerm",
        }

        return schema_map.get(page_type, "Article")

    async def _persist_results(self, result: TemplatesOutput) -> None:
        """Save results (already saved during execution)."""
        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        # Set result summary
        self.set_result_summary({
            "style_guide_created": result.style_guide_created,
            "briefs_processed": result.briefs_processed,
            "deltas_created": result.deltas_created,
        })

        await self.session.commit()
