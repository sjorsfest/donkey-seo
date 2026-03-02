"""Step 3: Expand and merge ICP recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.agents.icp_recommender import ICPRecommenderAgent, ICPRecommenderInput
from app.models.brand import BrandProfile
from app.models.generated_dtos import BrandProfilePatchDTO
from app.models.project import Project
from app.services.steps.base_step import BaseStepService
from app.services.steps.setup.brand_shared import merge_target_audience

logger = logging.getLogger(__name__)


@dataclass
class BrandIcpInput:
    """Input for setup step 3."""

    project_id: str


@dataclass
class BrandIcpOutput:
    """Output for setup step 3."""

    suggested_icp_niches: list[dict[str, Any]]
    target_audience: dict[str, list[str]]
    recommendation_attempts: int = 1
    recommendation_warnings: list[str] = field(default_factory=list)


class Step03BrandIcpService(BaseStepService[BrandIcpInput, BrandIcpOutput]):
    """Step 3: Generate ICP niche recommendations and merge audience fields."""

    step_number = 3
    step_name = "brand_icp"
    is_optional = False
    _MAX_RECOMMENDATION_ATTEMPTS = 3

    @staticmethod
    def _has_non_empty_strings(values: list[Any]) -> bool:
        return any(str(value).strip() for value in values)

    @classmethod
    def _is_actionable_niche(cls, niche: dict[str, Any]) -> bool:
        for key in (
            "target_roles",
            "target_industries",
            "company_sizes",
            "primary_pains",
            "desired_outcomes",
            "likely_objections",
            "objections",
        ):
            values = niche.get(key, [])
            if isinstance(values, list) and cls._has_non_empty_strings(values):
                return True
        return False

    async def _validate_preconditions(self, input_data: BrandIcpInput) -> None:
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = result.scalar_one_or_none()
        if not brand:
            raise ValueError("Brand profile not found. Run setup step 2 first.")

    async def _execute(self, input_data: BrandIcpInput) -> BrandIcpOutput:
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one()

        extracted_target_audience = {
            "target_roles": [str(item) for item in list(brand.target_roles or [])],
            "target_industries": [
                str(item)
                for item in list(brand.target_industries or [])
            ],
            "company_sizes": [str(item) for item in list(brand.company_sizes or [])],
            "primary_pains": [str(item) for item in list(brand.primary_pains or [])],
            "desired_outcomes": [str(item) for item in list(brand.desired_outcomes or [])],
            "objections": [str(item) for item in list(brand.objections or [])],
        }

        await self._update_progress(20, "Expanding ICP opportunities with AI...")
        icp_niches: list[dict[str, Any]] = []
        recommendation_attempts = 0
        recommendation_warnings: list[str] = []
        recommender = ICPRecommenderAgent()
        recommender_input = ICPRecommenderInput(
            company_name=brand.company_name or "Company",
            tagline=brand.tagline,
            products_services=[
                {
                    "name": product.get("name"),
                    "description": product.get("description"),
                    "category": product.get("category"),
                    "target_audience": product.get("target_audience"),
                    "core_benefits": product.get("core_benefits"),
                }
                for product in list(brand.products_services or [])
                if isinstance(product, dict)
            ],
            unique_value_props=list(brand.unique_value_props or []),
            differentiators=list(brand.differentiators or []),
            current_target_audience=extracted_target_audience,
        )

        for attempt in range(1, self._MAX_RECOMMENDATION_ATTEMPTS + 1):
            recommendation_attempts = attempt
            try:
                recommendation = await recommender.run(recommender_input)
            except Exception as exc:
                recommendation_warnings.append(
                    f"attempt_{attempt}:agent_error:{exc.__class__.__name__}"
                )
                if attempt < self._MAX_RECOMMENDATION_ATTEMPTS:
                    logger.warning(
                        "ICP niche recommendation attempt failed; retrying",
                        extra={
                            "project_id": input_data.project_id,
                            "attempt": attempt,
                            "max_attempts": self._MAX_RECOMMENDATION_ATTEMPTS,
                        },
                    )
                    continue

                logger.exception(
                    "ICP niche recommendation failed after max retries; continuing with extracted audience only",
                    extra={
                        "project_id": input_data.project_id,
                        "attempts": self._MAX_RECOMMENDATION_ATTEMPTS,
                    },
                )
                break

            icp_niches = [
                niche
                for niche in (
                    niche.model_dump()
                    for niche in recommendation.suggested_niches
                )
                if self._is_actionable_niche(niche)
            ]
            if icp_niches:
                break

            recommendation_warnings.append("attempt_%d:low_signal:no_actionable_niches" % attempt)
            if attempt < self._MAX_RECOMMENDATION_ATTEMPTS:
                logger.warning(
                    "ICP niche recommendation returned no actionable niches; retrying",
                    extra={
                        "project_id": input_data.project_id,
                        "attempt": attempt,
                        "max_attempts": self._MAX_RECOMMENDATION_ATTEMPTS,
                    },
                )
                continue

            logger.warning(
                "ICP niche recommendation produced no actionable niches after max retries; continuing with extracted audience only",
                extra={
                    "project_id": input_data.project_id,
                    "attempts": self._MAX_RECOMMENDATION_ATTEMPTS,
                },
            )

        merged_target_audience = merge_target_audience(
            extracted_target_audience=extracted_target_audience,
            suggested_icp_niches=icp_niches,
        )

        await self._update_progress(100, "ICP enrichment complete")

        return BrandIcpOutput(
            suggested_icp_niches=icp_niches,
            target_audience=merged_target_audience,
            recommendation_attempts=recommendation_attempts,
            recommendation_warnings=recommendation_warnings,
        )

    async def _validate_output(self, result: BrandIcpOutput, input_data: BrandIcpInput) -> None:
        return None

    async def _persist_results(self, result: BrandIcpOutput) -> None:
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == self.project_id)
        )
        brand = brand_result.scalar_one()

        brand.patch(
            self.session,
            BrandProfilePatchDTO.from_partial(
                {
                    "target_roles": result.target_audience.get("target_roles", []),
                    "target_industries": result.target_audience.get(
                        "target_industries", []
                    ),
                    "company_sizes": result.target_audience.get("company_sizes", []),
                    "primary_pains": result.target_audience.get("primary_pains", []),
                    "desired_outcomes": result.target_audience.get(
                        "desired_outcomes", []
                    ),
                    "objections": result.target_audience.get("objections", []),
                    "suggested_icp_niches": result.suggested_icp_niches,
                }
            ),
        )

        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        self.set_result_summary(
            {
                "icp_niches_count": len(result.suggested_icp_niches),
                "target_roles_count": len(result.target_audience.get("target_roles", [])),
                "target_industries_count": len(
                    result.target_audience.get("target_industries", [])
                ),
                "recommendation_attempts": result.recommendation_attempts,
                "recommendation_warnings_count": len(result.recommendation_warnings),
                "recommendation_warnings": result.recommendation_warnings,
            }
        )

        await self.session.commit()
