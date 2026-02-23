"""Step 3: Expand and merge ICP recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


class Step03BrandIcpService(BaseStepService[BrandIcpInput, BrandIcpOutput]):
    """Step 3: Generate ICP niche recommendations and merge audience fields."""

    step_number = 3
    step_name = "brand_icp"
    is_optional = False

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
        try:
            recommender = ICPRecommenderAgent()
            recommendation = await recommender.run(
                ICPRecommenderInput(
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
            )
            icp_niches = [niche.model_dump() for niche in recommendation.suggested_niches]
        except Exception:
            logger.exception(
                "ICP niche recommendation failed; using extracted website audience only",
                extra={"project_id": input_data.project_id},
            )

        merged_target_audience = merge_target_audience(
            extracted_target_audience=extracted_target_audience,
            suggested_icp_niches=icp_niches,
        )

        await self._update_progress(100, "ICP enrichment complete")

        return BrandIcpOutput(
            suggested_icp_niches=icp_niches,
            target_audience=merged_target_audience,
        )

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
            }
        )

        await self.session.commit()
