"""Setup pipeline module definitions."""

from __future__ import annotations

from app.services.steps.setup.step_01_setup_project import (
    SetupProjectInput,
    Step01SetupProjectService,
)
from app.services.steps.setup.step_02_brand_core import BrandCoreInput, Step02BrandCoreService
from app.services.steps.setup.step_03_brand_icp import BrandIcpInput, Step03BrandIcpService
from app.services.steps.setup.step_04_brand_assets import (
    BrandAssetsInput,
    Step04BrandAssetsService,
)
from app.services.steps.setup.step_05_brand_visual import (
    BrandVisualInput,
    Step05BrandVisualService,
)

SETUP_LOCAL_STEP_NAMES: dict[int, str] = {
    1: "setup_project",
    2: "brand_core",
    3: "brand_icp",
    4: "brand_assets",
    5: "brand_visual",
}

SETUP_LOCAL_STEP_DEPENDENCIES: dict[int, list[int]] = {
    1: [],
    2: [1],
    3: [2],
    4: [3],
    5: [4],
}

SETUP_LOCAL_TO_SERVICE = {
    1: Step01SetupProjectService,
    2: Step02BrandCoreService,
    3: Step03BrandIcpService,
    4: Step04BrandAssetsService,
    5: Step05BrandVisualService,
}

SETUP_LOCAL_TO_INPUT = {
    1: SetupProjectInput,
    2: BrandCoreInput,
    3: BrandIcpInput,
    4: BrandAssetsInput,
    5: BrandVisualInput,
}

SETUP_DEFAULT_START_STEP = 1
SETUP_DEFAULT_END_STEP = 5
