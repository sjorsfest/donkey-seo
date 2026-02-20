"""Setup pipeline module definitions."""

from __future__ import annotations

from app.services.steps.discovery.step_00_setup import SetupInput, Step00SetupService
from app.services.steps.discovery.step_01_brand import BrandInput, Step01BrandService

SETUP_LOCAL_STEP_NAMES: dict[int, str] = {
    0: "setup",
    1: "brand_profile",
}

SETUP_LOCAL_STEP_DEPENDENCIES: dict[int, list[int]] = {
    0: [],
    1: [0],
}

SETUP_LOCAL_TO_SERVICE = {
    0: Step00SetupService,
    1: Step01BrandService,
}

SETUP_LOCAL_TO_INPUT = {
    0: SetupInput,
    1: BrandInput,
}

SETUP_DEFAULT_START_STEP = 0
SETUP_DEFAULT_END_STEP = 1
