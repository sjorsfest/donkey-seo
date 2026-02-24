"""Utilities to build deterministic image-generation prompts."""

from __future__ import annotations

from typing import Any, Callable

_REQUIRED_VARIABLES = [
    "article_topic",
    "audience",
    "intent",
    "visual_goal",
    "brand_voice",
    "asset_refs",
]

_DEFAULT_TEMPLATE = (
    "Create a brand-aligned image for '{article_topic}' targeting {audience}. "
    "Intent: {intent}. Visual goal: {visual_goal}. "
    "Brand voice: {brand_voice}. Asset references: {asset_refs}."
)

_DEFAULT_COMPONENT_TEMPLATE = (
    "Render a brand-aligned component scene for '{article_topic}' targeting {audience}. "
    "Intent: {intent}. Visual goal: {visual_goal}. "
    "Brand voice: {brand_voice}. Asset references: {asset_refs}."
)


class ImagePromptBuilder:
    """Compile strict prompt contracts into deterministic prompt strings."""

    def build_prompt_payload(
        self,
        *,
        visual_style_guide: dict[str, Any] | None,
        visual_prompt_contract: dict[str, Any] | None,
        article_topic: str,
        audience: str,
        intent: str,
        visual_goal: str,
        brand_voice: str,
        assets: list[dict[str, Any]] | None,
        sign_asset_url: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Build a deterministic prompt payload and optional signed asset URLs."""
        contract = visual_prompt_contract or {}
        template = str(contract.get("template") or _DEFAULT_TEMPLATE)
        component_template = str(contract.get("component_template") or _DEFAULT_COMPONENT_TEMPLATE)

        asset_refs_payload = self._build_asset_refs(
            assets=assets or [],
            sign_asset_url=sign_asset_url,
        )
        render_variables = {
            "article_topic": self._normalize(article_topic),
            "audience": self._normalize(audience),
            "intent": self._normalize(intent),
            "visual_goal": self._normalize(visual_goal),
            "brand_voice": self._normalize(brand_voice),
            "asset_refs": "; ".join(asset_refs_payload["labels"]),
        }

        prompt = self._render_template(template, render_variables)
        prompt = self._append_style_rules(prompt, visual_style_guide or {})
        component_prompt = self._render_template(component_template, render_variables)

        style_guide = visual_style_guide or {}
        component_render_context = self._build_component_render_context(
            style_guide=style_guide,
            contract=contract,
            component_prompt=component_prompt,
        )

        return {
            "prompt": prompt,
            "required_variables": _REQUIRED_VARIABLES,
            "render_variables": render_variables,
            "asset_refs": asset_refs_payload["assets"],
            "component_render_context": component_render_context,
        }

    def _build_asset_refs(
        self,
        *,
        assets: list[dict[str, Any]],
        sign_asset_url: Callable[[str], str] | None,
    ) -> dict[str, Any]:
        ordered_assets = sorted(
            assets,
            key=lambda item: (
                -float(item.get("role_confidence") or 0.0),
                str(item.get("object_key") or ""),
            ),
        )

        labels: list[str] = []
        payload_assets: list[dict[str, Any]] = []
        for asset in ordered_assets:
            object_key = str(asset.get("object_key") or "").strip()
            if not object_key:
                continue

            role = str(asset.get("role") or "reference")
            label = f"{role}:{object_key}"
            labels.append(label)

            payload_asset = {
                "asset_id": str(asset.get("asset_id") or ""),
                "object_key": object_key,
                "role": role,
                "role_confidence": float(asset.get("role_confidence") or 0.0),
            }
            if sign_asset_url is not None:
                payload_asset["signed_url"] = sign_asset_url(object_key)
            payload_assets.append(payload_asset)

        return {
            "labels": labels,
            "assets": payload_assets,
        }

    @staticmethod
    def _render_template(template: str, variables: dict[str, str]) -> str:
        rendered = template
        for name in _REQUIRED_VARIABLES:
            placeholder = "{" + name + "}"
            rendered = rendered.replace(placeholder, variables[name])
        return " ".join(rendered.split())

    @staticmethod
    def _append_style_rules(prompt: str, style_guide: dict[str, Any]) -> str:
        style_segments: list[str] = []
        for key in (
            "contrast_rules",
            "composition_rules",
            "subject_rules",
            "camera_lighting_rules",
            "logo_usage_rules",
            "negative_rules",
            "accessibility_rules",
        ):
            rules = style_guide.get(key)
            if isinstance(rules, list) and rules:
                normalized_rules = [
                    " ".join(str(rule).split())
                    for rule in rules
                    if str(rule).strip()
                ]
                if normalized_rules:
                    style_segments.append(f"{key}: {', '.join(normalized_rules)}")

        if not style_segments:
            return prompt

        return f"{prompt} {' | '.join(style_segments)}"

    @staticmethod
    def _build_component_render_context(
        *,
        style_guide: dict[str, Any],
        contract: dict[str, Any],
        component_prompt: str,
    ) -> dict[str, Any]:
        render_modes = [
            str(item).strip()
            for item in (contract.get("render_modes") or [])
            if str(item).strip()
        ]
        enabled = bool(
            "component_render" in [mode.casefold() for mode in render_modes]
            or style_guide.get("component_recipes")
        )
        return {
            "enabled": enabled,
            "render_modes": render_modes,
            "component_prompt": component_prompt,
            "component_render_targets": [
                str(item).strip()
                for item in (contract.get("component_render_targets") or [])
                if str(item).strip()
            ],
            "component_fallback_rules": [
                str(item).strip()
                for item in (contract.get("component_fallback_rules") or [])
                if str(item).strip()
            ],
            "design_tokens": style_guide.get("design_tokens") or {},
            "component_style_rules": style_guide.get("component_style_rules") or [],
            "component_layout_rules": style_guide.get("component_layout_rules") or [],
            "component_recipes": style_guide.get("component_recipes") or [],
            "component_negative_rules": style_guide.get("component_negative_rules") or [],
        }

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(str(value).split())
