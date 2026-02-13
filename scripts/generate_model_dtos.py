"""Generate typed dataclass DTOs from SQLAlchemy models.

This script inspects SQLAlchemy mappers and emits static dataclasses:
- `<Model>Row`: typed DTO for read/return boundaries
- `<Model>CreateDTO`: typed DTO for create/persist boundaries

Generated file: app/models/generated_dtos.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Mapper

from app.models import Base  # noqa: F401  # Import side-effect registers models
from app.models.base import Base as SA_Base
from app.models.base import StringUUID

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "app" / "models" / "generated_dtos.py"
SKIP_CREATE_FIELDS = {"id", "created_at", "updated_at"}


def _extract_mapped_inner(annotation: Any) -> str | None:
    """Extract inner type from `Mapped[...]` annotation."""
    if isinstance(annotation, str):
        text = annotation.strip()
        if text.startswith("Mapped[") and text.endswith("]"):
            return text[len("Mapped[") : -1]
        return text
    return None


def _annotation_for_field(mapper: Mapper[Any], field_name: str) -> str:
    """Resolve Python type annotation string for a mapped column field."""
    cls = mapper.class_

    for base_cls in cls.__mro__:
        annotations = getattr(base_cls, "__annotations__", {})
        if field_name in annotations:
            mapped_inner = _extract_mapped_inner(annotations[field_name])
            if mapped_inner:
                return mapped_inner

    column = mapper.columns[field_name]
    col_type = column.type

    if isinstance(col_type, StringUUID):
        return "str"

    try:
        py_type = col_type.python_type
    except Exception:
        return "Any"

    if py_type is str:
        return "str"
    if py_type is int:
        return "int"
    if py_type is float:
        return "float"
    if py_type is bool:
        return "bool"
    if py_type is dict:
        return "dict[str, Any]"
    if py_type is list:
        return "list[Any]"
    if py_type.__name__ == "datetime":
        return "datetime"

    return "Any"


def _contains_none(type_expr: str) -> bool:
    return "None" in type_expr or "Optional[" in type_expr


def _add_none(type_expr: str) -> str:
    if _contains_none(type_expr):
        return type_expr
    return f"{type_expr} | None"


def _is_required_create_field(mapper: Mapper[Any], field_name: str) -> bool:
    """Return True if this create field must be provided by caller."""
    column = mapper.columns[field_name]
    if column.primary_key:
        return False
    if column.nullable:
        return False
    if column.default is not None:
        return False
    if column.server_default is not None:
        return False
    return True


def _should_drop_none_on_create(mapper: Mapper[Any], field_name: str) -> bool:
    """Drop `None` so DB/ORM defaults can kick in for non-null columns."""
    column = mapper.columns[field_name]
    return (
        not column.nullable
        and (column.default is not None or column.server_default is not None)
    )


def _render_row_dto(mapper: Mapper[Any]) -> list[str]:
    model_name = mapper.class_.__name__
    dto_name = f"{model_name}Row"

    lines = [
        "@dataclass(slots=True)",
        f"class {dto_name}:",
        f'    """Read DTO for `{model_name}`."""',
        "",
    ]

    for field_name in mapper.columns.keys():
        type_expr = _annotation_for_field(mapper, field_name)
        column = mapper.columns[field_name]
        if column.nullable:
            type_expr = _add_none(type_expr)
        lines.append(f"    {field_name}: {type_expr}")

    lines.extend(
        [
            "",
            "    @classmethod",
            f'    def from_model(cls, model: Any) -> "{dto_name}":',
            "        return cls(",
        ]
    )

    for field_name in mapper.columns.keys():
        lines.append(f"            {field_name}=model.{field_name},")

    lines.extend(
        [
            "        )",
            "",
        ]
    )
    return lines


def _render_create_dto(mapper: Mapper[Any]) -> list[str]:
    model_name = mapper.class_.__name__
    dto_name = f"{model_name}CreateDTO"

    create_fields = [
        field_name
        for field_name in mapper.columns.keys()
        if field_name not in SKIP_CREATE_FIELDS
    ]
    required_fields = [
        field_name
        for field_name in create_fields
        if _is_required_create_field(mapper, field_name)
    ]
    optional_fields = [
        field_name
        for field_name in create_fields
        if field_name not in required_fields
    ]
    drop_none_fields = [
        field_name
        for field_name in optional_fields
        if _should_drop_none_on_create(mapper, field_name)
    ]

    lines = [
        "@dataclass(slots=True)",
        f"class {dto_name}:",
        f'    """Create DTO for `{model_name}`."""',
        "",
    ]

    for field_name in required_fields:
        type_expr = _annotation_for_field(mapper, field_name)
        lines.append(f"    {field_name}: {type_expr}")

    for field_name in optional_fields:
        type_expr = _annotation_for_field(mapper, field_name)
        type_expr = _add_none(type_expr)
        lines.append(f"    {field_name}: {type_expr} = None")

    if not create_fields:
        lines.append("    pass")
        lines.append("")
        return lines

    lines.extend(
        [
            "",
            "    _DROP_NONE_FIELDS: ClassVar[set[str]] = {",
        ]
    )
    for field_name in drop_none_fields:
        lines.append(f'        "{field_name}",')
    if drop_none_fields:
        lines.append("    }")
    else:
        lines[-1] = "    _DROP_NONE_FIELDS: ClassVar[set[str]] = set()"

    lines.extend(
        [
            "",
            "    def to_orm_kwargs(self) -> dict[str, Any]:",
            "        payload = asdict(self)",
            "        for key in self._DROP_NONE_FIELDS:",
            "            if payload.get(key) is None:",
            "                payload.pop(key, None)",
            "        return payload",
            "",
        ]
    )
    return lines


def _render_patch_dto(mapper: Mapper[Any]) -> list[str]:
    model_name = mapper.class_.__name__
    dto_name = f"{model_name}PatchDTO"

    patch_fields = [
        field_name
        for field_name in mapper.columns.keys()
        if field_name not in SKIP_CREATE_FIELDS
    ]

    lines = [
        "@dataclass(slots=True)",
        f"class {dto_name}:",
        f'    """Sparse patch DTO for `{model_name}`."""',
        "",
    ]

    for field_name in patch_fields:
        type_expr = _annotation_for_field(mapper, field_name)
        type_expr = _add_none(type_expr)
        lines.append(f"    {field_name}: {type_expr} = None")

    lines.extend(
        [
            "    _provided_fields: set[str] = field(",
            "        default_factory=set,",
            "        repr=False,",
            "        compare=False,",
            "    )",
            "",
            "    @classmethod",
            f'    def from_partial(cls, payload: dict[str, Any]) -> "{dto_name}":',
            "        obj = cls(**payload)",
            "        obj._provided_fields = set(payload.keys())",
            "        return obj",
            "",
            "    def to_patch_dict(self) -> dict[str, Any]:",
            "        payload = asdict(self)",
            "        payload.pop(\"_provided_fields\", None)",
            "        return {",
            "            key: value",
            "            for key, value in payload.items()",
            "            if key in self._provided_fields",
            "        }",
            "",
        ]
    )

    return lines


def build_output() -> str:
    mappers = sorted(
        SA_Base.registry.mappers,
        key=lambda mapper: mapper.class_.__name__,
    )

    lines: list[str] = [
        '"""Auto-generated dataclass DTOs from SQLAlchemy models.',
        "",
        "Generated by: scripts/generate_model_dtos.py",
        "Do not edit manually; regenerate instead.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import asdict, dataclass, field",
        "from datetime import datetime",
        "from typing import Any, ClassVar",
        "",
        "",
    ]

    exported: list[str] = []
    for mapper in mappers:
        model_name = mapper.class_.__name__
        exported.append(f"{model_name}Row")
        exported.append(f"{model_name}CreateDTO")
        exported.append(f"{model_name}PatchDTO")
        lines.extend(_render_row_dto(mapper))
        lines.extend(_render_create_dto(mapper))
        lines.extend(_render_patch_dto(mapper))

    lines.append("__all__ = [")
    for name in exported:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.write_text(build_output(), encoding="utf-8")
    print(f"Generated DTOs: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
