"""Reusable API dependencies shared across v1 routes."""

from app.api.v1.dependencies.project_access import get_user_project

__all__ = ["get_user_project"]
