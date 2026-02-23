"""Custom exception classes for the application."""

from typing import Any


class DonkeySEOError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# Authentication Errors
class AuthenticationError(DonkeySEOError):
    """Authentication failed."""

    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password."""

    def __init__(self) -> None:
        super().__init__("Invalid email or password")


class InvalidTokenError(AuthenticationError):
    """Invalid or expired token."""

    def __init__(self, message: str = "Invalid or expired token") -> None:
        super().__init__(message)


class UserNotFoundError(AuthenticationError):
    """User not found."""

    def __init__(self, user_id: str | None = None) -> None:
        message = f"User not found: {user_id}" if user_id else "User not found"
        super().__init__(message)


class UserAlreadyExistsError(DonkeySEOError):
    """User with this email already exists."""

    def __init__(self, email: str) -> None:
        super().__init__(f"User with email {email} already exists")


# Project Errors
class ProjectNotFoundError(DonkeySEOError):
    """Project not found."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project not found: {project_id}")


class ProjectAccessDeniedError(DonkeySEOError):
    """User does not have access to this project."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Access denied to project: {project_id}")


# Pipeline Errors
class PipelineError(DonkeySEOError):
    """Base class for pipeline errors."""

    pass


class StepNotFoundError(PipelineError):
    """Pipeline step not found."""

    def __init__(self, step_number: int) -> None:
        super().__init__(f"Step {step_number} not found")


class StepPreconditionError(PipelineError):
    """Step preconditions not met."""

    def __init__(self, step_number: int, message: str) -> None:
        super().__init__(f"Step {step_number} precondition failed: {message}")


class StepExecutionError(PipelineError):
    """Error during step execution."""

    def __init__(self, step_number: int, message: str) -> None:
        super().__init__(f"Step {step_number} execution failed: {message}")


class PipelineAlreadyRunningError(PipelineError):
    """Pipeline is already running for this project."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Pipeline already running for project: {project_id}")


class PipelineDelayedResumeRequested(PipelineError):
    """Pipeline should pause and be resumed by the worker after a delay."""

    def __init__(
        self,
        *,
        delay_seconds: float,
        reason: str = "delayed_resume_requested",
    ) -> None:
        self.delay_seconds = max(0.1, float(delay_seconds))
        self.reason = reason
        super().__init__(
            f"Delayed resume requested in {self.delay_seconds:.1f}s: {self.reason}"
        )


# External API Errors
class ExternalAPIError(DonkeySEOError):
    """Error calling external API."""

    def __init__(self, api_name: str, message: str) -> None:
        super().__init__(f"{api_name} API error: {message}")


class RateLimitExceededError(ExternalAPIError):
    """Rate limit exceeded for external API."""

    def __init__(self, api_name: str) -> None:
        super().__init__(api_name, "Rate limit exceeded")


class APIKeyMissingError(ExternalAPIError):
    """API key not configured."""

    def __init__(self, api_name: str) -> None:
        super().__init__(api_name, "API key not configured")


# Data Errors
class KeywordNotFoundError(DonkeySEOError):
    """Keyword not found."""

    def __init__(self, keyword_id: str) -> None:
        super().__init__(f"Keyword not found: {keyword_id}")


class TopicNotFoundError(DonkeySEOError):
    """Topic not found."""

    def __init__(self, topic_id: str) -> None:
        super().__init__(f"Topic not found: {topic_id}")


class ContentBriefNotFoundError(DonkeySEOError):
    """Content brief not found."""

    def __init__(self, brief_id: str) -> None:
        super().__init__(f"Content brief not found: {brief_id}")


# Validation Errors
class ValidationError(DonkeySEOError):
    """Data validation failed."""

    pass
