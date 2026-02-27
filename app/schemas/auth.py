"""Authentication schemas."""

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    full_name: str | None = None


class UserLogin(BaseModel):
    """Schema for user login."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""

    id: str
    email: str
    full_name: str | None
    email_verified: bool
    is_active: bool

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for refresh token request."""

    refresh_token: str


class EmailVerificationTokenRequest(BaseModel):
    """Schema for verifying email with a token."""

    token: str = Field(min_length=1)


class AuthMessageResponse(BaseModel):
    """Simple auth message response."""

    message: str
