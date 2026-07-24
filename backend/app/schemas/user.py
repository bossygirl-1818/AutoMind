from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import UserRole, UserStatus


class UserCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    role: UserRole = UserRole.ENGINEER


class UserResponse(BaseModel):
    id: UUID
    full_name: str
    email: EmailStr
    role: UserRole
    status: UserStatus
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class UserLogin(BaseModel):
    """
    Schema used for authenticating an existing AutoMind user.
    """

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """
    JWT token returned after successful authentication.
    """

    access_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    """
    Response returned after a successful login.
    """

    user: UserResponse
    token: TokenResponse