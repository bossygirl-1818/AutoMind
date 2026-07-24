from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.database.session import get_db
from app.schemas.user import (
    LoginResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.services.auth_service import (
    authenticate_user,
    create_user,
    get_user_by_email,
    update_last_login,
)

from app.api.deps.auth import get_current_user
from app.models.user import User


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new AutoMind user",
    responses={
        status.HTTP_201_CREATED: {
            "description": "User registered successfully.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "A user with this email address already exists.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The submitted registration data is invalid.",
        },
    },
)
def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
) -> UserResponse:
    """
    Register a new AutoMind user.
    """

    existing_user = get_user_by_email(
        db=db,
        email=user_data.email,
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email address already exists.",
        )

    return create_user(
        db=db,
        user_data=user_data,
    )


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate an AutoMind user",
    responses={
        status.HTTP_200_OK: {
            "description": "Authentication successful.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid email or password.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The submitted login data is invalid.",
        },
    },
)
def login_user(
    credentials: UserLogin,
    db: Session = Depends(get_db),
) -> LoginResponse:
    """
    Authenticate a user and issue a JWT access token.
    """

    user = authenticate_user(
        db=db,
        email=credentials.email,
        password=credentials.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = update_last_login(
        db=db,
        user=user,
    )

    access_token = create_access_token(
        subject=str(user.id),
        additional_claims={
            "email": user.email,
            "role": user.role.value,
        },
    )

    return LoginResponse(
        user=UserResponse.model_validate(user),
        token=TokenResponse(
            access_token=access_token,
            token_type="bearer",
        ),
    )
@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the authenticated user's profile",
    responses={
        status.HTTP_200_OK: {
            "description": "Authenticated user profile returned successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "The authenticated user account is inactive.",
        },
    },
)
def get_authenticated_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Return the profile of the currently authenticated AutoMind user.

    A valid JWT access token must be supplied through the Authorization
    header using the Bearer authentication scheme.
    """

    return current_user