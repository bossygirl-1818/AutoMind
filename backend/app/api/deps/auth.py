from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.database.session import get_db
from app.models.user import User
from app.services.auth_service import get_user_by_email


bearer_scheme = HTTPBearer(
    scheme_name="BearerAuth",
    description="Enter the JWT access token received from the login endpoint.",
    auto_error=False,
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Authenticate the current request using a JWT bearer token.

    The token is extracted from the Authorization header, decoded,
    validated, and matched to an active user in PostgreSQL.
    """

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise credentials_exception

    if credentials.scheme.lower() != "bearer":
        raise credentials_exception

    payload = decode_access_token(credentials.credentials)

    if payload is None:
        raise credentials_exception

    user_email = payload.get("email")

    if not user_email:
        raise credentials_exception

    user = get_user_by_email(
        db=db,
        email=user_email,
    )

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account.",
        )

    return user