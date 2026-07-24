from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.user import User
from app.schemas.user import UserCreate


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user_data: UserCreate) -> User:
    user = User(
        full_name=user_data.full_name,
        email=user_data.email.lower(),
        password_hash=hash_password(user_data.password),
        role=user_data.role,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user

def authenticate_user(
    db: Session,
    email: str,
    password: str,
) -> User | None:
    """
    Authenticate an AutoMind user using email and password.

    Returns the user when authentication succeeds.
    Returns None when the user does not exist, the password is incorrect,
    or the account is inactive.
    """

    user = get_user_by_email(
        db=db,
        email=email,
    )

    if user is None:
        return None

    if not verify_password(
        plain_password=password,
        hashed_password=user.password_hash,
    ):
        return None

    if not user.is_active:
        return None

    return user

def update_last_login(
    db: Session,
    user: User,
) -> User:
    """
    Update the user's latest successful login timestamp.

    The timestamp is stored in UTC so login activity remains consistent
    across environments and deployment regions.
    """

    user.last_login_at = datetime.now(timezone.utc)

    db.add(user)
    db.commit()
    db.refresh(user)

    return user