from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.schemas.project import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
)
from app.services.project_service import (
    archive_project,
    create_project,
    get_project_by_id_for_user,
    get_project_by_key,
    get_projects_for_user,
    update_project,
)


router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new project workspace",
    responses={
        status.HTTP_201_CREATED: {
            "description": "Project workspace created successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "A project with the provided project key already exists.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The submitted project data is invalid.",
        },
    },
)
def create_project_workspace(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectResponse:
    """
    Create a secure AutoMind project workspace.

    The authenticated user becomes both the project creator
    and the initial project owner.
    """

    existing_project = get_project_by_key(
        db=db,
        project_key=project_data.project_key,
    )

    if existing_project is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A project with this project key already exists.",
        )

    return create_project(
        db=db,
        project_data=project_data,
        current_user=current_user,
    )


@router.get(
    "",
    response_model=list[ProjectResponse],
    status_code=status.HTTP_200_OK,
    summary="List the authenticated user's projects",
    responses={
        status.HTTP_200_OK: {
            "description": "Project workspaces returned successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
    },
)
def list_project_workspaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ProjectResponse]:
    """
    Return all project workspaces owned by the authenticated user.

    Projects are returned from newest to oldest.
    """

    return get_projects_for_user(
        db=db,
        current_user=current_user,
    )


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a project workspace by ID",
    responses={
        status.HTTP_200_OK: {
            "description": "Project workspace returned successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project workspace not found.",
        },
    },
)
def get_project_workspace(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectResponse:
    """
    Retrieve a single project workspace owned by the authenticated user.
    """

    project = get_project_by_id_for_user(
        db=db,
        project_id=project_id,
        current_user=current_user,
    )

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project workspace not found.",
        )

    return project


@router.put(
    "/{project_id}",
    response_model=ProjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a project workspace",
    responses={
        status.HTTP_200_OK: {
            "description": "Project workspace updated successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project workspace not found.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The submitted project data is invalid.",
        },
    },
)
def update_project_workspace(
    project_id: UUID,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectResponse:
    """
    Update an existing project workspace owned by the authenticated user.

    Only mutable fields defined in ProjectUpdate may be changed.
    """

    project = get_project_by_id_for_user(
        db=db,
        project_id=project_id,
        current_user=current_user,
    )

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project workspace not found.",
        )

    return update_project(
        db=db,
        project=project,
        project_data=project_data,
    )

@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Archive a project workspace",
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Project workspace archived successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project workspace not found.",
        },
    },
)
def archive_project_workspace(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Archive an existing project workspace owned by the authenticated user.

    The project remains in the database for audit history,
    but its status becomes ARCHIVED.
    """

    project = get_project_by_id_for_user(
        db=db,
        project_id=project_id,
        current_user=current_user,
    )

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project workspace not found.",
        )

    archive_project(
        db=db,
        project=project,
    )

    return None