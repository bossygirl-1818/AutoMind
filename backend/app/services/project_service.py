from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session
from app.models.project import Project, ProjectStatus

from app.models.user import User
from app.schemas.project import ProjectCreate, ProjectUpdate


def get_project_by_key(
    db: Session,
    project_key: str,
) -> Project | None:
    """
    Retrieve a project using its unique project key.
    """

    normalized_key = project_key.strip().upper()

    return (
        db.query(Project)
        .filter(Project.project_key == normalized_key)
        .first()
    )


def create_project(
    db: Session,
    project_data: ProjectCreate,
    current_user: User,
) -> Project:
    """
    Create a new AutoMind project workspace.

    The authenticated user becomes both the initial owner and creator.
    """

    project = Project(
        name=project_data.name,
        project_key=project_data.project_key,
        description=project_data.description,
        status=project_data.status,
        owner_id=current_user.id,
        created_by_id=current_user.id,
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return project


def get_projects_for_user(
    db: Session,
    current_user: User,
) -> list[Project]:
    """
    Retrieve all project workspaces owned by the authenticated user.

    Projects are returned with the most recently created workspace first.
    """

    return (
        db.query(Project)
        .filter(Project.owner_id == current_user.id)
        .order_by(Project.created_at.desc())
        .all()
    )


def get_project_by_id_for_user(
    db: Session,
    project_id: UUID,
    current_user: User,
) -> Project | None:
    """
    Retrieve a project by UUID only when it belongs to the authenticated user.
    """

    return (
        db.query(Project)
        .filter(
            Project.id == project_id,
            Project.owner_id == current_user.id,
        )
        .first()
    )


def update_project(
    db: Session,
    project: Project,
    project_data: ProjectUpdate,
) -> Project:
    """
    Partially update an existing AutoMind project workspace.

    Only fields explicitly supplied by the client are updated.
    Immutable fields such as project_key, owner_id, and created_by_id
    are not included in the ProjectUpdate schema.
    """

    update_data = project_data.model_dump(
        exclude_unset=True,
    )

    for field_name, field_value in update_data.items():
        setattr(project, field_name, field_value)

    db.add(project)
    db.commit()
    db.refresh(project)

    return project

def archive_project(
    db: Session,
    project: Project,
) -> None:
    """
    Soft-delete an AutoMind project workspace.

    The project remains stored for audit history and future restoration,
    but its lifecycle status is changed to archived.
    """

    project.status = ProjectStatus.ARCHIVED
    project.archived_at = datetime.utcnow()

    db.add(project)
    db.commit()