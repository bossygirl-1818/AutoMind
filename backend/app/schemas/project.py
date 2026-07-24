from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.project import ProjectStatus


class ProjectCreate(BaseModel):
    """
    Request schema used to create a new AutoMind project workspace.
    """

    name: str = Field(
        min_length=3,
        max_length=150,
        examples=["BMW iX ADAS Safety Platform"],
    )

    project_key: str = Field(
        min_length=2,
        max_length=50,
        examples=["BMW-IX-ADAS"],
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
        examples=[
            "Secure engineering workspace for ADAS requirements, "
            "safety analysis, testing, and AI-assisted development."
        ],
    )

    status: ProjectStatus = ProjectStatus.PLANNING

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """
        Remove unnecessary leading, trailing, and repeated whitespace.
        """

        normalized_value = " ".join(value.split())

        if not normalized_value:
            raise ValueError("Project name cannot be empty.")

        return normalized_value

    @field_validator("project_key")
    @classmethod
    def normalize_project_key(cls, value: str) -> str:
        """
        Normalize the project key into an uppercase identifier.

        Allowed characters:
        - letters
        - numbers
        - hyphens
        - underscores
        """

        normalized_value = value.strip().upper()

        if not normalized_value:
            raise ValueError("Project key cannot be empty.")

        if not all(
            character.isalnum() or character in {"-", "_"}
            for character in normalized_value
        ):
            raise ValueError(
                "Project key may contain only letters, numbers, "
                "hyphens, and underscores."
            )

        return normalized_value


class ProjectUpdate(BaseModel):
    """
    Request schema used to partially update an existing project.
    """

    name: str | None = Field(
        default=None,
        min_length=3,
        max_length=150,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    status: ProjectStatus | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized_value = " ".join(value.split())

        if not normalized_value:
            raise ValueError("Project name cannot be empty.")

        return normalized_value


class ProjectResponse(BaseModel):
    """
    Safe project representation returned by the API.
    """

    id: UUID
    name: str
    project_key: str
    description: str | None
    status: ProjectStatus

    owner_id: UUID
    created_by_id: UUID

    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None

    model_config = ConfigDict(from_attributes=True)