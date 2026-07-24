from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AutoMind"
    API_VERSION: str = "v1"
    ENVIRONMENT: str = "development"

    BACKEND_CORS_ORIGINS: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )

    # PostgreSQL
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "automind_db"

    # JWT authentication
    JWT_SECRET_KEY: str = "change-this-secret-key-later"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Document storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 25

    ALLOWED_DOCUMENT_EXTENSIONS: str = (
        ".pdf,"
        ".docx,"
        ".txt,"
        ".md"
    )

    ALLOWED_DOCUMENT_MIME_TYPES: str = (
        "application/pdf,"
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document,"
        "text/plain,"
        "text/markdown"
    )

    # Future AI infrastructure
    VECTOR_DB_PROVIDER: str = "chromadb"
    LLM_PROVIDER: str = "openai"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def allowed_document_extensions(self) -> set[str]:
        """
        Return normalized document extensions configured for upload.
        """

        return {
            extension.strip().lower()
            for extension in self.ALLOWED_DOCUMENT_EXTENSIONS.split(",")
            if extension.strip()
        }

    @property
    def allowed_document_mime_types(self) -> set[str]:
        """
        Return normalized MIME types configured for upload.
        """

        return {
            mime_type.strip().lower()
            for mime_type in self.ALLOWED_DOCUMENT_MIME_TYPES.split(",")
            if mime_type.strip()
        }

    @property
    def max_upload_size_bytes(self) -> int:
        """
        Convert the configured upload limit from megabytes to bytes.
        """

        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()