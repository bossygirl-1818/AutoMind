from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import logger, setup_logging
from sqlalchemy import text
from app.database.session import engine
from app.core.logging import logger

setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Secure Multi-Agent AI Safety Engineering Copilot for Software-Defined Vehicles",
    version="1.0.0"
)

app.include_router(api_router, prefix=f"/api/{settings.API_VERSION}")

logger.info("🚀 AutoMind backend started successfully.")

try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        logger.info("✅ PostgreSQL connection successful.")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "message": "Welcome to AutoMind 🚗🤖"
    }