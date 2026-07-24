from fastapi import APIRouter

from app.api.v1 import (
    auth,
    documents,
    grounded_answers,
    health,
    projects,
    retrieval,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(projects.router)
api_router.include_router(documents.router)
api_router.include_router(retrieval.router)
api_router.include_router(grounded_answers.router)