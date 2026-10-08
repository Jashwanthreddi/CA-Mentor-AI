from fastapi import APIRouter

from app.api.v1.endpoints import chat
from app.api.v1.endpoints import health


router = APIRouter()


router.include_router(
    health.router,
    tags=["Health"],
)

router.include_router(
    chat.router,
    tags=["Chat"],
)