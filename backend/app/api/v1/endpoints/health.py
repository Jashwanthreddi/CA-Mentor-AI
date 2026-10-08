from fastapi import APIRouter

from app.core import logger
from app.core.responses import success_response
from app.services.health_service import HealthService


router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Return the application status",
)
async def health_check():
    logger.info("Health endpoint accessed")

    return success_response(
        message="Application is running",
        data=HealthService.get_health_status(),
    )