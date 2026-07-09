from fastapi import APIRouter

from app.core import logger
from app.core.responses import success_response


router=APIRouter()

@router.get(
    "/health",
    summary="Health check",
    description="Return the application status"

)
async def health_check():
    logger.info("Health endpoint accessed")
    return success_response(
        message="Application is running",
        data={
            "application":"CA Mentor AI",
            "version":"1.0.0",
            "status_code":"running",
        },
    )

