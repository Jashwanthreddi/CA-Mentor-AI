from app.core import settings

class HealthService:
    # service responsible for application health information
    @staticmethod
    def get_health_status() -> dict:
        return {
            "application":settings.APP_NAME,
            "version":settings.APP_VERSION,
            "status":"running",
        }