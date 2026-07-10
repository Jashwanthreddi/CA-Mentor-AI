#purpose : provide centralized location to handle all application exceptions and convert them into consistent api respones
# rep: handle custom APPException, FASTAPI validation errors, handle unexperted exceptions, log every error, return standardize json response
# used by fastapi application
# separation of concerns, centralized error handling

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from app.core.exceptions import(
    AppException,
)
from app.core.logging import logger
from app.core.responses import error_response


def register_exception_handlers(app:FastAPI) ->None:
    # register all global exception handlers
    @app.exception_handler(AppException)
    async def app_exception_handler(
        request:Request,
        exc:AppException,
    ):
        logger.error(
            f"{request.method} {request.url.path} | {exc.message}"

        )
        return error_response(
            message=exc.message,
            status_code=exc.status_code,
            errors=exc.details,
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request:Request,
        exc:RequestValidationError,
    ):
        logger.warning(
            f"Validation Error | {request.url.path}"

        )
        return error_response(
            message="Validation error",
            status_code=422,
            errors=exc.errors(),

        )
    
    @app.exception_handler(Exception)
    async def unexperted_exception_handler(
        request:Request,
        exc:Exception,
    ):
        logger.exception(exc)

        return error_response(
            message="Internal server error",
            status_code=500,
            errors=None,
        )
