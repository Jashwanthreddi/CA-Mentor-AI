from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.router import router
from app.core import logger,settings
from app.core.exception_handlers import register_exception_handlers


@asynccontextmanager
async def lifespan(app:FASTAPI):
    logger.info("Starting CA Mentor AI Backend...")

    yield 

    logger.info("Shutting down CA mentor AI Backend...")


app=FASTAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,

)
register_exception_handlers(app)

app.include_router(router)

