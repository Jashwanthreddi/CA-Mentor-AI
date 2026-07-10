from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.router import router
from app.core import logger, settings
from app.core.exception_handlers import register_exception_handlers
from app.core.middleware import register_middlewares

@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("Starting CA Mentor AI Backend...")

    yield 

    logger.info("Shutting down CA mentor AI Backend...")


app=FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,

)



register_middlewares(app)
register_exception_handlers(app)
app.include_router(router)
