# # Middleware is like the airport security checkpoint.
# # Intercept every request before it reaches the API
# # res:generate request id, neasure execution time, log request start, log request completion.
# Intercept every incoming request before it reaches the router and every outgoing response before it leaves the application.

import time
import uuid

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger

class RequestMiddleware(BaseHTTPMiddleware):
    #middleware for request logging and timing
    async def dispatch(self,request:Request,call_next):
        # generate unique request id
        request_id=str(uuid.uuid4())
        request.state.request_id=request_id
        start_time=time.perf_counter()

        logger.info(
            f"[{request_id}]"
            f"incoming{request.method} {request.url.path}"

        )
        response=await call_next(request)

        process_time=round(
            (time.perf_counter() - start_time) * 1000,2

        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time} ms"

        logger.info(
            f"[{request_id}]"
            f"completed"
            f"Status={response.status_code}"
            f"Time={process_time} ms"
        )

        return response
    

def register_middlewares(app:FastAPI) ->None:
    # register all application middelware
    app.add_middleware(RequestMiddleware)