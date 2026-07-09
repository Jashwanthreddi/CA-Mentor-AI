# create a single standard response format for the entire appliation
#instead of everypoint returning different json everything returns one structure

# resp: create success , error responses, keep api responses consistent, reduce duplicate code

from typing import Any
from fastapi.responses import JSONResponse

def success_response(
        *,
        message:str,
        data:Any=None,
        status_code=200,
) ->JSONResponse:
    #standar success response
    return JSONResponse(
        status_code=status_code,
        content={
            "success":True,
            "message":message,
            "data":data,
            "errors":None,
        },
    )
    
def error_response(
        *,
        message:str,
        status_code:int,
        errors:Any=None,

) ->JSONResponse:
    # standard errpr response.
    return JSONResponse(
        status_code=status_code,
        content={
            "success":False,
            "message":message,
            "data":None,
            "errors":errors,
        },
    )