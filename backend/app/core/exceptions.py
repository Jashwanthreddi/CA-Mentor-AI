# it provides a centralized exception system for the entire application
# respon: define application-specific excceptions,std error msgs,
from typing import Any
from fastapi import status

class AppException(Exception):
    #base application exception
    def __init__(
            self,
            message:str,
            status_code:int=status.HTTP_400_BAD_REQUEST,
            details:Any |None=None,
    ):
        self.message=message
        self.status_code=status_code
        self.details=details
        super().__init__(message)

class NotFoundException(AppException):
    #resourcces not found
    def __init__(
            self,
            message:str="Resource not found",
            details:Any  | None=None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )

class ValidationExcetion(AppException):
    #validation failed
    def __init__(
            self,
            message="Validation failed",
            details:Any |None=None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )
        

class InternalServerException(AppException):
    #internal server error
    def __init__(
            self,
            message="Internal server error",
            detils:Any |None=None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=self.details,
        )