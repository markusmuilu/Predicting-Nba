"""
Custom exception handling for the NBA prediction project.
This class logs detailed error information but does not halt execution.
"""

import os
import sys

from predict_nba.utils.logger import logger


def error_message_detail(error, error_detail: sys):
    """
    Build a detailed error message including file name, line number,
    and original exception message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    project_root = os.getcwd()
    relative_path = os.path.relpath(file_name, project_root)

    return (
        f"Error in script: {relative_path}, "
        f"line: {exc_tb.tb_lineno}, "
        f"message: {str(error)}"
    )


class CustomException(Exception):
    """
    Custom exception class that logs an error with detailed traceback information.
    Does not stop program execution unless manually raised.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as error:
        CustomException(error, sys)
