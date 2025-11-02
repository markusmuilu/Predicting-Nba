import sys
import logging
import os 
from src.utils.logger import logger

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename

    project_root = os.getcwd()
    relative_path = os.path.relpath(file_name, project_root)

    error_message="error occured in python script: {0}, line: {1}, with message: {2}".format(relative_path, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        a= 1/0
    except Exception as error:
        logging.info("Divide by zero")
        raise CustomException(error, sys)