import sys
import logging

# Custom fuction for Error Details
def error_message_details(error, error_detail: sys):
    _,_,exe_tb= error_detail.exc_info()
    file_name= exe_tb.tb_frame.f_code.co_filename
    error_message= "Error occured in python script name [{0}] line number [{1}] with error message [{2}]".format(file_name,exe_tb.tb_lineno,str(error))

    return error_message


# Custom Exception Class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):

        super().__init__(error_message)
        self.error_message= error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


#just checking the exception functionality
# if __name__ == "__main__":
#     try: 1/0

#     except Exception as e: 
#         raise CustomException(e, sys)


    
