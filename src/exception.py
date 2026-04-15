"""
Custom exception class
"""

import sys
import logging


def _get_error_details(error: Exception, error_detail: sys) -> str:
    """Extract file name, line number, and message from an exception."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"
    return (
        f"Error in [{file_name}] at line [{line_number}]: {str(error)}"
    )


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = _get_error_details(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message

