class CustomException(Exception):
    """Base class for custom exceptions."""
    pass

class FileAbsenceError(CustomException):
    """Raised when the specified file is not found."""
    def __init__(self, message="File not found. Please check the path."):
        super().__init__(message)

class ModelLoadError(CustomException):
    """Raised when a model fails to load."""
    def __init__(self, message="Failed to load the model."):
        super().__init__(message)
