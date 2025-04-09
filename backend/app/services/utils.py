import math
import logging
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crocodile_services")

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize a Python object for JSON serialization,
    replacing any float infinity or NaN values with None.
    
    Args:
        obj: The object to sanitize
        
    Returns:
        Sanitized object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        # Handle special float values that are not JSON compliant
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def format_classification(raw_classification: dict, header: list) -> dict:
    """
    Format column classification results into a structured format
    compatible with the Crocodile entity linking system.
    
    Args:
        raw_classification: Raw classification output from ColumnClassifier
        header: List of column headers
        
    Returns:
        Formatted classification dictionary with NE, LIT, and IGNORED keys
    """
    ne_types = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
    ne, lit = {}, {}
    
    for i, col_name in enumerate(header):
        col_result = raw_classification.get(col_name, {})
        classification = col_result.get("classification", "UNKNOWN")
        if classification in ne_types:
            ne[str(i)] = classification
        else:
            lit[str(i)] = classification
    
    all_indexes = set(str(i) for i in range(len(header)))
    recognized = set(ne.keys()).union(lit.keys())
    ignored = list(all_indexes - recognized)
    
    return {"NE": ne, "LIT": lit, "IGNORED": ignored}

def log_error(message: str, error: Exception = None) -> None:
    """
    Log an error message with optional exception details.
    
    Args:
        message: Error message 
        error: Optional exception object
    """
    if error:
        logger.error(f"{message}: {str(error)}")
    else:
        logger.error(message)

def log_info(message: str) -> None:
    """
    Log an informational message.
    
    Args:
        message: Information message
    """
    logger.info(message)
