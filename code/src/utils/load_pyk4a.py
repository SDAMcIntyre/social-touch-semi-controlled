import importlib
from typing import Any, Optional

def load_pyk4a():
    """
    Dynamically import pyk4a only when needed.
    
    Returns:
        PyK4APlayback class if import successful, None otherwise.
        
    Example:
        PyK4APlayback = load_pyk4a()
        if PyK4APlayback is None:
            raise ImportError("pyk4a is required for this operation")
    """
    try:
        pyk4a = importlib.import_module('pyk4a')
        PyK4APlayback = pyk4a.PyK4APlayback
        return PyK4APlayback
    except ImportError:
        return None