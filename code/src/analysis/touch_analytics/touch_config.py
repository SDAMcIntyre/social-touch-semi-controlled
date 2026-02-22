# touch_config.py
from typing import Dict, List, Any

# Configuration for variable discretization
DISCRETIZATION_CONFIG: Dict[str, Any] = {
    'continuous_vars': {
        'max_depth': {
            'method': 'qcut', 
            'q': 3
        },
        'max_contact_area': {
            'method': 'qcut', 
            'q': 3
        },
        'max_velocity': {
            'method': 'qcut', 
            'q': 3
        },
        'max_acceleration': {
            'method': 'qcut', 
            'q': 3
        }
    },
    'categorical_vars': ['type_metadata', 'direction']
}