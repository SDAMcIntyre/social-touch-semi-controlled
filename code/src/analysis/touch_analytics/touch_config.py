# touch_config.py
from typing import Dict, Any

KINEMATIC_SIGNALS = ('contact_depth', 'contact_area', 'velocity_magnitude', 'acceleration_magnitude', 'geo_pressure')


def get_discretization_config(aggregation: str) -> Dict[str, Any]:
    """Return discretization config for a given kinematic aggregation suffix."""
    return {
        'continuous_vars': {
            f'{sig}_{aggregation}': {'method': 'qcut', 'q': 3}
            for sig in KINEMATIC_SIGNALS
        },
        'categorical_vars': ['type_metadata', 'direction'],
    }


# Backward-compat alias — identical to get_discretization_config('max')
DISCRETIZATION_CONFIG: Dict[str, Any] = get_discretization_config('max')