# reduction/scaling.py
"""
Scaler factory for the reduction pipeline.
"""

from sklearn.preprocessing import RobustScaler, StandardScaler


def get_scaler(name: str):
    """
    Return a scikit-learn scaler instance for the given name.

    Parameters
    ----------
    name : str
        One of ``"standard"``, ``"robust"``, or ``"none"``.

    Returns
    -------
    scaler instance or None
        ``StandardScaler()`` for ``"standard"``, ``RobustScaler()`` for
        ``"robust"``, and ``None`` for ``"none"``.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    name = name.lower() if name else "none"
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "none":
        return None
    raise ValueError(
        f"Unknown scaler '{name}'. Choose from: 'standard', 'robust', 'none'."
    )
