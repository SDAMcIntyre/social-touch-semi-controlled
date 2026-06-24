"""Shared data-type catalog for GUI dialogs that compose feature names."""

__all__ = ["DATA_TYPES"]

DATA_TYPES: list[str] = [
    "contact_area",
    "contact_depth",
    "hand_velocity",
    "hand_velocity_amplitude",
    "hand_velocity_signed",
    "hand_acceleration",
    "pressure",
    "hand_position",
    "mos_strain",
    "mos_stress_kpa",
    "mos_strain_rate",
    "mos_elastic_energy_mj",
    "mos_impulse_mns",
    "mechanics_of_solids",
    "location",
]
