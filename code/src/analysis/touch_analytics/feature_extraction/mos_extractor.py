# feature_extraction/mos_extractor.py
"""
Mechanics-of-Solids extractor.

Derives physically meaningful quantities from depth/area/velocity under a
simple uniaxial linear-elastic skin model with configurable tissue properties.

Default tissue parameters (from literature):
  Young's modulus  : 100.0 kPa
  Poisson's ratio  : 0.45
  Skin thickness   : 1.5 mm
  Frame rate       : 30 fps  (used for strain-rate calculation)

See: docs/development/knowledge-base/note-somatosensory-units-and-calculations.md
"""

import numpy as np
import pandas as pd
from .base import FeatureExtractor
from .kinematics import compute_velocity_magnitudes

_DEFAULTS = {
    'youngs_modulus_kpa': 100.0,
    'poissons_ratio': 0.45,
    'skin_thickness_mm': 1.5,
    'fps': 30.0,
}


class MechanicsOfSolidsExtractor(FeatureExtractor):
    """
    Per-touch mechanics features:

    | Feature          | Formula                                  | Units |
    |------------------|------------------------------------------|-------|
    | strain_max/mean  | depth / skin_thickness                   | —     |
    | stress_max/mean  | E * strain                               | kPa   |
    | strain_rate_max  | velocity / (skin_thickness * fps)        | 1/s   |
    | elastic_energy   | 0.5 * E * strain^2 * area * thickness   | mJ    |
    | impulse          | sum(stress * area * dt)                  | mN·s  |
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        E = config.get('youngs_modulus_kpa', _DEFAULTS['youngs_modulus_kpa'])          # kPa
        h = config.get('skin_thickness_mm', _DEFAULTS['skin_thickness_mm']) * 1e-3     # m
        fps = config.get('fps', _DEFAULTS['fps'])
        dt = 1.0 / fps  # s

        depth_m = group['contact_depth'].values * 1e-3          # mm → m
        area_m2 = group['contact_area'].values * 1e-6           # mm² → m²

        vel_magnitudes = compute_velocity_magnitudes(group).values * 1e-3  # mm/s → m/s

        strain = depth_m / h                                     # dimensionless
        stress = E * strain                                      # kPa (E in kPa, strain dimensionless)

        strain_rate = vel_magnitudes / (h * fps)                 # 1/s

        # Elastic energy per frame: 0.5 * E[kPa] * strain^2 * area[m^2] * h[m] → kPa·m^3 = kJ → mJ
        elastic_energy_per_frame = 0.5 * E * strain ** 2 * area_m2 * h * 1e6  # mJ

        # Impulse: stress[kPa] * area[m^2] * dt → kN·s → mN·s ×1e6
        impulse_per_frame = stress * area_m2 * dt * 1e6  # mN·s

        return {
            'strain_max': float(strain.max()),
            'strain_mean': float(strain.mean()),
            'stress_max_kpa': float(stress.max()),
            'stress_mean_kpa': float(stress.mean()),
            'strain_rate_max': float(strain_rate.max()),
            'elastic_energy_max_mj': float(elastic_energy_per_frame.max()),
            'elastic_energy_total_mj': float(elastic_energy_per_frame.sum()),
            'impulse_total_mns': float(impulse_per_frame.sum()),
        }
