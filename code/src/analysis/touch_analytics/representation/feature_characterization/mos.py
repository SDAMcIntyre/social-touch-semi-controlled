# representation/feature_characterization/mos.py
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

import pandas as pd
from .base import FeatureExtractor
from ..series_level.mechanics import get_mechanics

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
    | strain_rate_max  | velocity / skin_thickness                | 1/s   |
    | elastic_energy   | 0.5 * E * strain^2 * area * thickness   | mJ    |
    | impulse          | sum(stress * area * dt)                  | mN·s  |
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        E = config.get('youngs_modulus_kpa', _DEFAULTS['youngs_modulus_kpa'])
        h_mm = config.get('skin_thickness_mm', _DEFAULTS['skin_thickness_mm'])
        fps = config.get('fps', _DEFAULTS['fps'])

        mos = get_mechanics(group, E, h_mm, fps)
        strain = mos['mos_strain'].values
        stress = mos['mos_stress_kpa'].values
        strain_rate = mos['mos_strain_rate'].values
        elastic_energy = mos['mos_elastic_energy_mj'].values
        impulse = mos['mos_impulse_mns'].values

        return {
            'strain_max': float(strain.max()),
            'strain_mean': float(strain.mean()),
            'stress_max_kpa': float(stress.max()),
            'stress_mean_kpa': float(stress.mean()),
            'strain_rate_max': float(strain_rate.max()),
            'elastic_energy_max_mj': float(elastic_energy.max()),
            'elastic_energy_total_mj': float(elastic_energy.sum()),
            'impulse_total_mns': float(impulse.sum()),
        }
