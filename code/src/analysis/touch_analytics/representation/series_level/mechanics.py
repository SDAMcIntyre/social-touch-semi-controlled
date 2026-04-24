# representation/series_level/mechanics.py
import pandas as pd
from .kinematics import get_kinematics

MOS_COLUMNS = ['mos_strain', 'mos_stress_kpa', 'mos_strain_rate', 'mos_elastic_energy_mj', 'mos_impulse_mns']


def compute_mos_series(
    group: pd.DataFrame,
    E_kpa: float,
    h_mm: float,
    fps: float,
) -> dict[str, pd.Series]:
    """Per-frame MoS quantities under a uniaxial linear-elastic skin model."""
    h = h_mm * 1e-3
    dt = 1.0 / fps
    depth_m = group['contact_depth'] * 1e-3
    area_m2 = group['contact_area'] * 1e-6
    vel_m_s, _ = get_kinematics(group, fps)
    vel_m_s = vel_m_s * 1e-3
    mos_strain = depth_m / h
    mos_stress_kpa = E_kpa * mos_strain
    mos_strain_rate = vel_m_s / h
    mos_elastic_energy_mj = 0.5 * E_kpa * mos_strain ** 2 * area_m2 * h * 1e6
    mos_impulse_mns = mos_stress_kpa * area_m2 * dt * 1e6
    return {
        'mos_strain': mos_strain,
        'mos_stress_kpa': mos_stress_kpa,
        'mos_strain_rate': mos_strain_rate,
        'mos_elastic_energy_mj': mos_elastic_energy_mj,
        'mos_impulse_mns': mos_impulse_mns,
    }


def get_mechanics(
    group: pd.DataFrame,
    E_kpa: float,
    h_mm: float,
    fps: float,
) -> dict[str, pd.Series]:
    """Return pre-computed MoS columns if all present, else compute from raw data."""
    if all(col in group.columns for col in MOS_COLUMNS):
        return {col: group[col] for col in MOS_COLUMNS}
    return compute_mos_series(group, E_kpa, h_mm, fps)
