import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict, List

# --- 3. Math Kernel ---

@dataclass
class CalibrationResult:
    mean_1: np.ndarray
    R1: np.ndarray
    mean_2: np.ndarray
    R2: np.ndarray
    
    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "primary_mean": self.mean_1.tolist(),
            "primary_rotation_R1": self.R1.tolist(),
            "secondary_mean_2d": self.mean_2.tolist(),
            "secondary_rotation_R2": self.R2.tolist()
        }

class PCACalibrationEngine:
    """Pure logic for computing and applying PCA transformations."""
    
    @staticmethod
    def compute_calibration(tapping_data: np.ndarray, stroking_data: np.ndarray) -> CalibrationResult:
        """
        Phase 1: Z-Axis alignment using Tapping data.
        Phase 2: XY-Plane alignment using Stroking data.
        """
        # 1. Primary PCA (Z-Axis) on Tapping
        pca_z = PCA(n_components=3)
        pca_z.fit(tapping_data)
        
        mean_1 = pca_z.mean_
        comps_1 = pca_z.components_ 
        
        # Assign 1st PC to Z-axis: New Z' = PC1, New X' = PC2, New Y' = PC3
        # This forces the axis with most variation (tapping depth) to become Z
        R1 = np.vstack([comps_1[1], comps_1[2], comps_1[0]])

        # 2. Secondary PCA (XY-Plane) on Stroking
        # Transform stroking data using Phase 1 parameters (Z-align)
        str_step1 = PCACalibrationEngine._apply_z_alignment(stroking_data, mean_1, R1)
        
        # Extract X', Y' for 2D PCA
        str_xy = str_step1[:, :2]
        
        pca_xy = PCA(n_components=2)
        pca_xy.fit(str_xy)
        
        mean_2 = pca_xy.mean_
        comps_2 = pca_xy.components_
        
        # Expand R2 to 3x3 identity for Z preservation
        R2 = np.eye(3)
        R2[:2, :2] = comps_2
        
        return CalibrationResult(mean_1, R1, mean_2, R2)

    @staticmethod
    def _apply_z_alignment(coords: np.ndarray, mean_1: np.ndarray, R1: np.ndarray) -> np.ndarray:
        """Internal helper for Step 1 transformation."""
        return (coords - mean_1) @ R1.T

    @staticmethod
    def apply_step1_transform(coords: np.ndarray, calib: CalibrationResult) -> np.ndarray:
        """
        Public method to apply ONLY the first PCA transformation (Z-alignment).
        Useful for debugging and intermediate visualization.
        """
        return PCACalibrationEngine._apply_z_alignment(coords, calib.mean_1, calib.R1)

    @staticmethod
    def apply_full_transform(coords: np.ndarray, calib: CalibrationResult) -> np.ndarray:
        """Applies the complete calculated transformation to a coordinate set."""
        # 1. Step 1: Z-Alignment
        coords_step1 = PCACalibrationEngine._apply_z_alignment(coords, calib.mean_1, calib.R1)
        
        # 2. Step 2: XY-Alignment
        # Center XY
        coords_step1[:, 0] -= calib.mean_2[0]
        coords_step1[:, 1] -= calib.mean_2[1]
        
        # Rotate XY
        coords_final = coords_step1 @ calib.R2.T
        
        return coords_final
