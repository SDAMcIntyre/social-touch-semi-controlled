import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
from matplotlib.widgets import Slider
from typing import List, Optional

def hsv_to_xyz(sample_hsv):
    """
    Converts an array of HSV values into Cartesian XYZ coordinates,
    treating the HSV color space as a cylinder.
    """
    h, s, v = sample_hsv[:, 0], sample_hsv[:, 1], sample_hsv[:, 2]
    r = s
    theta_rad = h * 2 * np.pi
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    z = v
    return np.stack((x, y, z), axis=1)

def hsv_to_circular_features(sample_hsv):
    """
    Converts an array of HSV values into a 4D feature representation
    by unwrapping the circular Hue component into two dimensions.
    """
    h, s, v = sample_hsv[:, 0], sample_hsv[:, 1], sample_hsv[:, 2]
    cos_h = np.cos(2 * np.pi * h)
    sin_h = np.sin(2 * np.pi * h)
    return np.stack([cos_h, sin_h, s, v], axis=1)


class ColorFamilyModel:
    """
    A class to model a family of colors using a multivariate Gaussian distribution.
    Supports discriminative modeling by accepting 'negative' (discarded) samples
    to penalize false positives.
    """
    def __init__(
            self, 
            sample_colors: np.ndarray, 
            color_space: str = 'rgb',
            *,
            conversion_mode: str = 'xyz',
            negative_samples_list: Optional[List[np.ndarray]] = None):
        """
        Initializes the model.

        Args:
            sample_colors (np.ndarray): An N x 3 numpy array of sample colors (Target).
            color_space (str, optional): The format of `sample_colors`. 
                                         Can be 'rgb', 'bgr', or 'hsv'. Defaults to 'rgb'.
            conversion_mode (str, optional): The feature space to use for distance calculation.
                                             Options: 'xyz' (3D cylindrical) or 
                                             'circular' (4D circular-aware). Defaults to 'xyz'.
            negative_samples_list (list, optional): A list of N x 3 arrays containing 'discarded'
                                                    color samples. Pixels close to these will be penalized.
        """
        self.input_space = color_space.lower()
        self.conversion_mode = conversion_mode.lower()
        
        # Select conversion strategy
        if self.conversion_mode == 'xyz':
            self._hsv_to_features_converter = hsv_to_xyz
        elif self.conversion_mode == 'circular':
            self._hsv_to_features_converter = hsv_to_circular_features
        else:
            raise ValueError(f"Unsupported conversion_mode: '{conversion_mode}'. Use 'xyz' or 'circular'.")

        # --- Train Target Model ---
        self.target_stats = self._train_single_model(sample_colors, self.input_space)
        self.mean_vector = self.target_stats['mean'] # Kept for backward compat
        self.inv_covariance_matrix = self.target_stats['inv_cov'] # Kept for backward compat
        self.sample_colors_rgb = self.target_stats['rgb_samples']

        # --- Train Negative Models (if any) ---
        self.negative_models = []
        if negative_samples_list:
            for neg_samples in negative_samples_list:
                if neg_samples is not None and len(neg_samples) > 0:
                    self.negative_models.append(
                        self._train_single_model(neg_samples, self.input_space)
                    )
            if self.negative_models:
                print(f"Initialized {len(self.negative_models)} negative discriminative models.")

        print("Color Family Model Initialized.")
        print(f"Input format: {self.input_space.upper()}, Internal feature space: {self.conversion_mode.upper()}")

    def _train_single_model(self, samples: np.ndarray, input_space: str) -> dict:
        """Internal helper to calculate Mean and Inverse Covariance for a set of samples."""
        # 1. Standardize input colors to HSV [0,1]
        if input_space in ['rgb', 'bgr']:
            if samples.dtype != np.uint8:
                raise TypeError(f"{input_space.upper()} samples must be uint8 [0, 255].")
            
            samples_reshaped = samples.reshape(-1, 1, 3)
            rgb_samples = cv2.cvtColor(samples_reshaped, cv2.COLOR_BGR2RGB) if input_space == 'bgr' else samples_reshaped
            sample_hsv_uint8 = cv2.cvtColor(rgb_samples, cv2.COLOR_RGB2HSV_FULL)
            sample_hsv = (sample_hsv_uint8.astype(np.float32) / 255.0).reshape(-1, 3)
            rgb_display = rgb_samples.reshape(-1, 3)

        elif input_space == 'hsv':
            if not np.issubdtype(samples.dtype, np.floating) or np.max(samples) > 1.0 or np.min(samples) < 0.0:
                raise ValueError("HSV samples must be float values in the range [0, 1].")
            
            sample_hsv = samples
            sample_hsv_3d = sample_hsv[np.newaxis, ...].astype(np.float32)
            rgb_float = cv2.cvtColor(sample_hsv_3d, cv2.COLOR_HSV2RGB)
            rgb_display = (rgb_float[0] * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported color space: {input_space}.")

        # 2. Transform to feature space
        feature_vectors = self._hsv_to_features_converter(sample_hsv)

        # 3. Calculate Stats
        mean_vec = np.mean(feature_vectors, axis=0) 
        cov_matrix = np.cov(feature_vectors, rowvar=False)

        # 4. Inverse Covariance with Regularization
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            num_dimensions = cov_matrix.shape[0]
            epsilon = 1e-6
            inv_cov = np.linalg.inv(cov_matrix + np.eye(num_dimensions) * epsilon)

        return {
            'mean': mean_vec,
            'inv_cov': inv_cov,
            'rgb_samples': rgb_display
        }

    def calculate_mahalanobis_map(self, image: np.ndarray, color_space: str = 'rgb') -> np.ndarray:
        """
        Processes an image to generate a heatmap of Mahalanobis distances.
        If negative models exist, distances are penalized (increased) if pixels 
        are close to negative samples.
        """
        height, width, _ = image.shape
        if image.size == 0:
            return np.empty((height, width), dtype=np.float64)

        input_space = color_space.lower()

        # Step 1: Convert input image to normalized HSV
        if input_space == 'bgr':
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        elif input_space == 'rgb':
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        elif input_space == 'hsv':
            image_hsv = image
        else:
            raise ValueError(f"Unsupported color space: {input_space}.")
        
        pixels_hsv = image_hsv.reshape(-1, 3).astype(np.float32) / 255.0
        feature_vectors = self._hsv_to_features_converter(pixels_hsv)

        # Step 2: Calculate Target Mahalanobis distance
        # cdist returns a (N, 1) array for a single mean vector
        dist_target = cdist(feature_vectors, [self.target_stats['mean']],
                          metric='mahalanobis', VI=self.target_stats['inv_cov']).flatten()

        # Step 3: Apply Discriminative Penalty (if negative models exist)
        if self.negative_models:
            # Calculate distance to all negative models
            neg_distances = []
            for neg_model in self.negative_models:
                d = cdist(feature_vectors, [neg_model['mean']],
                          metric='mahalanobis', VI=neg_model['inv_cov']).flatten()
                neg_distances.append(d)
            
            # Find the minimum distance to ANY negative class for each pixel
            # shape: (num_pixels,)
            min_neg_dist = np.min(np.stack(neg_distances, axis=1), axis=1)

            # --- Penalty Logic ---
            # We want to increase the target distance if min_neg_dist is small.
            # Formula: D_final = D_target * (1 + PENALTY_FACTOR * exp(-D_neg))
            # If D_neg is 0 (identical to discarded), we multiply D_target by (1 + PENALTY_FACTOR).
            # If D_neg is large, the term vanishes.
            penalty_factor = 50.0  # Tunable parameter: severity of the discard
            suppression_weight = np.exp(-min_neg_dist) # 1.0 at 0 dist, decays to 0
            
            dist_target = dist_target * (1.0 + penalty_factor * suppression_weight)

        return dist_target.reshape(height, width)
    
    def _get_rgb_for_display(self, image, color_space='rgb'):
        """Internal helper to get a display-ready RGB version of an image."""
        color_space = color_space.lower()
        if color_space == 'rgb':
            return image
        elif color_space == 'bgr':
            return image[..., ::-1] 
        elif color_space == 'hsv':
            return (mcolors.hsv_to_rgb(image) * 255).astype(np.uint8)
        return None

    def visualize_analysis(
            self,
            test_image: np.ndarray, 
            color_space: str = 'rgb', 
            title: str = "Color Correlation Analysis",
            *,
            fig=None,
            block=True
    ):
        """Displays summary plot of sample colors, test image, and correlation map."""
        test_image_rgb = self._get_rgb_for_display(test_image, color_space)
        if test_image_rgb is None:
             raise ValueError(f"Unsupported color space: {color_space}.")

        correlation_map = self.calculate_mahalanobis_map(test_image, color_space=color_space)
        
        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig.clear()
            axes = fig.subplots(1, 3)
        
        # Visualize Target Samples
        axes[0].imshow(self.sample_colors_rgb.reshape(-1, 1, 3))
        axes[0].set_title('Target Colors (RGB)')
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(test_image_rgb)
        axes[1].set_title(f'Test Image ({color_space.upper()})')
        axes[1].axis('off')

        im = axes[2].imshow(correlation_map, cmap='hot_r')
        axes[2].set_title('Correlation Map (Penalized)')
        axes[2].axis('off')

        fig.colorbar(im, ax=axes[2], orientation='vertical')
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.show(block=block)
        if not block:
            plt.draw()
            plt.pause(0.001)

    def interactive_thresholding(self, image, color_space='rgb', initial_threshold=2.0, title="Interactive Thresholding", fig=None, block=True):
        """Creates an interactive plot to dynamically threshold the correlation map."""
        image_rgb_display = self._get_rgb_for_display(image, color_space)
        correlation_map = self.calculate_mahalanobis_map(image, color_space=color_space)

        if fig is None:
            fig, ax = plt.subplots()
        else:
            fig.clear()
            ax = fig.add_subplot(1, 1, 1)

        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(left=0.1, bottom=0.25, top=0.9)

        mask = correlation_map < initial_threshold
        result_image = np.zeros_like(image_rgb_display)
        result_image[mask] = image_rgb_display[mask]
        
        img_display = ax.imshow(result_image)
        ax.set_title(f'Threshold = {initial_threshold:.2f}')
        ax.axis('off')

        ax_slider = fig.add_axes([0.1, 0.1, 0.8, 0.05])
        threshold_slider = Slider(
            ax=ax_slider, label='Threshold', valmin=np.min(correlation_map),
            valmax=np.max(correlation_map), valinit=initial_threshold
        )

        def update(val):
            new_mask = correlation_map < val
            new_result_image = np.zeros_like(image_rgb_display)
            new_result_image[new_mask] = image_rgb_display[new_mask]
            img_display.set_data(new_result_image)
            ax.set_title(f'Threshold = {val:.2f}')
            fig.canvas.draw_idle()

        threshold_slider.on_changed(update)
        fig.slider = threshold_slider

        plt.show(block=block)
        if not block:
            plt.draw()
            plt.pause(0.001)

if __name__ == '__main__':
    # Demo with negative samples
    target_green = np.array([[25, 130, 45], [35, 145, 55]], dtype=np.uint8)
    discarded_yellowish = np.array([[20, 180, 180], [30, 200, 200]], dtype=np.uint8)

    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :50] = [25, 130, 45] # Target Green
    test_image[:, 50:] = [25, 180, 180] # Confusing Yellowish Green (should be penalized)

    print("--- DEMO: Discriminative Model ---")
    model = ColorFamilyModel(
        target_green, 
        color_space='rgb', 
        negative_samples_list=[discarded_yellowish]
    )
    
    model.visualize_analysis(test_image, title="Green Target vs Yellow Discarded")