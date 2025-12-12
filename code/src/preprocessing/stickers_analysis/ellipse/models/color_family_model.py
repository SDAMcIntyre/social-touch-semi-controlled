import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from typing import List, Optional, Tuple, Union

class ColorFamilyModel:
    """
    Architectural wrapper for Support Vector Machine (SVM) based color modeling.
    Supports One-Class (outlier detection) and Binary Classification (target vs. background).
    
    Attributes:
        model_type (str): 'binary' or 'one_class'.
        pipeline (sklearn.pipeline.Pipeline): The compiled inference pipeline.
    """

    def __init__(
            self, 
            sample_colors: np.ndarray, 
            color_space: str = 'rgb',
            *,
            conversion_mode: str = 'circular',
            negative_samples_list: Optional[List[np.ndarray]] = None,
            gamma: Union[float, str] = 'scale',
            nu: float = 0.05
    ):
        """
        Args:
            sample_colors: N x 3 array of Target colors.
            color_space: Input format 'rgb', 'bgr', or 'hsv'.
            conversion_mode: Feature engineering strategy ('circular' recommended for Hue).
            negative_samples_list: List of N x 3 arrays containing different groups of colors to discard.
            gamma: Kernel coefficient for RBF. 'scale' uses 1 / (n_features * X.var()).
            nu: Upper bound on fraction of training errors (OneClassSVM only).
        """
        self.input_space = color_space.lower()
        self.conversion_mode = conversion_mode.lower()
        self.gamma = gamma
        self.nu = nu

        # Validation
        if self.conversion_mode not in ['xyz', 'circular']:
            raise ValueError(f"Unknown conversion mode: {self.conversion_mode}")

        # Data Preparation
        X_train, y_train, self.sample_colors_rgb = self._prepare_training_data(
            sample_colors, negative_samples_list
        )

        # Pipeline Construction
        # 1. StandardScaler: Essential for SVM convergence and RBF distance calculation.
        # 2. SVM: The core decision engine.
        if negative_samples_list and len(negative_samples_list) > 0 and y_train is not None:
            # Binary Classification Path
            print(f"[System] Initializing Binary SVC. Training Samples: {len(X_train)} (Target + Negatives)")
            clf = SVC(
                kernel='rbf', 
                gamma=self.gamma, 
                probability=True, 
                class_weight='balanced' # Critical for multiple negative sets
            )
            self.model_type = 'binary'
        else:
            # One-Class Path
            print(f"[System] Initializing OneClassSVM. Training Samples: {len(X_train)} (Target Only)")
            clf = OneClassSVM(kernel='rbf', gamma=self.gamma, nu=self.nu)
            self.model_type = 'one_class'

        self.pipeline = make_pipeline(StandardScaler(), clf)

        # Training Execution
        if self.model_type == 'binary':
            self.pipeline.fit(X_train, y_train)
        else:
            self.pipeline.fit(X_train)
        
        print("[System] Model Training Complete.")

    @staticmethod
    def _hsv_to_circular_features(sample_hsv: np.ndarray) -> np.ndarray:
        """
        Strategy: Unwraps circular Hue into Sin/Cos components.
        Output: [cos(H), sin(H), Saturation, Value]
        """
        h, s, v = sample_hsv[:, 0], sample_hsv[:, 1], sample_hsv[:, 2]
        # Hue is typically 0-1 normalized here. 2*pi*h covers the circle.
        cos_h = np.cos(2 * np.pi * h)
        sin_h = np.sin(2 * np.pi * h)
        return np.stack([cos_h, sin_h, s, v], axis=1)

    @staticmethod
    def _hsv_to_xyz(sample_hsv: np.ndarray) -> np.ndarray:
        """
        Strategy: Cylinder coordinate transform.
        Output: [X, Y, Z]
        """
        h, s, v = sample_hsv[:, 0], sample_hsv[:, 1], sample_hsv[:, 2]
        r = s
        theta_rad = h * 2 * np.pi
        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)
        z = v
        return np.stack((x, y, z), axis=1)

    def _prepare_training_data(
            self, 
            target_samples: np.ndarray, 
            negative_samples_list: Optional[List[np.ndarray]]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        
        # 1. Process Target
        target_hsv, target_rgb_display = self._standardize_input(target_samples)
        target_features = self._extract_features(target_hsv)
        
        X = [target_features]
        # Label 1 for Target
        y = [np.ones(len(target_features))]
        
        # 2. Process Negatives (Iterate over the list)
        if negative_samples_list:
            for i, neg_samples in enumerate(negative_samples_list):
                if neg_samples is None or len(neg_samples) == 0:
                    continue
                
                neg_hsv, _ = self._standardize_input(neg_samples)
                neg_features = self._extract_features(neg_hsv)
                
                X.append(neg_features)
                # Label 0 for Negative
                y.append(np.zeros(len(neg_features)))
        
        # Concatenate
        X_final = np.concatenate(X, axis=0)
        y_final = np.concatenate(y, axis=0) if len(y) > 1 else None
        
        return X_final, y_final, target_rgb_display

    def _standardize_input(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes input to float32 HSV [0..1] and provides uint8 RGB for display.
        """
        # Ensure array structure
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            
        # Handle Data Types and Normalization
        if samples.dtype != np.uint8:
            # Assume float 0-1, convert to uint8 for robust colorspace conversion via OpenCV
            samples_uint8 = (samples * 255).astype(np.uint8)
        else:
            samples_uint8 = samples

        # Reshape for OpenCV: (N, 1, 3)
        samples_reshaped = samples_uint8.reshape(-1, 1, 3)

        # Convert to RGB (for display) and HSV (for features)
        if self.input_space == 'bgr':
            rgb_samples = cv2.cvtColor(samples_reshaped, cv2.COLOR_BGR2RGB)
        elif self.input_space == 'rgb':
            rgb_samples = samples_reshaped
        elif self.input_space == 'hsv':
            # If input is HSV, we need to convert to RGB for display purposes
            rgb_samples = cv2.cvtColor(samples_reshaped, cv2.COLOR_HSV2RGB)
        else:
            raise ValueError("Unsupported color space")

        # Create standardized HSV (Float32, 0.0-1.0 range)
        # using COLOR_RGB2HSV_FULL maps H to [0, 255] which we divide by 255.0
        # This matches the 0-1 expectation of the feature extractors.
        hsv_full = cv2.cvtColor(rgb_samples, cv2.COLOR_RGB2HSV_FULL)
        hsv_float = (hsv_full.astype(np.float32) / 255.0).reshape(-1, 3)
        
        rgb_display = rgb_samples.reshape(-1, 3)
        
        return hsv_float, rgb_display

    def _extract_features(self, hsv_data: np.ndarray) -> np.ndarray:
        if self.conversion_mode == 'xyz':
            return self._hsv_to_xyz(hsv_data)
        return self._hsv_to_circular_features(hsv_data)

    def calculate_probability_map(self, image: np.ndarray, image_color_space: str = 'rgb') -> np.ndarray:
        """
        Inference engine. Returns a heatmap of target probability.
        """
        if image.size == 0:
            return np.array([])
            
        h, w = image.shape[:2]
        
        # 1. Preprocess: Re-use the class setting or override? 
        # Ideally, we follow the instance's configured logic, but image input might differ.
        # We will assume image matches the passed arg, but internal normalization uses instance methods.
        
        # Local standardization just for the image
        # Note: _standardize_input relies on self.input_space. 
        # We temporarily swap it if the image argument differs (a bit of a hack, but flexible)
        original_space = self.input_space
        self.input_space = image_color_space.lower()
        
        try:
            # Flatten image for batch processing
            flat_image = image.reshape(-1, 3)
            image_hsv, _ = self._standardize_input(flat_image)
            features = self._extract_features(image_hsv)
        finally:
            self.input_space = original_space

        # 2. Prediction
        if self.model_type == 'binary':
            # Binary SVC: predict_proba returns [Prob(0), Prob(1)]
            # We want Prob(1)
            probs = self.pipeline.predict_proba(features)[:, 1]
        else:
            # OneClassSVM: decision_function returns signed distance
            dist = self.pipeline.decision_function(features)
            # Sigmoid normalization
            probs = 1 / (1 + np.exp(-10 * dist))

        return probs.reshape(h, w)

    def visualize_analysis(self, test_image: np.ndarray, color_space: str = 'rgb'):
        """Static visualization."""
        prob_map = self.calculate_probability_map(test_image, color_space)
        
        # Get RGB for display
        if color_space.lower() == 'bgr':
            display_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        elif color_space.lower() == 'hsv':
            display_img = cv2.cvtColor(test_image, cv2.COLOR_HSV2RGB)
        else:
            display_img = test_image

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target Colors
        ax[0].imshow(self.sample_colors_rgb.reshape(1, -1, 3), aspect='auto')
        ax[0].set_title("Target Samples")
        ax[0].axis('off')
        
        # Test Image
        ax[1].imshow(display_img)
        ax[1].set_title("Test Image")
        ax[1].axis('off')
        
        # Heatmap
        im = ax[2].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
        ax[2].set_title("Probability Map")
        ax[2].axis('off')
        
        plt.colorbar(im, ax=ax[2])
        plt.tight_layout()
        plt.show()

    def interactive_thresholding(self, image: np.ndarray, color_space: str = 'rgb'):
        """Interactive visualization with slider."""
        prob_map = self.calculate_probability_map(image, color_space)
        
        if color_space.lower() == 'bgr':
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space.lower() == 'hsv':
            display_img = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            display_img = image

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial State
        initial_thresh = 0.5
        mask = prob_map > initial_thresh
        
        # Create overlay: Darken background
        def create_overlay(m):
            overlay = display_img.copy()
            # Alpha blend for rejected areas
            overlay[~m] = (overlay[~m] * 0.2).astype(np.uint8)
            return overlay

        img_obj = ax.imshow(create_overlay(mask))
        ax.set_title(f"Threshold: {initial_thresh:.2f}")
        ax.axis('off')

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, 'Prob Thresh', 0.0, 1.0, valinit=initial_thresh)

        def update(val):
            mask = prob_map > val
            img_obj.set_data(create_overlay(mask))
            ax.set_title(f"Threshold: {val:.2f}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

if __name__ == "__main__":
    # --- PROOF OF ARCHITECTURE: MULTIPLE DISCARD SETS ---
    
    # Scenario: Detect ORANGE fruit.
    # Distractors: PURPLE (Grapes) and GREEN (Leaves).
    
    # 1. Define Target: Orange
    target_orange = np.array([
        [255, 165, 0],
        [255, 140, 0],
        [255, 100, 10]
    ], dtype=np.uint8)

    # 2. Define Discard Set 1: Purple
    discard_purple = np.array([
        [128, 0, 128],
        [75, 0, 130],
        [148, 0, 211]
    ], dtype=np.uint8)

    # 3. Define Discard Set 2: Green
    discard_green = np.array([
        [0, 128, 0],
        [34, 139, 34],
        [0, 255, 0]
    ], dtype=np.uint8)

    # 4. Create Synthetic Test Image containing all three colors
    # Layout: [ Purple | Orange | Green ]
    height, width = 100, 300
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill Purple (Left)
    test_image[:, 0:100] = [128, 0, 128]
    # Fill Orange (Middle - Target)
    test_image[:, 100:200] = [255, 165, 0] 
    # Fill Green (Right)
    test_image[:, 200:300] = [0, 128, 0]

    # Add noise to make it realistic for the SVM
    noise = np.random.randint(-20, 20, test_image.shape)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print("--- Initializing Multi-Negative Model ---")
    
    # Pass BOTH negative sets in the list
    model = ColorFamilyModel(
        sample_colors=target_orange,
        color_space='rgb',
        negative_samples_list=[discard_purple, discard_green], # <--- HERE IS THE PROOF
        conversion_mode='circular',
        gamma='scale' # Auto-tune RBF
    )

    print("--- Visualizing Result ---")
    print("Expectation: Middle section (Orange) is High Probability. Left (Purple) and Right (Green) are Low.")
    model.visualize_analysis(test_image)
    
    print("--- Starting Interactive Thresholding ---")
    model.interactive_thresholding(test_image)