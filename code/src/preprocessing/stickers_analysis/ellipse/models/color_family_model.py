import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
from matplotlib.widgets import Slider

# (hsv_to_xyz and hsv_to_circular_features functions remain the same)

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
    This version centralizes the logic for handling different feature space conversions.
    """
    def __init__(
            self, 
            sample_colors: np.ndarray, 
            color_space: str = 'rgb',
            *,
            conversion_mode: str = 'xyz'):
        """
        Initializes the model.

        Args:
            sample_colors (np.ndarray): An N x 3 numpy array of sample colors.
            color_space (str, optional): The format of `sample_colors`. 
                                         Can be 'rgb', 'bgr', or 'hsv'. Defaults to 'rgb'.
            conversion_mode (str, optional): The feature space to use for distance calculation.
                                             Options: 'xyz' (3D cylindrical) or 
                                             'circular' (4D circular-aware). Defaults to 'xyz'.
        """
        input_space = color_space.lower()
        
        # --- New: Centralized logic for selecting the conversion function ---
        self.conversion_mode = conversion_mode.lower()
        if self.conversion_mode == 'xyz':
            self._hsv_to_features_converter = hsv_to_xyz
        elif self.conversion_mode == 'circular':
            self._hsv_to_features_converter = hsv_to_circular_features
        else:
            raise ValueError(f"Unsupported conversion_mode: '{conversion_mode}'. Use 'xyz' or 'circular'.")

        # --- Step 1: Standardize input colors to HSV [0,1] ---
        if input_space in ['rgb', 'bgr']:
            # ... (this block remains unchanged)
            if sample_colors.dtype != np.uint8:
                raise TypeError(f"{input_space.upper()} samples must be uint8 [0, 255].")
            
            sample_colors_reshaped = sample_colors.reshape(-1, 1, 3)
            self.sample_colors_rgb = cv2.cvtColor(sample_colors_reshaped, cv2.COLOR_BGR2RGB) if input_space == 'bgr' else sample_colors_reshaped
            sample_hsv_uint8 = cv2.cvtColor(self.sample_colors_rgb, cv2.COLOR_RGB2HSV_FULL)
            sample_hsv = (sample_hsv_uint8.astype(np.float32) / 255.0).reshape(-1, 3)

        elif input_space == 'hsv':
            # ... (this block remains unchanged)
            if not np.issubdtype(sample_colors.dtype, np.floating) or np.max(sample_colors) > 1.0 or np.min(sample_colors) < 0.0:
                raise ValueError("HSV samples must be float values in the range [0, 1].")
            
            sample_hsv = sample_colors
            sample_hsv_3d = sample_hsv[np.newaxis, ...].astype(np.float32)
            rgb_float = cv2.cvtColor(sample_hsv_3d, cv2.COLOR_HSV2RGB)
            self.sample_colors_rgb = (rgb_float[0] * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported color space: {input_space}. Use 'rgb', 'bgr', or 'hsv'.")

        # --- Step 2: Transform HSV to the chosen feature space using the stored function ---
        self.feature_vectors = self._hsv_to_features_converter(sample_hsv)

        # --- Step 3: Calculate Mean and Covariance in the chosen space ---
        self.mean_vector = np.mean(self.feature_vectors, axis=0) 
        self.covariance_matrix = np.cov(self.feature_vectors, rowvar=False)

        # --- Step 4: Calculate the Inverse of the Covariance Matrix ---
        try:
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix is singular. Regularizing with a small epsilon.")
            num_dimensions = self.covariance_matrix.shape[0]
            epsilon = 1e-6
            regularization = np.eye(num_dimensions) * epsilon
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix + regularization)

        print("Color Family Model Initialized.")
        print(f"Input format: {input_space.upper()}, Internal feature space: {self.conversion_mode.upper()}")

    def calculate_mahalanobis_map(self, image: np.ndarray, color_space: str = 'rgb') -> np.ndarray:
        """
        Processes an image to generate a heatmap of Mahalanobis distances.
        This map can be interpreted as an anomaly score for each pixel.
        """
        height, width, _ = image.shape
        if image.size == 0:
            return np.empty((height, width), dtype=np.float64)

        input_space = color_space.lower()

        # Step 1: Convert the input image to the HSV color space.
        if input_space == 'bgr':
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        elif input_space == 'rgb':
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        elif input_space == 'hsv':
            image_hsv = image
        else:
            raise ValueError(f"Unsupported color space: {input_space}. Use 'rgb', 'bgr', or 'hsv'.")
        
        pixels_hsv = image_hsv.reshape(-1, 3).astype(np.float32) / 255.0

        # --- Modified: Use the stored converter function directly ---
        # This removes the hardcoded logic and ensures consistency with the
        # feature space used during model initialization.
        feature_vectors = self._hsv_to_features_converter(pixels_hsv)

        # Step 2: Calculate Mahalanobis distance for each pixel.
        distances = cdist(feature_vectors, [self.mean_vector],
                          metric='mahalanobis', VI=self.inv_covariance_matrix)

        # Step 3: Reshape the resulting distances back to the original image dimensions.
        return distances.reshape(height, width)
    
    def _get_rgb_for_display(self, image, color_space='rgb'):
        """Internal helper to get a display-ready RGB version of an image."""
        color_space = color_space.lower()
        if color_space == 'rgb':
            return image
        elif color_space == 'bgr':
            return image[..., ::-1] # BGR to RGB
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
        """
        Displays a summary plot of sample colors, test image, and correlation map.

        Args:
            test_image (np.ndarray): HxWx3 image. uint8 [0,255] for 'rgb'/'bgr', float [0,1] for 'hsv'.
            color_space (str, optional): Format of `image`. 'rgb', 'bgr', or 'hsv'. Defaults to 'rgb'.
            title (str, optional): The main title for the plot window. Defaults to "Color Correlation Analysis".
            fig (matplotlib.figure.Figure, optional): An existing figure to draw on. If None, a new one is created.
            block (bool, optional): If True, `plt.show()` blocks execution until the window is closed.
        """
        test_image_rgb = self._get_rgb_for_display(test_image, color_space)
        if test_image_rgb is None:
             raise ValueError(f"Unsupported color space: {color_space}.")

        correlation_map = self.calculate_mahalanobis_map(test_image, color_space=color_space)
        
        # If no figure is provided, create one. Otherwise, clear the existing one for reuse.
        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig.clear()
            axes = fig.subplots(1, 3)
        
        axes[0].imshow(self.sample_colors_rgb.reshape(-1, 1, 3))
        axes[0].set_title('Sample Colors (as RGB)')
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(test_image_rgb)
        axes[1].set_title(f'Original Test Image ({color_space.upper()})')
        axes[1].axis('off')

        im = axes[2].imshow(correlation_map, cmap='hot_r')
        axes[2].set_title('Correlation Map (Mahalanobis Distance)')
        axes[2].axis('off')

        fig.colorbar(im, ax=axes[2], orientation='vertical')
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.show(block=block)
        # If non-blocking, we must manually draw the canvas and pause briefly
        # to give the GUI time to render the plot.
        if not block:
            plt.draw()
            plt.pause(0.001)

    def interactive_thresholding(self, image, color_space='rgb', initial_threshold=2.0, title="Interactive Thresholding", fig=None, block=True):
        """
        Creates an interactive plot to dynamically threshold the correlation map.

        Args:
            image (np.ndarray): HxWx3 image to be processed.
            color_space (str, optional): The color space of the image. Defaults to 'rgb'.
            initial_threshold (float, optional): The starting threshold value for the slider. Defaults to 2.0.
            title (str, optional): The main title for the plot window. Defaults to "Interactive Thresholding".
            fig (matplotlib.figure.Figure, optional): An existing figure to draw on. If None, a new one is created.
            block (bool, optional): If True, `plt.show()` blocks execution until the window is closed.
        """
        image_rgb_display = self._get_rgb_for_display(image, color_space)
        if image_rgb_display is None:
             raise ValueError(f"Unsupported color space: {color_space}.")

        correlation_map = self.calculate_mahalanobis_map(image, color_space=color_space)

        # If no figure is provided, create one. Otherwise, clear it for reuse.
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

        # Add new axes for the slider each time the function is called
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
        
        # IMPORTANT: Store the slider on the figure to prevent it from being
        # garbage collected, which would deactivate the callback.
        fig.slider = threshold_slider

        plt.show(block=block)
        # Handle non-blocking display
        if not block:
            plt.draw()
            plt.pause(0.001)

# --- ILLUSTRATION OF THE NEW FUNCTIONALITY ---
if __name__ == '__main__':
    # 1. Define sample colors and a test image
    sample_green_colors = np.array([
        [25, 130, 45], [35, 145, 55], [20, 120, 40],
        [40, 150, 60], [50, 160, 75], [30, 135, 50]
    ], dtype=np.uint8)

    test_image_rgb = np.zeros((100, 300, 3), dtype=np.uint8)
    test_image_rgb[20:80, 20:80] = [33, 140, 52]  # Green patch with noise
    noise = np.random.randint(-10, 10, test_image_rgb[20:80, 20:80].shape)
    test_image_rgb[20:80, 20:80] = np.clip(test_image_rgb[20:80, 20:80] + noise, 0, 255)
    test_image_rgb[20:80, 120:180] = [20, 50, 150]  # Blue patch
    test_image_rgb[20:80, 220:280] = [160, 82, 45]  # Reddish-brown patch

    print("--- DEMO: Reusing a figure and non-blocking plots ---")
    
    # 2. Create the model instance
    green_model = ColorFamilyModel(sample_green_colors, color_space='rgb')

    # 3. Create a single figure to be reused for all subsequent plots
    reusable_fig = plt.figure(figsize=(15, 7))

    # --- First visualization: Analysis Plot (non-blocking) ---
    print("\nDisplaying initial analysis plot (non-blocking)...")
    green_model.visualize_analysis(
        test_image_rgb, 
        color_space='rgb', 
        title="Analysis of Green Color Family",
        fig=reusable_fig,
        block=False  # Set to False to continue script execution immediately
    )
    
    print("Script continues while plot is open... waiting 4 seconds before updating the figure.")
    plt.pause(4)  # Use plt.pause() to keep the GUI responsive and show the plot

    # --- Second visualization: Interactive Plot (reusing the figure) ---
    print("\nSwitching to interactive thresholding plot in the SAME window...")
    # This call will now clear reusable_fig and draw the interactive plot on it.
    # It will be blocking, so the script waits here until the window is closed.
    green_model.interactive_thresholding(
        test_image_rgb, 
        color_space='rgb',
        title="Interactive Green Detection (Close window to end script)",
        fig=reusable_fig,
        block=True  # Set to True to make it interactive and blocking
    )

    print("\nInteractive plot window closed. Script finished. ✅")