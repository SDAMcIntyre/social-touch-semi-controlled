import numpy as np
import matplotlib.pyplot as plt

class ColorCorrelationVisualizer:
    """
    A class to efficiently visualize the color correlation analysis for a stream of images.
    
    This visualizer sets up the matplotlib figure and artists once during initialization.
    The `update` method then efficiently updates the displayed data without redrawing
    the entire plot, which is suitable for real-time video processing.
    """
    def __init__(self, model, title="Color Correlation Analysis"):
        """
        Initializes the visualizer and sets up the plot structure.

        Args:
            model: The analysis model instance which has the `calculate_mahalanobis_map`
                   and `sample_colors_rgb` attributes.
            title (str): The main title for the plot window.
        """
        self.model = model
        
        # --- One-time plot setup ---
        plt.ion() 
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle(title, fontsize=16)

        # Plot 1: Sample Colors (static, drawn only once)
        self.axes[0].set_title('Sample Colors (as RGB)')
        self.axes[0].imshow(self.model.sample_colors_rgb.reshape(-1, 1, 3))
        self.axes[0].set_xticks([]); self.axes[0].set_yticks([])

        # Plot 2: Original Test Image (placeholder)
        self.axes[1].set_title('Original Test Image')
        self.axes[1].axis('off')
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        self.im_test = self.axes[1].imshow(dummy_image)

        # Plot 3: Correlation Map (placeholder)
        self.axes[2].set_title('Correlation Map (Mahalanobis Distance)')
        self.axes[2].axis('off')
        dummy_map = np.zeros((10, 10))
        self.im_corr = self.axes[2].imshow(dummy_map, cmap='hot_r', vmin=0, vmax=1)

        self.cbar = self.fig.colorbar(self.im_corr, ax=self.axes[2], orientation='vertical')
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        self.fig.canvas.draw()
        plt.show()


    def update(self, test_image_bgr: np.ndarray):
        """
        Updates the visualizer with a new test image.

        Args:
            test_image_bgr (np.ndarray): The new HxWx3 BGR image to process and display.
        """
        test_image_rgb = test_image_bgr[..., ::-1]
        correlation_map = self.model.calculate_mahalanobis_map(test_image_bgr, color_space='bgr')
        
        self.im_test.set_data(test_image_rgb)
        self.im_corr.set_data(correlation_map)
        self.im_corr.set_clim(vmin=correlation_map.min(), vmax=correlation_map.max())
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Closes the matplotlib figure."""
        plt.close(self.fig)