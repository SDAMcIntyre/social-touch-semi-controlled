import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Slider, Button
import logging

logger = logging.getLogger(__name__)


class Trajectory3DVisualizer:
    """
    A generic class to visualize 3D trajectory data.
    Completely agnostic of data source (CSV/Database) or column naming conventions.
    """

    def __init__(self, data: np.ndarray, initial_window: int = 500):
        """
        Initialize the visualizer with raw 3D data.

        Args:
            data (np.ndarray): A numpy array of shape (N, 3) representing X, Y, Z coordinates.
            initial_window (int): The default number of points to show in the sliding window.
        """
        # Validate Input
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Data shape must be (N, 3). Received {data.shape}.")

        # Remove NaNs to ensure continuity in visualization
        self.data = data[~np.isnan(data).any(axis=1)]
        self.total_points = len(self.data)
        self.initial_window = min(initial_window, self.total_points)

        if self.total_points < 2:
            logger.error("Insufficient data points to plot.")
            return

        # Pre-calculate Global Context (for static axis limits)
        self.x_min, self.x_max = self.data[:, 0].min(), self.data[:, 0].max()
        self.y_min, self.y_max = self.data[:, 1].min(), self.data[:, 1].max()
        self.z_min, self.z_max = self.data[:, 2].min(), self.data[:, 2].max()

        # Global time array for coloring
        self.global_t_vals = np.linspace(0, 1, self.total_points)
        self.norm = plt.Normalize(0, 1)

        # Initialize internal state variables
        self.fig = None
        self.ax = None
        self.lc = None
        self.title_text = None
        self.slider_window = None
        self.slider_start = None
        self.btn_recenter = None

    def _setup_plot(self):
        """Sets up the figure, axes, and initial plot elements."""
        self.fig = plt.figure(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.30)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initial slice
        s_idx = 0
        e_idx = self.initial_window
        
        data_subset = self.data[s_idx:e_idx]
        t_subset = self.global_t_vals[s_idx:e_idx]

        # Create Segments
        points = data_subset.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Line Collection
        self.lc = Line3DCollection(segments, cmap='jet', norm=self.norm)
        self.lc.set_array(t_subset[:-1])
        self.lc.set_linewidth(1.5)
        self.lc.set_alpha(0.8)
        self.ax.add_collection(self.lc)

        # Axis Labels and Limits
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_zlim(self.z_min, self.z_max)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.title_text = self.ax.set_title(f'Window: {s_idx} - {e_idx} ({e_idx - s_idx} pts)')

        # Colorbar
        cbar = plt.colorbar(self.lc, ax=self.ax, shrink=0.6, pad=0.1)
        cbar.set_label('Global Time Progression (0.0=Start, 1.0=End)')

    def _setup_widgets(self):
        """Configures the interactive sliders and buttons."""
        # Window Size Slider
        ax_window = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider_window = Slider(
            ax_window, 'Window Size', 
            valmin=10, 
            valmax=self.total_points, 
            valinit=self.initial_window, 
            valstep=10
        )

        # Start Index Slider
        ax_start = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider_start = Slider(
            ax_start, 'Start Index', 
            valmin=0, 
            valmax=self.total_points - 10, 
            valinit=0, 
            valstep=1
        )

        # Recenter Button
        ax_button = plt.axes([0.8, 0.15, 0.1, 0.04])
        self.btn_recenter = Button(ax_button, 'Recenter View', hovercolor='0.975')

        # Connect callbacks
        self.slider_window.on_changed(self._update)
        self.slider_start.on_changed(self._update)
        self.btn_recenter.on_clicked(self._recenter_view)

    def _get_current_indices(self):
        w_size = int(self.slider_window.val)
        s_idx = int(self.slider_start.val)
        
        if s_idx + w_size > self.total_points:
            s_idx = self.total_points - w_size
        
        e_idx = s_idx + w_size
        return s_idx, e_idx

    def _update(self, val):
        """Callback for slider updates."""
        s_idx, e_idx = self._get_current_indices()

        # Slice Data
        new_data = self.data[s_idx:e_idx]
        new_t = self.global_t_vals[s_idx:e_idx]

        if len(new_data) < 2:
            self.lc.set_segments([])
            self.lc.set_array(np.array([]))
            self.title_text.set_text(f'Window: {s_idx} - {e_idx} (Insufficient data)')
            self.fig.canvas.draw_idle()
            return

        # Update Line Collection
        new_pts = new_data.reshape(-1, 1, 3)
        new_segs = np.concatenate([new_pts[:-1], new_pts[1:]], axis=1)
        
        self.lc.set_segments(new_segs)
        self.lc.set_array(new_t[:-1]) 
        self.title_text.set_text(f'Window: {s_idx} - {e_idx} ({e_idx - s_idx} pts)')
        
        self.fig.canvas.draw_idle()

    def _recenter_view(self, event):
        """Recalculates axis limits based on the currently displayed data window."""
        s_idx, e_idx = self._get_current_indices()
        current_slice = self.data[s_idx:e_idx]
        
        if len(current_slice) < 2:
            return
        
        pad_factor = 0.05 
        mins = current_slice.min(axis=0)
        maxs = current_slice.max(axis=0)
        ranges = maxs - mins
        
        for i in range(len(ranges)):
            if ranges[i] == 0:
                ranges[i] = 1.0 
                mins[i] -= 0.5 
                maxs[i] += 0.5

        self.ax.set_xlim(mins[0] - ranges[0]*pad_factor, maxs[0] + ranges[0]*pad_factor)
        self.ax.set_ylim(mins[1] - ranges[1]*pad_factor, maxs[1] + ranges[1]*pad_factor)
        self.ax.set_zlim(mins[2] - ranges[2]*pad_factor, maxs[2] + ranges[2]*pad_factor)
        
        logger.info(f"View recentered on window {s_idx}:{e_idx}")
        self.fig.canvas.draw_idle()

    def show(self):
        """Display the plot."""
        if self.total_points < 2:
            logger.warning("Cannot show plot: Insufficient data.")
            return

        self._setup_plot()
        self._setup_widgets()
        logger.info("Displaying interactive plot...")
        plt.show()
