import tkinter as tk
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class Hand3DViewer:
    """
    Dedicated class for managing the 3D visualization window using Matplotlib.
    Maintains the Toplevel window separate from the main selection GUI.
    """
    def __init__(self, master: tk.Tk):
        self.window = tk.Toplevel(master)
        self.window.title("3D Hand Model Analysis")
        # Position the 3D window slightly offset so it doesn't block the main view initially
        self.window.geometry("500x500+50+50") 
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.fig = plt.figure(figsize=(5, 5))
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')
        self._configure_axes()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def _configure_axes(self):
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')

    def _on_closing(self):
        self.window.withdraw()  # Hide instead of destroy to allow re-opening logic if needed

    def _set_axes_equal(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, global_max_range: Optional[float] = None):
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5

        if global_max_range is not None:
            max_range = global_max_range
        else:
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        
        self.ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        self.ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        self.ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    def update_3d_plot(self, geometry: Optional[Tuple[np.ndarray, np.ndarray]], global_max_range: Optional[float] = None):
        self.ax.cla()
        self._configure_axes()
        self.ax.set_title("3D Reconstruction")

        if geometry is None:
            self.ax.text2D(0.5, 0.5, "No Data", transform=self.ax.transAxes, ha='center')
        else:
            vertices, faces = geometry
            X, Y, Z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            
            self.ax.plot_trisurf(X, Y, Z, triangles=faces, cmap='viridis', 
                                 edgecolor='gray', linewidth=0.5, alpha=0.7)
            
            self._set_axes_equal(X, Y, Z, global_max_range=global_max_range)

        self.canvas.draw()
        
        if self.window.state() == 'withdrawn':
            self.window.deiconify()