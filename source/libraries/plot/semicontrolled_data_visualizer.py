from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image
import pyglet
import seaborn as sns
from typing import List, Optional, Tuple
import tkinter as tk
from tkinter import ttk
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402


class SemiControlledDataVisualizer:
    def __init__(self, scd=None, *, title="", auto_positioning=False):
        # create the figures
        self.fig2D_TTL = DataVisualizer2D(4, f"{title}: TTLs", auto_positioning=auto_positioning)
        self.fig2D_global = DataVisualizer2D(5, f"{title}: Neuron, Depth, and Area", auto_positioning=auto_positioning)
        
        self.figpos = DataVisualizer3D(f"{title}: Position")

        self.window_positioning()

        if isinstance(scd, SemiControlledData):
            self.update(scd)

    def __del__(self):
        del self.fig2D_TTL
        del self.fig2D_global
        del self.figpos

    def window_positioning(self):
        # Function to get screen size
        def get_screen_size():
            display = pyglet.canvas.Display()
            screen = display.get_default_screen()
            return screen.width, screen.height

        screen_width, screen_height = get_screen_size()

        # Set the positions and sizes of the figures
        fig1_width, fig1_height = screen_width // 2, screen_height
        fig2_width, fig2_height = screen_width // 2, screen_height * 3 // 5
        fig3_width, fig3_height = screen_width // 2, screen_height * 2 // 5

        # Helper function to move and resize the figures
        def move_resize_figure(fig, x, y, width, height):
            backend = plt.get_backend()
            if backend == 'TkAgg':
                fig.canvas.manager.window.wm_geometry(f"{width}x{height}+{x}+{y}")
            elif backend == 'Qt5Agg':
                fig.canvas.manager.window.setGeometry(x, y, width, height)
            elif backend == 'WXAgg':
                fig.canvas.manager.window.SetPosition((x, y))
                fig.canvas.manager.window.SetSize((width, height))
            else:
                print(f"Backend {backend} is not supported")

        # Position the figures
        move_resize_figure(self.figpos.fig, 0, 0, fig1_width, fig1_height)
        move_resize_figure(self.fig2D_global.fig, fig1_width, 0, fig2_width, fig2_height)
        move_resize_figure(self.fig2D_TTL.fig, fig1_width, fig2_height, fig3_width, fig3_height)

    def save(self, output_filename_abs):
        # set correct dimensions
        display = pyglet.canvas.Display()
        screen = display.get_default_screen()
        screenratio = screen.width / screen.height
        dpi = 100
        height = 1080 / dpi
        width = screenratio * height
        self.figpos.fig.set_size_inches(width / 2, height)
        self.fig2D_global.fig.set_size_inches(width / 2, height * 2 / 3)
        self.fig2D_TTL.fig.set_size_inches(width / 2, height * 1 / 3)

        # temporarily save the figures
        self.figpos.fig.savefig("fispos_tmp.png")
        self.fig2D_global.fig.savefig("global_tmp.png")
        self.fig2D_TTL.fig.savefig("ttl_tmp.png")

        # Load the temporary images
        figpos = Image.open("fispos_tmp.png")
        fig2D_global = Image.open("global_tmp.png")
        fig2D_TTL = Image.open("ttl_tmp.png")

        # Get the image dimensions
        width1, height1 = figpos.size
        width2, height2 = fig2D_global.size
        width3, height3 = fig2D_TTL.size

        # Create a new image with the combined height of the three images
        combined_image = Image.new('RGB', (width1 + width2, height1))

        # Get the dimensions of each image
        combined_image.paste(figpos, (0, 0))
        combined_image.paste(fig2D_global, (width1, 0))
        combined_image.paste(fig2D_TTL, (width1, height2))

        # Save the combined image
        combined_image.save(output_filename_abs)

        os.remove("fispos_tmp.png")
        os.remove("global_tmp.png")
        os.remove("ttl_tmp.png")


    def set_lim(self, fig_choice, limits):
        if fig_choice == "Depth and Area":
            self.fig2D_global.set_lim(limits)
        elif fig_choice == "Position":
            self.figpos.set_lim(limits)

    def update(self, scd: SemiControlledData, title=None):
        time = scd.md.time
        info_str = ("Neuron Info\n"
                    f"ID: {scd.neural.unit_id}\n"
                    f"Type: {scd.neural.unit_type}\n"
                    "Stimulus Info\n"
                    f"Type: {scd.stim.type}\n"
                    f"Force: {scd.stim.force}\n"
                    f"Size: {scd.stim.size}\n"
                    f"Velocity: {scd.stim.vel} cm/s")

        self.fig2D_TTL.update(0, time, scd.neural.TTL, 'TTL Nerve')
        self.fig2D_TTL.update(1, time, scd.contact.TTL, 'TTL contact')
        self.fig2D_TTL.update(2, time, scd.neural.TTL, 'both')
        self.fig2D_TTL.update(2, time, scd.contact.TTL, 'both', linestyle="--", reset=False)
        try:
            d = [n - c for n, c in zip(scd.neural.TTL, scd.contact.TTL)]
            self.fig2D_TTL.update(3, time, d, 'diff')
        except:
            pass

        self.fig2D_global.update(0, time, scd.contact.pos_1D, 'Position (Principal Component)', showxlabel=False)
        self.fig2D_global.update(1, time, scd.contact.depth, 'Depth', showxlabel=False)
        self.fig2D_global.update(2, time, scd.contact.area, 'Area size', showxlabel=False)
        self.fig2D_global.update(3, time, scd.neural.iff, 'IFF', showxlabel=False)
        self.fig2D_global.update(4, time, scd.neural.spike, 'Spikes', showxlabel=True)

        self.figpos.update(time, scd.contact.pos, info_str)
        if title is not None:
            self.figpos.fig.suptitle(title)

    def add_vertical_lines(self, xlocs):
        self.fig2D_global.add_vertical_lines(xlocs)

        for ax in self.fig2D_global.axs:
            # Add red vertical lines at specified x-locations
            for xloc in xlocs:
                ax.axvline(x=xloc, color='red', linestyle='--', linewidth=1)


class DataVisualizer2D:
    def __init__(self, nsubplot, title, auto_positioning=False):
        self.title = title
        self.fig, self.axs = plt.subplots(nsubplot, 1, figsize=(10, 12))
        self.fig.tight_layout(pad=5.0)

        if auto_positioning:
            # Get the current figure manager
            manager = plt.get_current_fig_manager()
            # Get the size of the screen
            screen_width, screen_height = manager.window.maxsize()
            # Set the position of the first figure to the right half of the screen
            windows_geometry_depth_n_area = [screen_width // 2, 0, screen_width // 2, 600]
            #if pos is not None:
            #    self.fig.canvas.manager.window.setGeometry(pos[0], pos[1], pos[2], pos[3])

        plt.subplots_adjust(hspace=0.5)
        plt.ion()  # Turn on interactive mode
        self.fig.show()

        self.limits = None

    def __del__(self):
        plt.close(self.fig)

    def set_lim(self, limits):
        self.limits = limits

    def update(self, ax_idx, time: List[float], data: List[float], data_name, linestyle="-", reset=True, showxlabel=True):
        if reset:
            self.axs[ax_idx].clear()

        if not(isinstance(data, np.ndarray)):
            data = np.array(data)
            time = np.array(time)
        
        try:
            if data.size <= 1:
                return
        except:  # means it is not even a list/array
            return

        # Remove NaNs
        valid = ~(np.isnan(data))
        data = data[valid]
        time = time[valid]  # ensure time array is also filtered

        self.axs[ax_idx].plot(time, data, label=data_name, linestyle=linestyle)
        self.axs[ax_idx].set_title(data_name)
        if showxlabel:
            self.axs[ax_idx].set_xlabel('Time')
        self.axs[ax_idx].set_ylabel(data_name)
        self.axs[ax_idx].legend()

        if self.limits is not None:
            try:
                self.axs[ax_idx].set_ylim(self.limits[ax_idx][0], self.limits[ax_idx][1])
            except:
                pass

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_vertical_lines(self, xlocs):
        pass


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection # <--- CORRECT

# --- Imports potentially needed if running standalone or embedding ---
# import tkinter as tk # If embedding in Tkinter
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # If embedding

class DataVisualizer3D:
    """
    A class for creating and updating 3D scatter/line plots from (3, N) data arrays.

    Uses constrained_layout to help keep the plot centered during window resizing.
    Allows optional blocking behaviour after each update.
    """
    def __init__(self, title: str = "3D Plot", dot_size: int = 20,
                 block_on_update: bool = False, # Option to control blocking
                 figsize: Tuple[int, int] = (10, 8)):
        """
        Initializes the 3D plotter.

        Args:
            title (str): The title of the plot.
            dot_size (int): The size of the scatter plot markers.
            block_on_update (bool): If True, plt.show(block=True) is called
                                    at the end of each update(), pausing script
                                    execution until the plot window is closed.
                                    If False (default), uses non-blocking
                                    plt.draw() and plt.pause() to update.
            figsize (Tuple[int, int]): The figure size (width, height) in inches.
        """
        # --- Setup the figure ---

        # Turn interactive mode on if not blocking, often needed for plt.pause
        # Do this *before* creating the figure if possible
        if not block_on_update:
            plt.ion() # Turn interactive mode ON for non-blocking updates
        else:
            plt.ioff() # Ensure interactive mode is OFF for reliable blocking

        # *** MODIFICATION: Enable constrained_layout ***
        # This automatically adjusts plot elements to prevent overlap and
        # helps maintain centering/layout during resizes.
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        # self.fig = plt.figure(figsize=figsize) # Original line

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.title = title
        self.dot_size = dot_size
        self.block_on_update: bool = block_on_update # Store the option
        self.cbar = None # Initialize colorbar tracker
        self.limits = None # Initialize limits
        # Store scatter and line collection to potentially update data later (advanced)
        self._scatter = None
        self._line = None

        # --- Notes on Embedding (if applicable) ---
        # The following lines seem intended for Tkinter embedding but are misplaced
        # and incomplete in the original snippet (e.g., self.root not defined).
        # If you intend to embed this in Tkinter, you would typically:
        # 1. Create a Tkinter root window BEFORE this class or pass it in.
        # 2. Create the FigureCanvasTkAgg *after* self.fig is created.
        # 3. Pack/grid the canvas widget into the Tkinter window.
        # 4. Bind the resize event (<Configure>) to the Tkinter window/frame
        #    containing the canvas, and potentially call self.fig.canvas.draw_idle()
        #    in the handler (though constrained_layout often makes this automatic).
        #
        # self.root = tk.Tk() # Example: Need a root window
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.root) # Create canvas AFTER fig
        # self.canvas.draw() # Initial draw
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Place canvas
        # self.root.bind("<Configure>", self.on_resize) # Bind resize
        # --- End Notes on Embedding ---

    def __del__(self):
        plt.close(self.fig)
        
    # --- MODIFIED UPDATE METHOD SIGNATURE AND DATA HANDLING --- (No changes needed here for layout)
    def update(self, time_arr: np.ndarray, data: np.ndarray, # Changed type hint for data
               info_str: Optional[str] = None,
               colorsMap: str = 'viridis_r', withLine: bool = True):
        """
        Updates the 3D plot with new data.

        Args:
            time_arr (np.ndarray): 1D array of time values corresponding to data points.
            data (np.ndarray): Array of shape (3, N) containing X, Y, Z coordinates
                               in rows 0, 1, 2 respectively.
            info_str (Optional[str]): Optional text to display on the figure.
            colorsMap (str): Colormap name for time gradient.
            withLine (bool): Whether to draw lines connecting points in time order.

        Depending on the 'block_on_update' setting during initialization,
        this method will either block execution until the plot is closed
        or update the plot non-blockingly.
        """
        # Clear previous elements carefully
        if self.fig.texts:
            # Filter out colorbar text which should not be removed
            texts_to_remove = [t for t in self.fig.texts if not hasattr(t, '_colorbar')]
            for text in texts_to_remove:
                try:
                    text.remove()
                except Exception: # Handle cases where text might already be gone
                    pass
        self.ax.clear() # Clear axes content (scatter, line, etc.)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # --- DATA VALIDATION AND UNPACKING (Unchanged) ---
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[0] != 3:
            print(f"Warning: Invalid data provided. Expected a NumPy array of shape (3, N), but got shape {getattr(data, 'shape', 'N/A')}.")
            self._draw_or_show()
            return # Exit if data is bad

        num_points = data.shape[1]
        if num_points == 0: # Check if N=0 (no points)
            print("Warning: Data array has shape (3, 0), no points to plot.")
            self._draw_or_show()
            return # Exit if no points

        time_arr = np.asarray(time_arr)
        if time_arr.ndim != 1 or time_arr.shape[0] != num_points:
            print(f"Warning: time_arr length ({time_arr.shape[0]}) does not match number of data points ({num_points}).")
            self._draw_or_show()
            return

        x = data[0, :]
        y = data[1, :]
        z = data[2, :]
        # --- END DATA HANDLING ---

        time_arr = np.asarray(time_arr)

        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(time_arr))
        if not np.any(valid_mask):
            print("Warning: No valid (non-NaN) data points found after filtering.")
            self._draw_or_show()
            return # Exit if no valid data

        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        time_arr = time_arr[valid_mask]

        # --- Plotting Logic (Unchanged) ---
        self._scatter = self.ax.scatter(x, y, z, c=time_arr, cmap=colorsMap, s=self.dot_size)
        if len(x) >= 2: # Check length after filtering NaNs
             # Highlight start and end points (optional, kept from original)
             # Consider checking if x,y,z are non-empty before indexing
            if len(x) > 0:
                self.ax.scatter(x[0], y[0], z[0], c='lime', s=self.dot_size*1.5, label='Start', depthshade=False, edgecolors='black')
                self.ax.scatter(x[-1], y[-1], z[-1], c='red', s=self.dot_size*1.5, label='End', depthshade=False, edgecolors='black')

        min_time = np.min(time_arr) if time_arr.size > 0 else 0
        max_time = np.max(time_arr) if time_arr.size > 0 else 1

        if withLine and len(x) > 1:
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm_min = min_time
            norm_max = max_time if abs(max_time - min_time) > 1e-9 else min_time + 1e-6
            norm = plt.Normalize(norm_min, norm_max)
            self._line = Line3DCollection(segments, cmap=colorsMap, norm=norm)
            self._line.set_array(time_arr[:-1])
            self.ax.add_collection(self._line)
        elif withLine and len(x) <= 1:
            print("Info: Cannot draw line with less than 2 points.")
            self._line = None
        else:
            self._line = None

        # --- Add or update the color bar (Unchanged, but constrained_layout helps position it) ---
        if abs(min_time - max_time) < 1e-9 and time_arr.size > 0 :
             norm_for_cbar = plt.Normalize(min_time - 0.5, max_time + 0.5)
             ticks_for_cbar = [min_time]
             if self._scatter:
                 self._scatter.set_norm(norm_for_cbar)
        elif time_arr.size > 0:
             norm_for_cbar = plt.Normalize(min_time, max_time)
             ticks_for_cbar = np.linspace(min_time, max_time, num=6)
        else: # Handle case with no valid data for colorbar
             norm_for_cbar = plt.Normalize(0, 1)
             ticks_for_cbar = [0, 1]


        mappable = self._scatter # Use scatter for color mapping reference

        if mappable is not None:
             # --- Improved Colorbar Handling ---
             # Remove existing colorbar before creating/updating if it exists
             # This is safer when axes are cleared and limits/data might change drastically
             if self.cbar is not None:
                 try:
                     self.cbar.remove()
                 except Exception as e:
                     print(f"Info: Could not remove previous colorbar: {e}")
                 self.cbar = None

             # Create a new colorbar
             try:
                 # Use the mappable's norm and cmap by default
                 self.cbar = self.fig.colorbar(mappable, ax=self.ax, shrink=0.7, aspect=20) # Adjust shrink/aspect as needed
                 self.cbar.set_label('Time')
                 # Set ticks explicitly AFTER creation if needed (e.g., for single-value case)
                 self.cbar.set_ticks(ticks_for_cbar)
                 tick_str = [f'{tick:.1f}' for tick in ticks_for_cbar]
                 self.cbar.set_ticklabels(tick_str)

             except Exception as e:
                 print(f"Warning: Could not create/update colorbar: {e}")
                 self.cbar = None
             # --- End Improved Colorbar Handling ---
        elif self.cbar is not None: # Remove orphan colorbar if scatter failed
             try:
                self.cbar.remove()
             except Exception as e:
                print(f"Info: Could not remove orphan colorbar: {e}")
             self.cbar = None
        # --- End color bar ---

        self.title = info_str

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.set_title(self.title)

        if self.limits is not None:
            self.ax.set_xlim(self.limits[0][0], self.limits[0][1])
            self.ax.set_ylim(self.limits[1][0], self.limits[1][1])
            self.ax.set_zlim(self.limits[2][0], self.limits[2][1])
        elif time_arr.size > 0:
             # Autoscale only if limits aren't fixed AND there's data
             # Constrained layout works better if axes limits are determined
             # *before* drawing, so autoscale_view might not be strictly needed
             # self.ax.autoscale_view(tight=False) # Can sometimes interfere with constrained_layout

             # Instead of autoscale_view, ensure equal aspect ratio if desired
             # This helps maintain the 'shape' of the data visually
             # self.ax.set_aspect('equal', adjustable='box') # Uncomment if aspect ratio is important
             pass # Let constrained_layout handle spacing with default limits


        if info_str is not None:
            self.fig.text(0.02, 0.02, info_str, verticalalignment='bottom', fontsize=10, transform=self.fig.transFigure)

        self._draw_or_show()

    def set_axis_limits(self, limits: List[List[float]]):
        """Optional: Set fixed axis limits e.g., [[xmin, xmax], [ymin, ymax], [zmin, zmax]]"""
        if len(limits) == 3 and all(len(lim) == 2 for lim in limits):
            self.limits = limits
            if hasattr(self, 'ax'):
                self.ax.set_xlim(self.limits[0][0], self.limits[0][1])
                self.ax.set_ylim(self.limits[1][0], self.limits[1][1])
                self.ax.set_zlim(self.limits[2][0], self.limits[2][1])
        else:
            print("Warning: Invalid limits format. Expected [[xmin, xmax], [ymin, ymax], [zmin, zmax]]")
            self.limits = None # Reset if format is wrong


    # This method is likely needed if you bind to <Configure> in Tkinter
    # def on_resize(self, event):
    #     """Callback for window resize events (if embedded)."""
    #     # Constrained_layout often handles this automatically.
    #     # If not, uncommenting draw_idle might help.
    #     # print("Window resized") # For debugging
    #     # self.fig.canvas.draw_idle()
    #     pass


    def _draw_or_show(self):
        """Internal helper to handle drawing/showing based on the block_on_update flag."""
        if self.block_on_update:
            print("Displaying blocking figure. Close the window to continue...")
            # plt.ioff() # Already handled in init/close
            try:
                plt.show(block=True)
            except Exception as e:
                # Catch errors if the figure was closed externally, etc.
                print(f"Info: plt.show() interaction ended or failed: {e}")
        else:
            # Non-blocking update
            # plt.ion() # Already handled in init
            try:
                # draw_idle is preferred for GUI event loops
                self.fig.canvas.draw_idle()
                # Pause allows the GUI backend to process events and render the update
                plt.pause(0.01) # Adjust pause duration if needed
            except Exception as e:
                # This commonly happens if the user closes the window interactively
                print(f"Info: Could not draw/pause figure (may have been closed): {e}")

    def close(self):
        """Closes the plot figure."""
        if hasattr(self, 'fig') and self.fig:
             # Check if the figure's canvas manager exists (i.e., window is open)
            if self.fig.canvas.manager is not None:
                plt.close(self.fig)
                print("Plot figure closed.")
            # Reset figure attribute to prevent trying to close again
            self.fig = None
        # Turn interactive mode off if it was on
        if plt.isinteractive():
             plt.ioff()






def display_scd_one_by_one(scd_list):
    scd_visualizer = SemiControlledDataVisualizer()

    # set up uniform limits to compare the trials
    pos = np.concatenate([s.contact.pos_1D for s in scd_list])
    depth = np.concatenate([s.contact.depth for s in scd_list])
    area = np.concatenate([s.contact.area for s in scd_list])
    limits = [[min(pos), max(pos)], [min(depth), max(depth)], [min(area), max(area)]]
    scd_visualizer.set_lim("Position, Depth and Area", limits)

    pos = np.concatenate([s.contact.pos for s in scd_list], axis=1)
    limits = [(min(_axis), max(_axis)) for _axis in pos]
    scd_visualizer.set_lim("Position", limits)

    vel = np.concatenate([s.contact.vel for s in scd_list], axis=1)
    limits = [(min(_axis), max(_axis)) for _axis in vel]
    scd_visualizer.set_lim("Velocity", limits)

    fig = Figure(figsize=(4, .5), dpi=100)
    current_index = 0

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Dataset Viewer")

    # Define button callback function
    def next_dataset():
        nonlocal current_index
        current_index = (current_index + 1) % len(scd_list)
        scd_visualizer.update(scd_list[current_index])

    # Add a button to the Tkinter window
    button = ttk.Button(root, text="Next", command=next_dataset)
    button.pack(pady=20)

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Initial plot
    scd_visualizer.update(scd_list[current_index])
    # Run the Tkinter main loop
    root.mainloop()


def display_attribute(df, selection=0, title=None):
    label_size = 20
    tick_size = 17
    legend_size = 14

    duos = [["estimated_velocity", "expected_velocity"],
            ["estimated_depth", "expected_depth"],
            ["estimated_area", "expected_area"]]
    current_feature = duos[selection][0]
    current_feature_expected = duos[selection][1]

    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')
    fig.set_size_inches(8, 5, forward=True)

    # stroke
    try:
        idx_tap = (df['contact_type'].values == "stroke")
        df_current = df[idx_tap]
        ax = axes[0]
        palette = sns.color_palette('Set2', n_colors=len(df_current[current_feature_expected].unique()))
        sns.histplot(df_current, x=current_feature, hue=current_feature_expected,
                     bins=50, palette=palette, multiple="stack", ax=ax)
        if title is not None:
            ax.set_title(title, size=label_size)
        else:
            ax.set_title(current_feature + '_stroke', size=label_size)
        ax.set_xlabel('', fontsize=label_size)
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
        ax.yaxis.set_tick_params(labelsize=tick_size)
    except:
        pass

    # tap
    try:
        idx_tap = (df['contact_type'].values == "tap")
        df_current = df[idx_tap]
        ax = axes[1]
        palette = sns.color_palette('Set2', n_colors=len(df_current[current_feature_expected].unique()))
        sns.histplot(df_current, x=current_feature, hue=current_feature_expected,
                     bins=50, palette=palette, multiple="stack", ax=ax)
        if title is not None:
            ax.set_title(title, size=label_size)
        else:
            ax.set_title(current_feature + '_tap', size=label_size)
        ax.set_xlabel('', fontsize=label_size)
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.set_tick_params(labelsize=tick_size, rotation=0)
        ax.yaxis.set_tick_params(labelsize=tick_size)
    except:
        pass

    plt.ion()
    plt.show(block=False)
    #plt.draw()
