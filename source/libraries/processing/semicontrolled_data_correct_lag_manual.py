import tkinter as tk
from tkinter import Scale, Button, Frame, Checkbutton, BooleanVar # Added Checkbutton, BooleanVar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import re

class TimeSeriesLagGUI:
    """
    A GUI for visualizing and adjusting the lag and offset between a shiftable time series
    and one or more reference time series.
    Includes zoom/pan toolbar. Plot updates when slider handle is released.
    Reference signal visibility can be toggled with checkboxes on the right.

    Attributes:
        lag_samples (int | None): The lag in samples selected by the user when
                                  the 'Save Lag and Close' button is pressed.
                                  None if the window was closed otherwise.
        lag_seconds (float | None): The lag in seconds calculated from lag_samples
                                    and the sampling frequency (fs).
                                    None if the window was closed otherwise.
    """

    def __init__(self, ref_signals, shift_signal,
                 ref_signal_labels=None, shift_signal_label="",
                 offset_references=False,
                 lag_samples=None, fs=1000,
                 title=None, master=None):
        """
        Initializes the GUI.

        Args:
            ref_signals (list of array-like): A list containing the reference time series vectors.
                                              If a single array-like is passed, it will be treated as a list with one element.
            shift_signal (array-like): The time series vector to be shifted.
            ref_signal_labels (list of str, optional): Labels for the reference time series.
                                                       Defaults to ["Reference 1", "Reference 2", ...].
                                                       Must have the same length as ref_signals if provided.
            shift_signal_label (str, optional): Label for the shiftable time series. Defaults to "Shiftable Signal".
            lag_samples (int, optional): Initial lag value in samples. Defaults to None (meaning 0).
            fs (int): The sampling frequency of the time series in Hz. Default is 1000.
            master (tk.Tk | tk.Toplevel | tk.Frame | None): The parent window or frame.
                               If None, a new Toplevel window is created.
        """
        if master is None:
            self.root = tk.Toplevel()
            # Use a more descriptive default title if none provided, and incorporate user title
            effective_title = "Time Series Lag Adjustment"
            if title:
                effective_title += f": {title}"
            self.root.title(effective_title)
            is_toplevel = True
        else:
            self.root = Frame(master)
            self.root.pack(fill=tk.BOTH, expand=True)
            is_toplevel = False

        # --- Input Handling ---
        if not isinstance(ref_signals, (list, tuple)):
            self.ref_signals_raw = [np.asarray(ref_signals)]
        else:
            self.ref_signals_raw = [np.asarray(rs) for rs in ref_signals]

        self.shift_signal_raw = np.asarray(shift_signal)
        self.fs = fs
        self.initial_lag_samples = lag_samples if lag_samples is not None else 0
        self.lag_samples = None
        self.lag_seconds = None
        self.max_v_offset = 0.5 # Max vertical offset for the shiftable signal slider

        num_ref_signals = len(self.ref_signals_raw)
        if ref_signal_labels is None or len(ref_signal_labels) != num_ref_signals:
            self.ref_signal_labels = [f"Reference {i+1}" for i in range(num_ref_signals)]
            if ref_signal_labels is not None:
                print("Warning: ref_signal_labels length mismatch or None. Using default labels.")
        else:
            self.ref_signal_labels = ref_signal_labels

        self.shift_signal_label = shift_signal_label if shift_signal_label else "Shiftable Signal"

        # --- Normalize Data ---
        self.ref_signals_norm = [self._normalize(rs) for rs in self.ref_signals_raw]
        self.shift_signal_norm = self._normalize(self.shift_signal_raw)

        # --- Determine Plotting Length & Time Axis ---
        all_signals_norm = self.ref_signals_norm + [self.shift_signal_norm]
        # Robust check for non-empty signals before finding min_len
        min_len = min(len(s) for s in all_signals_norm if len(s) > 0) if any(len(s) > 0 for s in all_signals_norm) else 0
        if min_len == 0:
            print("Warning: All input signals have zero or insufficient length for plotting.")
            min_len = 1 # Avoid division by zero for time_axis if fs is also 0

        self.time_axis = np.arange(min_len) / self.fs if self.fs > 0 else np.arange(min_len)

        self.ref_signals_plot = [rs_norm[:min_len].copy() for rs_norm in self.ref_signals_norm]
        if offset_references and len(self.ref_signals_plot) > 0 : # Apply offsets if enabled and signals exist
            # Spread reference signals vertically for better distinction
            # Use a portion of max_v_offset to prevent them from going too far off if they are also shifted by v_offset_slider
            spread_factor = 0.8 # How much of the [0,1] range to use for spreading centers
            if len(self.ref_signals_plot) > 1:
                base_offsets = np.linspace(-0.5 * spread_factor, 0.5 * spread_factor, len(self.ref_signals_plot))
            else: # Single reference signal, no offset needed relative to others
                base_offsets = [0]
            
            for i, ref_ts in enumerate(self.ref_signals_plot):
                # This offset is relative to the normalized [0,1] space of each signal
                self.ref_signals_plot[i] = ref_ts + base_offsets[i]

        self.shift_signal_plot_base = self.shift_signal_norm[:min_len].copy()

        # --- Setup Matplotlib Figure ---
        self.fig, self.ax = plt.subplots(figsize=(12, 7)) # Adjusted size for more controls
        plt.close(self.fig)

        self.ref_lines = []
        for i, ref_ts_plot_data in enumerate(self.ref_signals_plot):
            line, = self.ax.plot(self.time_axis, ref_ts_plot_data, label=self.ref_signal_labels[i], linewidth=1.2)
            self.ref_lines.append(line)

        initial_shifted_data = np.full_like(self.shift_signal_plot_base, np.nan)
        self.shift_line, = self.ax.plot(self.time_axis, initial_shifted_data, label=f"{self.shift_signal_label} (Shifted)", color='orangered', linewidth=1.5)

        self.ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1), fancybox=True, shadow=True, ncol=max(1, num_ref_signals // 2 +1)) # Move legend below plot
        self.fig.subplots_adjust(bottom=0.2) # Adjust bottom to make space for legend

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Normalized Amplitude")
        
        if title: # Append user's specific title if provided
            self.ax.set_title(title)
        else:
            self.ax.set_title(rf"Adjust Lag and Offset")
        self.ax.grid(True)

        if len(self.time_axis) > 1:
            self.ax.set_xlim(self.time_axis[0], self.time_axis[-1])
        elif min_len > 0 : # Handle single point or very short data
             self.ax.set_xlim(-0.5/self.fs if self.fs > 0 else -0.5, 0.5/self.fs if self.fs > 0 else 0.5)


        # Adjust Y limits based on data range and potential offsets
        y_min_overall = 0  # Normalized data is [0,1]
        y_max_overall = 1
        if self.ref_signals_plot: # Consider reference signals if they exist
            temp_min_refs = [np.nanmin(rs) for rs in self.ref_signals_plot if rs.size > 0]
            temp_max_refs = [np.nanmax(rs) for rs in self.ref_signals_plot if rs.size > 0]
            if temp_min_refs: y_min_overall = min(y_min_overall, min(temp_min_refs))
            if temp_max_refs: y_max_overall = max(y_max_overall, max(temp_max_refs))

        # Account for the v_offset slider range for the shiftable signal
        y_min_lim = y_min_overall - self.max_v_offset
        y_max_lim = y_max_overall + self.max_v_offset
        y_range = y_max_lim - y_min_lim
        y_padding = 0.1 * y_range if y_range > 0 else 0.1 # 10% padding
        
        self.ax.set_ylim(y_min_lim - y_padding, y_max_lim + y_padding)

        # --- Embed Plot in Tkinter ---
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame to hold plot and its direct controls (sliders, checkboxes)
        plot_controls_frame = Frame(main_frame)
        plot_controls_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Vertical offset slider on the LEFT
        self.v_offset_var = tk.DoubleVar(value=0.0)
        self.v_offset_slider = Scale(plot_controls_frame, from_=self.max_v_offset, to=-self.max_v_offset,
                                     resolution=0.01, orient=tk.VERTICAL, label="V Offset",
                                     variable=self.v_offset_var, length=300) # Adjusted length
        self.v_offset_slider.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=(35, 5)) # pady to align with plot top
        self.v_offset_slider.bind("<ButtonRelease-1>", self._update_plot)

        # Frame for reference signal toggle Checkbuttons on the RIGHT (NEW)
        self.ref_toggle_frame = Frame(plot_controls_frame)
        self.ref_toggle_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=(35,5))

        self.ref_visibility_vars = [] # Stores BooleanVar for each checkbox
        for i, label in enumerate(self.ref_signal_labels):
            var = BooleanVar(value=True) # Default to visible
            self.ref_visibility_vars.append(var)
            # Truncate long labels for checkbox display
            display_label = label if len(label) < 20 else label[:17] + "..."
            cb = Checkbutton(self.ref_toggle_frame, text=display_label, variable=var,
                                command=self._toggle_reference_visibility)
            cb.pack(anchor='w', fill='x', pady=2) # anchor='w' for left alignment in frame

        # Canvas and Toolbar frame in the CENTER
        canvas_toolbar_frame = Frame(plot_controls_frame)
        # This pack makes it fill the remaining space AFTER left and right elements are placed
        canvas_toolbar_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_toolbar_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True) # Canvas below toolbar

        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X) # Toolbar above canvas

        self.canvas.draw()

        # Horizontal lag slider BELOW the plot_controls_frame
        slider_frame = Frame(main_frame)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.max_lag_range_samples = int(3 * self.fs) if self.fs > 0 else 300 # Default if fs is 0
        clamped_initial_lag = max(-self.max_lag_range_samples, min(self.max_lag_range_samples, self.initial_lag_samples))
        self.lag_var = tk.IntVar(value=clamped_initial_lag)

        self.h_lag_slider = Scale(slider_frame, from_=-self.max_lag_range_samples, to=self.max_lag_range_samples,
                                  orient=tk.HORIZONTAL, label="Horizontal Lag (samples)",
                                  variable=self.lag_var, length=600) # Adjust length as needed
        self.h_lag_slider.pack(fill=tk.X, expand=True)
        self.h_lag_slider.bind("<ButtonRelease-1>", self._update_plot)

        # Buttons at the BOTTOM
        button_frame = Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.save_button = Button(button_frame, text="Save Lag and Close", command=self._save_and_close)
        self.save_button.pack(side=tk.RIGHT, padx=(5, 5), pady=5)

        self.close_button = Button(button_frame, text="Close", command=self._close_window)
        self.close_button.pack(side=tk.RIGHT, padx=(0, 5), pady=5)

        self._update_plot() # Draw the initial state

        if is_toplevel:
            self.root.update_idletasks()
            try:
                # Center the Toplevel window
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                x_screen = self.root.winfo_screenwidth()
                y_screen = self.root.winfo_screenheight()
                x = (x_screen // 2) - (width // 2)
                y = (y_screen // 2) - (height // 2)
                x = max(0, x) # Prevent positioning off-screen
                y = max(0, y) # Prevent positioning off-screen
                self.root.geometry(f'{width}x{height}+{x}+{y}')
            except tk.TclError:
                print("Warning: Could not get window dimensions for centering.") # Should not happen if update_idletasks is called
            self.root.protocol("WM_DELETE_WINDOW", self._close_window)
            self.root.grab_set()
            self.root.wait_window(self.root)

    def _normalize(self, data):
        data = np.asarray(data, dtype=float)
        data_finite = data[np.isfinite(data)]
        if len(data_finite) == 0:
            return np.full_like(data, 0.5, dtype=float) # All NaNs or empty becomes mid-value
        min_val = np.min(data_finite)
        max_val = np.max(data_finite)
        if max_val == min_val: # Constant data
            norm_data = np.full_like(data, np.nan, dtype=float) # Preserve NaNs
            norm_data[np.isfinite(data)] = 0.5 # Set finite values to 0.5
            return norm_data
        else:
            norm_data = np.full_like(data, np.nan, dtype=float) # Preserve NaNs
            finite_mask = np.isfinite(data)
            norm_data[finite_mask] = (data[finite_mask] - min_val) / (max_val - min_val)
            return norm_data

    def _update_plot(self, event=None):
        current_lag_samples = self.lag_var.get()
        v_offset = self.v_offset_var.get()
        shifted_data = np.full_like(self.shift_signal_plot_base, np.nan)
        source_data = self.shift_signal_plot_base
        data_len = len(source_data)

        if data_len > 0:
            if current_lag_samples >= 0: # Positive lag: shift data to the right
                if current_lag_samples < data_len:
                    target_slice = slice(current_lag_samples, data_len)
                    source_slice = slice(0, data_len - current_lag_samples)
                    shifted_data[target_slice] = source_data[source_slice]
            else: # Negative lag: shift data to the left
                abs_lag = abs(current_lag_samples)
                if abs_lag < data_len:
                    target_slice = slice(0, data_len - abs_lag)
                    source_slice = slice(abs_lag, data_len)
                    shifted_data[target_slice] = source_data[source_slice]
        
        # Apply vertical offset only to non-NaN values
        final_plot_data = np.full_like(shifted_data, np.nan)
        valid_mask = ~np.isnan(shifted_data)
        final_plot_data[valid_mask] = shifted_data[valid_mask] + v_offset
        
        self.shift_line.set_ydata(final_plot_data)
        self.canvas.draw_idle()

    # NEW METHOD: Toggles visibility of reference signals
    def _toggle_reference_visibility(self):
        """Updates the visibility of reference signals based on checkbutton states."""
        visibility_changed = False
        for i, line in enumerate(self.ref_lines):
            # Ensure that there's a corresponding visibility variable
            if i < len(self.ref_visibility_vars):
                is_visible = self.ref_visibility_vars[i].get()
                if line.get_visible() != is_visible:
                    line.set_visible(is_visible)
                    visibility_changed = True
        
        if visibility_changed:
            # Redraw legend to reflect changes in visibility
            # Re-create legend to ensure it only shows visible lines with labels
            self.ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1), fancybox=True, shadow=True, ncol=max(1, len(self.ref_signal_labels) // 2 +1))
            self.canvas.draw_idle()

    def _save_and_close(self):
        self.lag_samples = self.lag_var.get()
        self.lag_seconds = self.lag_samples / self.fs if self.fs != 0 else float('inf')
        print(f"Lag saved: {self.lag_samples} samples ({self.lag_seconds:.4f} seconds)")
        self._close_window()

    def _close_window(self):
        if isinstance(self.root, tk.Toplevel) and self.root.winfo_exists():
            try:
                self.root.grab_release()
            except tk.TclError:
                pass # Ignore error if grab wasn't set or window already gone
        if self.root and self.root.winfo_exists():
            self.root.destroy()
        plt.close(self.fig) # Release matplotlib resources

    def get_lag(self):
        return self.lag_samples, self.lag_seconds

# Example Usage (for testing the GUI directly)
if __name__ == '__main__':
    # Create a dummy root window if this GUI is the main application.
    # This is important for Toplevel windows if no other Tkinter main loop is running.
    root_main_app = tk.Tk()
    root_main_app.withdraw() # Hide the dummy root window

    # Sample Data
    fs_main = 100 # Hz
    t_main = np.linspace(0, 5, 5 * fs_main, endpoint=False)
    ref1_main = np.sin(2 * np.pi * 1 * t_main) + 0.2 * np.random.randn(len(t_main))
    ref2_main = np.cos(2 * np.pi * 1.5 * t_main + np.pi/4) + 0.15 * np.random.randn(len(t_main))
    ref3_main_long_label = np.sin(2 * np.pi * 0.5 * t_main - np.pi/3) + 0.25 * np.random.randn(len(t_main))
    
    t_shift_main = np.linspace(0, 4.8, int(4.8 * fs_main), endpoint=False) # Slightly different length
    shift_main = 0.8 * np.sin(2 * np.pi * 1 * t_shift_main + np.pi/2) + 0.1 * np.random.randn(len(t_shift_main))

    reference_signals_main = [ref1_main, ref2_main, ref3_main_long_label]
    reference_labels_main = ["Reference Alpha (sin)", "Reference Beta (cos)", "A Very Very Long Reference Signal Gamma Name That Needs Truncation For Sure"]
    
    gui_instance = TimeSeriesLagGUI(
        ref_signals=reference_signals_main,
        shift_signal=shift_main,
        ref_signal_labels=reference_labels_main,
        shift_signal_label="My Data",
        fs=fs_main,
        lag_samples=-30, # Initial lag
        title="Signal Synchronization Demo",
        offset_references=True # Stagger reference signals vertically
    )

    saved_lag_samples_main, saved_lag_seconds_main = gui_instance.get_lag()
    if saved_lag_samples_main is not None:
        print(f"GUI closed. Final Lag: {saved_lag_samples_main} samples ({saved_lag_seconds_main:.4f} seconds).")
    else:
        print("GUI closed without saving the lag.")

    # Cleanly destroy the dummy root window
    if root_main_app.winfo_exists():
        root_main_app.destroy()