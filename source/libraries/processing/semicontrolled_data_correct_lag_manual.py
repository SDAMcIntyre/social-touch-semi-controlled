import tkinter as tk
from tkinter import Scale, Button, Frame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

class TimeSeriesLagGUI:
    """
    A GUI for visualizing and adjusting the lag and offset between a shiftable time series
    and one or more reference time series.
    Includes zoom/pan toolbar. Plot updates when slider handle is released.
    Buttons are placed on the bottom right.

    Attributes:
        lag_samples (int | None): The lag in samples selected by the user when
                                  the 'Save Lag and Close' button is pressed.
                                  None if the window was closed otherwise.
        lag_seconds (float | None): The lag in seconds calculated from lag_samples
                                    and the sampling frequency (fs).
                                    None if the window was closed otherwise.
    """

    def __init__(self, ref_signals, shift_signal,
                 ref_signal_labels=None, shift_signal_label="", lag_samples=None, fs=1000, master=None):
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
            # If no master provided, create a new Toplevel window.
            # A hidden root window might be needed in the main script.
            self.root = tk.Toplevel()
            self.root.title("Time Series Lag Adjustment")
            is_toplevel = True
        else:
            # If a master is provided, create the GUI within that master.
            self.root = Frame(master) # Use a Frame if embedding
            self.root.pack(fill=tk.BOTH, expand=True) # Assume frame should fill parent
            is_toplevel = False

        # --- Input Handling ---
        # Ensure ref_signals is a list
        if not isinstance(ref_signals, (list, tuple)):
             self.ref_signals_raw = [np.asarray(ref_signals)]
        else:
             self.ref_signals_raw = [np.asarray(rs) for rs in ref_signals]

        self.shift_signal_raw = np.asarray(shift_signal)
        self.fs = fs
        self.initial_lag_samples = lag_samples if lag_samples is not None else 0
        self.lag_samples = None # Variable to store the final lag in samples
        self.lag_seconds = None # Variable to store the final lag in seconds
        self.max_v_offset = 0.5 # Define max vertical offset for limit calculation

        # Handle Labels
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
        min_len = min(len(s) for s in all_signals_norm if len(s) > 0) if all_signals_norm else 0
        if min_len == 0:
            print("Warning: One or more input signals have zero length.")
            # Handle gracefully, perhaps by setting a default length or showing an error
            min_len = 1 # Avoid division by zero for time_axis if possible

        self.time_axis = np.arange(min_len) / self.fs

        # Prepare data for plotting (trimmed to min_len)
        self.ref_signals_plot = [rs_norm[:min_len].copy() for rs_norm in self.ref_signals_norm]
        self.shift_signal_plot_base = self.shift_signal_norm[:min_len].copy() # Base for shifting

        # --- Setup Matplotlib Figure ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6)) # Slightly wider for potentially more legend items
        self.ref_lines = []
        for i, ref_ts in enumerate(self.ref_signals_plot):
            line, = self.ax.plot(self.time_axis, ref_ts, label=self.ref_signal_labels[i])
            self.ref_lines.append(line)

        # Plot the shiftable signal (will be updated by sliders)
        # Initialize with NaNs or the base data, update happens immediately after setup
        initial_shifted_data = np.full_like(self.shift_signal_plot_base, np.nan)
        self.shift_line, = self.ax.plot(self.time_axis, initial_shifted_data, label=f"{self.shift_signal_label} (Shifted)")

        self.ax.legend()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Normalized Amplitude")
        self.ax.set_title("Adjust Lag and Offset")
        self.ax.grid(True)

        # --- Set Initial Axis Limits ---
        if len(self.time_axis) > 1:
            self.ax.set_xlim(self.time_axis[0], self.time_axis[-1])
        else:
             self.ax.set_xlim(0, 1/self.fs if self.fs > 0 else 1) # Handle single point case

        # Set Y limits based on normalized range [0, 1] plus offset allowance
        y_padding = 0.1 * self.max_v_offset
        self.ax.set_ylim(0 - self.max_v_offset - y_padding, 1 + self.max_v_offset + y_padding)


        # --- Embed Plot in Tkinter ---
        main_frame = Frame(self.root) # All widgets go into this frame
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Vertical offset slider on the left
        plot_area_frame = Frame(main_frame)
        plot_area_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.v_offset_var = tk.DoubleVar(value=0.0)
        self.v_offset_slider = Scale(plot_area_frame, from_=self.max_v_offset, to=-self.max_v_offset,
                                     resolution=0.01, orient=tk.VERTICAL, label="V Offset",
                                     variable=self.v_offset_var)
        self.v_offset_slider.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=(35, 5)) # Add padding top to align roughly with plot
        self.v_offset_slider.bind("<ButtonRelease-1>", self._update_plot)

        # Canvas and Toolbar frame
        canvas_toolbar_frame = Frame(plot_area_frame)
        canvas_toolbar_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_toolbar_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X) # Toolbar above canvas

        self.canvas.draw()

        # Horizontal lag slider below the plot
        slider_frame = Frame(main_frame)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Adjust max lag based on data length? Or keep fixed? Fixed seems simpler.
        self.max_lag_range_samples = int(3 * self.fs) # Max lag adjustment range (+/-)
        # Clamp initial lag to slider range
        clamped_initial_lag = max(-self.max_lag_range_samples, min(self.max_lag_range_samples, self.initial_lag_samples))
        self.lag_var = tk.IntVar(value=clamped_initial_lag)

        self.h_lag_slider = Scale(slider_frame, from_=-self.max_lag_range_samples, to=self.max_lag_range_samples,
                                  orient=tk.HORIZONTAL, label="Horizontal Lag (samples)",
                                  variable=self.lag_var, length=600) # Adjust length as needed
        self.h_lag_slider.pack(fill=tk.X, expand=True)
        self.h_lag_slider.bind("<ButtonRelease-1>", self._update_plot)

        # Buttons at the bottom right
        button_frame = Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Add empty label to push buttons to the right
        # spacer_label = tk.Label(button_frame, text="")
        # spacer_label.pack(side=tk.LEFT, expand=True)

        self.save_button = Button(button_frame, text="Save Lag and Close", command=self._save_and_close)
        self.save_button.pack(side=tk.RIGHT, padx=(5, 5), pady=5) # Place save button first (rightmost)

        self.close_button = Button(button_frame, text="Close", command=self._close_window)
        self.close_button.pack(side=tk.RIGHT, padx=(0, 5), pady=5) # Place close button to the left of save


        # --- Initial Plot Update ---
        self._update_plot() # Draw the initial state with lag=initial_lag_samples


        # --- Make Window Blocking (Only if it's a Toplevel) ---
        if is_toplevel:
            # Center the Toplevel window manually
            self.root.update_idletasks() # Ensure window dimensions are calculated
            try: # Use try-except as winfo_width/height might fail if window closed prematurely
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
                 print("Warning: Could not get window dimensions for centering.")

            self.root.protocol("WM_DELETE_WINDOW", self._close_window) # Handle window close button [X]
            self.root.grab_set()  # Make window modal
            self.root.wait_window(self.root) # Wait until window is destroyed
        # If self.root is a Frame embedded in a master, blocking is handled by the caller


    def _normalize(self, data):
        """Normalizes data between 0 and 1, handling NaNs and constant values."""
        data = np.asarray(data, dtype=float) # Ensure it's a float numpy array
        data_finite = data[np.isfinite(data)]
        if len(data_finite) == 0:
            # If no finite values, return array of 0.5 (or NaNs)
            return np.full_like(data, 0.5, dtype=float) # Or np.nan? 0.5 is plottable.

        min_val = np.min(data_finite)
        max_val = np.max(data_finite)

        if max_val == min_val:
            # Handle constant data: return array of 0.5 where finite, NaN otherwise
            norm_data = np.full_like(data, np.nan, dtype=float)
            norm_data[np.isfinite(data)] = 0.5
            return norm_data
        else:
            # Perform normalization
            norm_data = np.full_like(data, np.nan, dtype=float)
            finite_mask = np.isfinite(data)
            norm_data[finite_mask] = (data[finite_mask] - min_val) / (max_val - min_val)
            return norm_data

    def _update_plot(self, event=None):
        """Callback function to update the plot data when sliders change (called on release)."""
        current_lag_samples = self.lag_var.get()
        v_offset = self.v_offset_var.get()

        # Prepare an array for the shifted data, initialized with NaNs
        shifted_data = np.full_like(self.shift_signal_plot_base, np.nan)
        source_data = self.shift_signal_plot_base # Use the base normalized data
        data_len = len(source_data)

        # Apply horizontal shift (lag)
        if data_len > 0: # Only shift if there's data
            if current_lag_samples >= 0:
                # Positive lag: shift data to the right (starts later)
                if current_lag_samples < data_len:
                    target_slice = slice(current_lag_samples, data_len)
                    source_slice = slice(0, data_len - current_lag_samples)
                    shifted_data[target_slice] = source_data[source_slice]
            else:
                # Negative lag: shift data to the left (starts earlier)
                abs_lag = abs(current_lag_samples)
                if abs_lag < data_len:
                    target_slice = slice(0, data_len - abs_lag)
                    source_slice = slice(abs_lag, data_len)
                    shifted_data[target_slice] = source_data[source_slice]

        # Apply vertical offset only to non-NaN values resulting from the shift
        final_plot_data = np.full_like(shifted_data, np.nan)
        valid_mask = ~np.isnan(shifted_data)
        final_plot_data[valid_mask] = shifted_data[valid_mask] + v_offset

        # Update the plot data for the shiftable line
        self.shift_line.set_ydata(final_plot_data)
        self.canvas.draw_idle() # Efficiently redraw

    def _save_and_close(self):
        """Saves the current lag in samples and seconds, then closes the window."""
        self.lag_samples = self.lag_var.get()
        self.lag_seconds = self.lag_samples / self.fs if self.fs != 0 else float('inf')
        print(f"Lag saved: {self.lag_samples} samples ({self.lag_seconds:.4f} seconds)")
        self._close_window()

    def _close_window(self):
        """Closes the Tkinter window and releases resources."""
        # Release grab if it's a modal Toplevel
        if isinstance(self.root, tk.Toplevel) and self.root.winfo_exists():
             try:
                 self.root.grab_release()
             except tk.TclError:
                 pass # Ignore error if grab wasn't set or window already gone

        # Destroy the main container (Toplevel or Frame)
        if self.root and self.root.winfo_exists():
             self.root.destroy()

        # Explicitly close the figure to release matplotlib resources
        # This prevents potential memory leaks if the GUI is created multiple times
        plt.close(self.fig)

    def get_lag(self):
        """
        Returns the saved lag values after the window is closed.

        Returns:
            tuple: A tuple containing (lag_samples, lag_seconds).
                   (None, None) if the window was closed without saving.
        """
        return self.lag_samples, self.lag_seconds

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure a root Tk instance exists for Toplevel windows, but keep it hidden.
    root_app = tk.Tk()
    root_app.withdraw()

    # Generate sample data
    fs = 1000  # Sampling frequency
    t = np.arange(0, 5, 1/fs)  # Time vector for 5 seconds

    # Reference Signals
    ref1 = np.sin(2 * np.pi * 5 * t) + np.random.randn(len(t)) * 0.1
    ref2 = np.cos(2 * np.pi * 5 * t + np.pi/4) + 0.5 + np.random.randn(len(t)) * 0.05 # Different phase and offset
    ref_signals = [ref1, ref2]
    ref_labels = ["Reference Sine", "Reference Cosine+Offset"]

    # Shiftable Signal (create a lag relative to the first reference)
    true_lag_s = 0.35
    true_lag_samples = int(true_lag_s * fs)
    shift_signal_base = np.sin(2 * np.pi * 5 * (t - true_lag_s)) + np.random.randn(len(t)) * 0.15
    # Apply the true lag correctly to simulate real data
    shift_signal = np.full_like(ref1, np.nan)
    if true_lag_samples >= 0:
        if true_lag_samples < len(t):
            shift_signal[true_lag_samples:] = shift_signal_base[:-true_lag_samples]
        else: # Lag is >= signal length
             pass # Signal remains all NaN
    else: # Negative lag
        abs_lag = abs(true_lag_samples)
        if abs_lag < len(t):
             shift_signal[:len(t)-abs_lag] = shift_signal_base[abs_lag:]
        else: # Negative lag is >= signal length
             pass # Signal remains all NaN

    shift_label = "Shiftable Signal (Sine)"

    print("Starting GUI... Script execution will pause here.")

    # Launch the GUI
    # Pass the list of reference signals and the shiftable signal
    gui = TimeSeriesLagGUI(ref_signals, shift_signal,
                           ref_signal_labels=ref_labels,
                           shift_signal_label=shift_label,
                           fs=fs,
                           lag_samples=50) # Optional: start with an initial lag guess

    # --- Execution resumes here after the GUI window is closed ---
    print("GUI closed.")

    # Retrieve the selected lag
    selected_lag_samples, selected_lag_seconds = gui.get_lag()

    if selected_lag_samples is not None:
        print(f"The selected lag from the GUI is: {selected_lag_samples} samples")
        print(f"The selected lag from the GUI is: {selected_lag_seconds:.4f} seconds")
        print(f"(True lag was: {true_lag_samples} samples = {true_lag_s:.4f} seconds)")
    else:
        print("GUI was closed without saving the lag.")

    # Clean up the hidden root window
    # Check if it still exists before destroying
    if root_app and root_app.winfo_exists():
        root_app.destroy()
    print("Script finished.")