import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.lines as mlines
import time

# Helper function remains the same
def remove_nans_and_sync(signal, associated_array):
    if signal is None or associated_array is None: return np.array([]), np.array([])
    signal = np.asarray(signal)
    valid = ~np.isnan(signal)
    return signal[valid], associated_array[valid]

class AdjustChunksViewer:
    MIN_CHUNK_WIDTH = 3

    def __init__(self, signals, initial_split_indices, labels_list=None, title=None):
        # --- Input Validation (unchanged) ---
        if not signals: raise ValueError("Signals list cannot be empty.")
        if not isinstance(initial_split_indices, list): raise TypeError("initial_split_indices must be a list of tuples.")
        if any(not isinstance(item, tuple) or len(item) != 2 or not all(isinstance(n, int) for n in item) or item[0] >= item[1] for item in initial_split_indices): raise ValueError("Each item in initial_split_indices must be a tuple (start_int, end_int) with start < end.")
        signal_length = len(signals[0]) if signals[0] is not None else 0
        if any(item[0] < 0 or item[1] > signal_length for item in initial_split_indices): raise ValueError("Chunk indices must be within the signal bounds (0 to length).")
        # --- End Validation ---

        self.signals = signals
        self.chunks = sorted([c for c in initial_split_indices if c is not None], key=lambda x: x[0])

        self.fig, self.axes = plt.subplots(len(signals), 1, sharex=True, constrained_layout=True)
        if title is not None: self.fig.suptitle(title.replace('\t', '    '))
        if len(signals) == 1: self.axes = [self.axes]
        self.ax_main = self.axes[0]

        # --- State Variables ---
        num_chunks = len(self.chunks)
        self.selected_chunks_indices = set(range(num_chunks)) # Preselect all

        # Stores span artists ON AX_MAIN mapped to chunk index (for click detection)
        self.chunk_spans_dict = {}
        # Stores lists of Line2D artists (one per axis) mapped to the junction's x-value
        self.junction_lines_by_val = {}

        # Blitting related state
        self.backgrounds = None # To store saved backgrounds of axes

        self.dragging_info = {
            'original_x': None,           # Value of the junction being dragged
            'affected_chunks_indices': None,
            'active_lines': None,         # List of Line2D artists being dragged (across all axes)
        }
        # Double click timing state
        self._last_click_time = 0
        self._click_pos = None

        # --- Initial Plotting ---
        self._plot_signals(labels_list)
        self._plot_chunks_and_junctions() # Initial draw
        self._connect_events()
        self._add_ok_button()
        self._connect_keyboard_shortcut()

        # --- Maximize Window Logic ---
        # Ensure the plot is shown or drawn before accessing the manager
        plt.show(block=False)
        plt.pause(0.1) # Small pause to allow window creation
        try:
            manager = plt.get_current_fig_manager()
            if manager is not None:
                backend = plt.get_backend()
                # print(f"Using Matplotlib backend: {backend}") # Helpful for debugging

                if hasattr(manager, 'window'):
                    window = manager.window
                    if backend.startswith('Qt'): # Covers QtAgg, PyQt5, PySide2 etc.
                        if hasattr(window, 'showMaximized'):
                            # print("Attempting Qt maximization...")
                            window.showMaximized()
                        # else: print("Qt backend detected, but 'showMaximized' not found.")
                    elif backend == 'TkAgg':
                         # Tkinter uses 'zoomed' state for maximized
                         if hasattr(window, 'state') and callable(window.state):
                             # print("Attempting TkAgg maximization...")
                             window.state('zoomed')
                         # else: print("TkAgg backend detected, but 'state' method not found or not callable.")
                    elif backend == 'WXAgg':
                         if hasattr(window, 'Maximize'):
                             # print("Attempting WXAgg maximization...")
                             window.Maximize(True)
                         # else: print("WXAgg backend detected, but 'Maximize' not found.")
                    elif backend.startswith('GTK'): # Covers GTK3Agg, GTK4Agg
                         if hasattr(window, 'maximize'):
                             # print("Attempting GTK maximization...")
                             window.maximize()
                         # else: print("GTK backend detected, but 'maximize' not found.")
                    elif backend == 'MacOSX':
                        # print("MacOSX backend detected. Programmatic maximization might not be fully supported or reliable. Use window controls.")
                        pass # Cannot reliably maximize programmatically
                    # else: print(f"Maximization not explicitly handled for backend: {backend}. Window may not maximize.")
                # else: print("Figure manager does not have a 'window' attribute. Cannot maximize.")
            # else: print("Warning: Could not get figure manager. Window maximization might not work.")

        except Exception as e:
            # Catch potential errors during manager/window access or method calls
            print(f"An error occurred during window maximization attempt: {e}")
        # --- End Maximize Window Logic ---

        plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # Adjust layout for button and potential title
        plt.show(block=True) # Display the plot and wait


    def _plot_signals(self, labels_list):
        # --- Plotting signals (unchanged from previous version) ---
        if labels_list is None: labels_list = []
        use_custom_label = len(labels_list) == len(self.signals)
        max_len = 0
        for i, signal in enumerate(self.signals):
            ax = self.axes[i]
            if signal is not None:
                clean_signal, clean_time = remove_nans_and_sync(signal, np.arange(len(signal)))
                if len(clean_time) > 0:
                    label = labels_list[i] if use_custom_label and i < len(labels_list) else f'Signal {i}'
                    ax.plot(clean_time, clean_signal, label=label, color="k")
                    max_len = max(max_len, len(signal))
                else: ax.text(0.5, 0.5, f'Signal {i} empty/NaNs', **{'ha':'center', 'va':'center', 'transform':ax.transAxes})
                ax.legend()
                ax.grid(True, axis='y', linestyle=':')
            else: ax.text(0.5, 0.5, f'Signal {i} is None', **{'ha':'center', 'va':'center', 'transform':ax.transAxes})
        if self.chunks:
             min_s = min(c[0] for c in self.chunks); max_e = max(c[1] for c in self.chunks)
             pad = 0.05 * (max_e - min_s) if max_e > min_s else 10
             self.ax_main.set_xlim(min_s - pad, max_e + pad)
        elif max_len > 0: self.ax_main.set_xlim(-0.05 * max_len, max_len * 1.05)
        # --- End Plotting signals ---


    def _plot_chunks_and_junctions(self):
        """
        Clears and redraws chunk spans and junction lines.
        Uses self.junction_lines_by_val (maps value to list of lines).
        Sets lines animated=False initially.
        """
        # 1. Clear old artists referenced in dictionaries
        # Clear Spans (only stored for ax_main)
        for span_artist in list(self.chunk_spans_dict.keys()):
            try: span_artist.remove()
            except (ValueError, AttributeError): pass
        self.chunk_spans_dict.clear()

        # Clear Junction Lines (stored by value, lists across axes)
        for junction_val, line_list in list(self.junction_lines_by_val.items()):
            for line in line_list:
                 try: line.remove()
                 except (ValueError, AttributeError): pass
        self.junction_lines_by_val.clear()

        # --- Minimal clearing on other axes (optional, might not be needed if redraw is robust) ---
        # for ax in self.axes:
        #     if ax != self.ax_main:
        #          for patch in list(ax.patches): # Less targeted clearing
        #               try: patch.remove()
        #               except ValueError: pass
        #          # Lines are handled by the dictionary clearing above

        if not self.chunks:
            self.fig.canvas.draw_idle()
            return

        # 2. Draw new Spans
        for idx, (start, end) in enumerate(self.chunks):
            if start is None or end is None or start >= end: continue
            color = 'red' if idx % 2 == 0 else 'green'
            alpha = 0.8 if idx in self.selected_chunks_indices else 0.2
            # Draw on ALL axes, store only ax_main reference
            span_main = None
            for ax in self.axes:
                is_main = (ax == self.ax_main)
                if is_main:
                    # picker=True only for main axis to detect clicks
                    span = ax.axvspan(start, end, color=color, alpha=alpha, zorder=1, picker=is_main)
                else:
                    span = ax.axvspan(start, end, color=color, alpha=alpha, zorder=1)
                        
                self.chunk_spans_dict[span] = idx

        # 3. Draw new Junction Lines
        added_junction_values = set()
        for i in range(len(self.chunks) - 1):
             junction_val = self.chunks[i][1]
             # Ensure it's a valid internal junction
             if self.chunks[i+1][0] == junction_val + 1 and junction_val not in added_junction_values:
                 lines_for_this_junction = []
                 # Draw on ALL axes, store all references
                 for ax in self.axes:
                     is_main = (ax == self.ax_main)
                     # picker=5 only for main axis to detect drag start
                     if is_main:
                        line = ax.axvline(junction_val, color='blue', linestyle='--',
                                        linewidth=2, zorder=10,
                                        animated=False, picker=5)
                     else:
                        line = ax.axvline(junction_val, color='blue', linestyle='--',
                                        linewidth=2, zorder=10,
                                        animated=False)
                     lines_for_this_junction.append(line)

                 self.junction_lines_by_val[junction_val] = lines_for_this_junction
                 added_junction_values.add(junction_val)

        # Use draw_idle for standard redraws
        self.fig.canvas.draw_idle()


    # --- Event Handling ---
    def _connect_events(self):
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _disconnect_events(self):
        if hasattr(self, 'cid_press') and self.cid_press: self.fig.canvas.mpl_disconnect(self.cid_press)
        if hasattr(self, 'cid_release') and self.cid_release: self.fig.canvas.mpl_disconnect(self.cid_release)
        if hasattr(self, 'cid_motion') and self.cid_motion: self.fig.canvas.mpl_disconnect(self.cid_motion)
        self.cid_press = self.cid_release = self.cid_motion = None

    def _add_ok_button(self):
        ok_ax = self.fig.add_axes([0.85, 0.01, 0.1, 0.04])
        self.ok_button = Button(ok_ax, 'OK')
        self.ok_button.on_clicked(self._on_ok)

    def _connect_keyboard_shortcut(self):
        self.key_press_cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        # Add your code to handle the spacebar press here
        print(f"{event.key} pressed!")
        if event.key == 'enter': self._on_ok(event)
        if event.key == 'space' or event.key == ' ': 
            # If not starting a drag, check click location relative to chunks
            clicked_chunk_idx = self._find_clicked_chunk_index(event)
            # --- Action 2: Create Junction (Double-click inside chunk) ---
            self._create_junction(clicked_chunk_idx, event.xdata)

    def _on_ok(self, event):
        self._disconnect_events()
        if hasattr(self, 'key_press_cid') and self.key_press_cid: self.fig.canvas.mpl_disconnect(self.key_press_cid); self.key_press_cid = None
        # Ensure blitting state is reset if closed during drag
        if self.dragging_info['active_lines']:
            for line in self.dragging_info['active_lines']: line.set_animated(False)
        self.backgrounds = None
        plt.close(self.fig)

    def _find_clicked_chunk_index(self, event):
        # --- Finds chunk index based on click on AX_MAIN (unchanged) ---
        if hasattr(event, 'artist') and event.artist in self.chunk_spans_dict: return self.chunk_spans_dict[event.artist]
        if event.xdata is None: return None
        for span_artist, index in self.chunk_spans_dict.items():
            try:
                start, end = span_artist.get_xy()[0, 0], span_artist.get_xy()[2, 0]
                if start <= event.xdata < end: return index
            except (AttributeError, IndexError): continue
        return None


    def _find_clicked_junction_value(self, event):
        """
        Finds the junction x-value clicked or nearby on AX_MAIN.
        Returns the value (int) if found, otherwise None.
        Checks direct artist click first, then proximity.
        """
        clicked_val = None
        # Check direct artist click (only possible on ax_main line due to picker)
        if hasattr(event, 'artist') and isinstance(event.artist, mlines.Line2D):
            for val, line_list in self.junction_lines_by_val.items():
                if event.artist == line_list[self.axes.index(self.ax_main)]: # Check if it's the ax_main line for any value
                    clicked_val = val
                    break

        # If not direct click, check proximity on ax_main
        if clicked_val is None:
            if event.xdata is None: return None
            min_dist = float('inf')
            pixel_tolerance = 5
            try:
                display_coords = self.ax_main.transData.transform([(event.xdata, event.ydata)])
                display_coords_plus_tol = display_coords + [[pixel_tolerance, 0]]
                data_coords_plus_tol = self.ax_main.transData.inverted().transform(display_coords_plus_tol)
                data_tolerance = abs(data_coords_plus_tol[0, 0] - event.xdata)
                if data_tolerance <= 0: data_tolerance = (self.ax_main.get_xlim()[1] - self.ax_main.get_xlim()[0]) * 0.005 # Safety fallback
            except Exception: data_tolerance = (self.ax_main.get_xlim()[1] - self.ax_main.get_xlim()[0]) * 0.005 # Fallback

            for val in self.junction_lines_by_val.keys():
                 dist = abs(event.xdata - val)
                 if dist < data_tolerance and dist < min_dist:
                     min_dist = dist
                     clicked_val = val
        return clicked_val


    def _on_press(self, event):
        if event.inaxes != self.ax_main: return # Interactions only on main axis

        # Double-click detection (unchanged)
        is_double_click = False
        current_time = time.time()
        if event.button == 1:
            if current_time - self._last_click_time < 0.3:
                 if self._click_pos and abs(event.xdata - self._click_pos[0]) < 5 and abs(event.ydata - self._click_pos[1]) < 5:
                      is_double_click = True
            self._last_click_time = current_time
            self._click_pos = (event.xdata, event.ydata) if event.xdata is not None else None

        clicked_junction_val = self._find_clicked_junction_value(event)

        # --- Right Click (Button 3): Delete Junction ---
        if event.button == 3 and clicked_junction_val is not None:
            self._delete_junction(clicked_junction_val)
            return

        # --- Left Click (Button 1) ---
        elif event.button == 1:
            # --- Action 1: Start Drag (Single-click on junction) ---
            if clicked_junction_val is not None and not is_double_click:
                original_x = clicked_junction_val
                idx1, idx2 = self._find_affected_chunks(original_x)
                if idx1 is not None and idx2 is not None: # Valid junction to drag
                    active_lines = self.junction_lines_by_val.get(original_x)
                    if not active_lines: return # Should not happen if value found

                    self.dragging_info['original_x'] = original_x
                    self.dragging_info['affected_chunks_indices'] = (idx1, idx2)
                    self.dragging_info['active_lines'] = active_lines

                    # --- Start Blitting Sequence ---
                    # 1. Set lines to animated and change style for visual feedback
                    for line in active_lines:
                        line.set_animated(True)
                        line.set_linestyle(':')
                        line.set_color('magenta')
                    # 2. Draw the canvas *once* with the updated styles
                    self.fig.canvas.draw_idle()
                    # 3. Copy the background of each axis
                    self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
                    # --- End Blitting Sequence Start ---
                    return # Handled drag initiation

            # If not starting a drag, check click location relative to chunks
            clicked_chunk_idx = self._find_clicked_chunk_index(event)

            # --- Action 2: Create Junction (Double-click inside chunk) ---
            if clicked_chunk_idx is not None and is_double_click:
                 self._create_junction(clicked_chunk_idx, event.xdata)
                 return

            # --- Action 3: Select/Deselect Chunk (Single-click inside chunk) ---
            elif clicked_chunk_idx is not None and not is_double_click:
                 self._toggle_chunk_selection(clicked_chunk_idx)
                 return


    def _on_motion(self, event):
        """Handles junction dragging using blitting."""
        # Only act if dragging, in the main axes, and xdata is valid AND blitting is active
        if (event.inaxes != self.ax_main or self.backgrounds is None or
            self.dragging_info['active_lines'] is None or event.xdata is None):
            return

        original_x = self.dragging_info['original_x']
        idx1, idx2 = self.dragging_info['affected_chunks_indices']
        active_lines = self.dragging_info['active_lines']
        new_x_float = event.xdata

        # --- Constraint Calculation (same as before) ---
        chunk1_start = self.chunks[idx1][0]; chunk2_end = self.chunks[idx2][1]
        prev_boundary = -np.inf
        if idx1 > 0: prev_boundary = self.chunks[idx1 - 1][1]
        next_boundary = np.inf
        if idx2 < len(self.chunks) - 1: next_boundary = self.chunks[idx2 + 1][0]
        min_allowed_x = max(chunk1_start + self.MIN_CHUNK_WIDTH, prev_boundary + 1)
        max_allowed_x = min(chunk2_end - self.MIN_CHUNK_WIDTH, next_boundary -1)
        constrained_x = np.clip(new_x_float, min_allowed_x, max_allowed_x)
        new_x_int = int(round(constrained_x))
        if new_x_int <= chunk1_start or new_x_int >= chunk2_end: return # Safeguard
        # --- End Constraint Calculation ---

        # --- Blitting Update ---
        # 1. Restore the saved backgrounds
        for bg, ax in zip(self.backgrounds, self.axes):
            self.fig.canvas.restore_region(bg)
        # 2. Update the position of *all* animated lines
        for line in active_lines:
            line.set_xdata([new_x_int, new_x_int])
        # 3. Draw *only* the updated lines onto their respective axes
        for ax, line in zip(self.axes, active_lines):
            ax.draw_artist(line)
        # 4. Blit the updated axes regions to the screen
        for ax in self.axes:
            self.fig.canvas.blit(ax.bbox)
        # --- End Blitting Update ---


    def _on_release(self, event):
        """Finalizes junction drag, disables blitting, updates state, redraws."""
        # Only act if we were dragging (blitting active) and it's a left button release
        if self.backgrounds is None or self.dragging_info['active_lines'] is None or event.button != 1:
            return

        original_x = self.dragging_info['original_x']
        idx1, idx2 = self.dragging_info['affected_chunks_indices']
        active_lines = self.dragging_info['active_lines']

        # Get the final position from one of the lines (they should all be the same)
        final_x_float = float(active_lines[0].get_xdata()[0])

        # --- Stop Blitting ---
        # 1. Set lines back to non-animated
        for line in active_lines:
            line.set_animated(False)
        # 2. Clear the saved backgrounds
        self.backgrounds = None
        # --- End Stop Blitting ---

        # --- Re-apply constraints (same as _on_motion, for final position) ---
        chunk1_start = self.chunks[idx1][0]; chunk2_end = self.chunks[idx2][1]
        prev_boundary = -np.inf
        if idx1 > 0: prev_boundary = self.chunks[idx1 - 1][1]
        next_boundary = np.inf
        if idx2 < len(self.chunks) - 1: next_boundary = self.chunks[idx2 + 1][0]
        min_allowed_x = max(chunk1_start + self.MIN_CHUNK_WIDTH, prev_boundary + 1)
        max_allowed_x = min(chunk2_end - self.MIN_CHUNK_WIDTH, next_boundary -1)
        constrained_x = np.clip(final_x_float, min_allowed_x, max_allowed_x)
        final_x_int = int(round(constrained_x))
        # --- End Constraint Re-application ---

        # --- Check if update is valid and needed ---
        needs_update = False
        if (final_x_int > chunk1_start and final_x_int < chunk2_end and
            final_x_int - chunk1_start >= self.MIN_CHUNK_WIDTH and
            chunk2_end - final_x_int >= self.MIN_CHUNK_WIDTH):
             if final_x_int != original_x:
                 needs_update = True
        # --- End Validity Check ---

        # Reset dragging state *before* potential redraw
        current_dragging_info = self.dragging_info # Keep local copy for potential update
        self.dragging_info = { 'original_x': None, 'affected_chunks_indices': None, 'active_lines': None }


        # --- Apply the change if valid and different ---
        if needs_update:
            # Update chunk definitions
            start1, _ = self.chunks[idx1]
            self.chunks[idx1] = (start1, final_x_int)
            _, end2 = self.chunks[idx2]
            self.chunks[idx2] = (final_x_int + 1, end2) # Ensure +1 gap

            # Update junction dictionary: Remove old value entry.
            # The new value/lines will be created by the full redraw.
            if original_x in self.junction_lines_by_val:
                del self.junction_lines_by_val[original_x]

            # Redraw everything to reflect the change and create new, non-animated artists
            self._plot_chunks_and_junctions()
        else:
            # No change needed or reverted, just redraw to reset visual style/position
            self._plot_chunks_and_junctions()


    def _create_junction(self, chunk_index, click_x):
        # --- Logic for creating junction (unchanged from previous version) ---
        if click_x is None: return
        if chunk_index < 0 or chunk_index >= len(self.chunks): return
        start, end = self.chunks[chunk_index]
        new_junction_x = int(round(click_x))
        if (new_junction_x <= start or new_junction_x >= end or
            new_junction_x - start < self.MIN_CHUNK_WIDTH or
            end - new_junction_x < self.MIN_CHUNK_WIDTH):
            print(f"Cannot create junction at {new_junction_x}: Too close to boundary {start}-{end} or violates minimum chunk width ({self.MIN_CHUNK_WIDTH}).")
            return
        new_chunk1 = (start, new_junction_x)
        new_chunk2 = (new_junction_x + 1, end)
        self.chunks = self.chunks[:chunk_index] + [new_chunk1, new_chunk2] + self.chunks[chunk_index+1:]
        # Update selection state
        was_selected = chunk_index in self.selected_chunks_indices
        shifted_selection = set()
        for selected_idx in self.selected_chunks_indices:
            if selected_idx < chunk_index: shifted_selection.add(selected_idx)
            elif selected_idx > chunk_index: shifted_selection.add(selected_idx + 1)
        self.selected_chunks_indices = shifted_selection
        # Redraw everything
        self._plot_chunks_and_junctions()


    def _toggle_chunk_selection(self, chunk_index):
        """Toggles selection and redraws everything (simplest robust approach)."""
        # --- Logic for toggling selection (unchanged, still calls full redraw) ---
        if chunk_index < 0 or chunk_index >= len(self.chunks): return
        # Find the main span artist to confirm click validity (optional but good)
        span_to_update = None
        for span, index in self.chunk_spans_dict.items():
             if index == chunk_index: span_to_update = span; break
        if span_to_update is None:
            self._plot_chunks_and_junctions() # Redraw if state seems inconsistent
            return
        # Toggle selection status
        if chunk_index in self.selected_chunks_indices:
            self.selected_chunks_indices.remove(chunk_index)
        else:
            self.selected_chunks_indices.add(chunk_index)
        # Redraw all chunks/junctions to reflect updated alpha
        # Note: Optimizing this to only update alpha is possible but adds complexity
        # in finding/updating spans on all axes. Full redraw is simpler for now.
        self._plot_chunks_and_junctions()


    def _delete_junction(self, junction_x):
        """Deletes the junction at junction_x and merges adjacent chunks."""
        # --- Logic for deleting junction (modified to use junction_x value) ---
        idx1, idx2 = self._find_affected_chunks(junction_x)
        if idx1 is None or idx2 is None:
            print(f"Error: Could not find valid adjacent chunks for junction at {junction_x}. Deletion aborted.")
            # Clean up inconsistent line reference if it exists
            if junction_x in self.junction_lines_by_val: del self.junction_lines_by_val[junction_x]
            self._plot_chunks_and_junctions() # Redraw to remove visual artifact
            return
        start1, end1 = self.chunks[idx1]; start2, end2 = self.chunks[idx2]
        # Sanity check
        if end1 != junction_x or start2 != (junction_x + 1):
             print(f"Error: Chunk data inconsistent at junction {junction_x}. Chunks: {self.chunks[idx1]}, {self.chunks[idx2]}. Deletion aborted.")
             if junction_x in self.junction_lines_by_val: del self.junction_lines_by_val[junction_x]
             self._plot_chunks_and_junctions()
             return
        # Determine selection status before modifying list
        chunk1_sel = idx1 in self.selected_chunks_indices
        chunk2_sel = idx2 in self.selected_chunks_indices
        new_chunk_should_be_selected = chunk1_sel or chunk2_sel
        # Create merged chunk
        merged_chunk = (start1, end2)
        # Update chunks list
        self.chunks = self.chunks[:idx1] + [merged_chunk] + self.chunks[idx2+1:]
        # Update selection state
        new_selection = set()
        for selected_idx in self.selected_chunks_indices:
            if selected_idx < idx1: new_selection.add(selected_idx)
            elif selected_idx > idx2: new_selection.add(selected_idx - 1)
        if new_chunk_should_be_selected: new_selection.add(idx1)
        self.selected_chunks_indices = new_selection

        # Clean up artist dictionary entry
        if junction_x in self.junction_lines_by_val:
            junction_val, line_list = self.junction_lines_by_val[junction_x]
            for line in line_list:
                 try: line.remove()
                 except (ValueError, AttributeError): pass
            del self.junction_lines_by_val[junction_x]
        # Redraw everything
        self._plot_chunks_and_junctions()


    # --- Helper and Getter Methods ---
    def _find_affected_chunks(self, junction_x):
        # --- Finds adjacent chunk indices (unchanged) ---
        idx_ending, idx_starting = None, None
        for i, (start, end) in enumerate(self.chunks):
            if end == junction_x: idx_ending = i
            if start == (junction_x + 1): idx_starting = i
            if idx_ending == i and i + 1 < len(self.chunks): # Optimization
                if self.chunks[i+1][0] == (junction_x + 1): idx_starting = i + 1; break
        if idx_ending is not None and idx_starting == idx_ending + 1: return idx_ending, idx_starting
        else: return None, None

    def get_final_chunks(self):
        # --- Returns all chunks (unchanged) ---
        return list(self.chunks)

    def get_selected_chunks(self):
        # --- Returns selected chunks (unchanged) ---
        selected_data = [self.chunks[i] for i in sorted(list(self.selected_chunks_indices))]
        return selected_data
