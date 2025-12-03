import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.lines as mlines
import time

# Helper function
def remove_nans_and_sync(signal, associated_array):
    if signal is None or associated_array is None: return np.array([]), np.array([])
    signal = np.asarray(signal)
    valid = ~np.isnan(signal)
    return signal[valid], associated_array[valid]

class AdjustChunksViewer:
    MIN_CHUNK_WIDTH = 3

    def __init__(self, signals, initial_split_indices, labels_list=None, title=None, initial_selected_indices=None):
        """
        Args:
            signals: List of 1D arrays or a single 1D numpy array.
            initial_split_indices: List of tuples (start, end).
            labels_list: List of strings for legend.
            title: Window title.
            initial_selected_indices: List/Set of indices that should be selected.
        """
        # --- Input Validation and Normalization ---
        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            self.signals = [signals]
        elif isinstance(signals, list):
            self.signals = signals
        else:
             raise ValueError("Signals must be a list of arrays or a single numpy array.")

        if not self.signals: raise ValueError("Signals list cannot be empty.")
        
        if not isinstance(initial_split_indices, list): raise TypeError("initial_split_indices must be a list of tuples.")
        if any(not isinstance(item, tuple) or len(item) != 2 or not all(isinstance(n, (int, np.integer)) for n in item) or item[0] >= item[1] for item in initial_split_indices): 
            raise ValueError("Each item in initial_split_indices must be a tuple (start_int, end_int) with start < end.")
        
        signal_length = len(self.signals[0]) if self.signals[0] is not None else 0
        
        if any(item[0] < 0 or item[1] > signal_length for item in initial_split_indices): 
            raise ValueError("Chunk indices must be within the signal bounds (0 to length).")
        # --- End Validation ---

        self.chunks = sorted([c for c in initial_split_indices if c is not None], key=lambda x: x[0])

        self.fig, self.axes = plt.subplots(len(self.signals), 1, sharex=True, constrained_layout=True)
        if title is not None: self.fig.suptitle(title.replace('\t', '    '))
        
        # Ensure axes is always a list
        if len(self.signals) == 1: 
            self.axes = [self.axes]
        else:
            self.axes = list(self.axes)
            
        self.ax_main = self.axes[0]

        # --- State Variables ---
        num_chunks = len(self.chunks)
        
        # Handle initial selection state
        if initial_selected_indices is not None:
            self.selected_chunks_indices = set(initial_selected_indices)
            self.selected_chunks_indices = {i for i in self.selected_chunks_indices if 0 <= i < num_chunks}
        else:
            self.selected_chunks_indices = set(range(num_chunks)) 

        # Stores span artists ON AX_MAIN ONLY mapped to chunk index
        self.chunk_spans_dict = {}
        # Stores lists of Line2D artists (one per axis) mapped to the junction's x-value
        self.junction_lines_by_val = {}

        # Blitting related state
        self.backgrounds = None 

        self.dragging_info = {
            'original_x': None,
            'affected_chunks_indices': None,
            'active_lines': None,
        }
        # Double click timing state
        self._last_click_time = 0
        self._click_pos = None

        # --- Initial Plotting ---
        self._plot_signals(labels_list)
        self._plot_chunks_and_junctions() 
        self._connect_events()
        self._add_ok_button()
        self._connect_keyboard_shortcut()

        # --- Maximize Window Logic ---
        # Logic applied before plt.show(block=True) to ensure window opens maximized without flicker.
        try:
            manager = plt.get_current_fig_manager()
            backend = plt.get_backend().lower()
            if manager is not None:
                if 'qt' in backend:
                    if hasattr(manager, 'window') and hasattr(manager.window, 'showMaximized'):
                        manager.window.showMaximized()
                elif 'tk' in backend:
                    if hasattr(manager, 'window') and hasattr(manager.window, 'state'):
                         manager.window.state('zoomed')
                elif 'wx' in backend:
                     if hasattr(manager, 'window') and hasattr(manager.window, 'Maximize'):
                          manager.window.Maximize(True)
                elif 'gtk' in backend:
                     if hasattr(manager, 'window') and hasattr(manager.window, 'maximize'):
                          manager.window.maximize()
        except Exception as e:
            print(f"Window maximization failed: {e}")
        # --- End Maximize Window Logic ---

        plt.tight_layout(rect=[0, 0.05, 1, 0.97]) 
        plt.show(block=True) 


    def _plot_signals(self, labels_list):
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
                else: ax.text(0.5, 0.5, f'Signal {i} empty/NaNs', ha='center', va='center', transform=ax.transAxes)
                ax.legend()
                ax.grid(True, axis='y', linestyle=':')
            else: ax.text(0.5, 0.5, f'Signal {i} is None', ha='center', va='center', transform=ax.transAxes)
        if self.chunks:
             min_s = min(c[0] for c in self.chunks); max_e = max(c[1] for c in self.chunks)
             pad = 0.05 * (max_e - min_s) if max_e > min_s else 10
             self.ax_main.set_xlim(min_s - pad, max_e + pad)
        elif max_len > 0: self.ax_main.set_xlim(-0.05 * max_len, max_len * 1.05)


    def _plot_chunks_and_junctions(self):
        """
        Clears and redraws chunk spans and junction lines.
        Only stores ax_main spans in chunk_spans_dict for hit testing.
        """
        # 1. Clear old artists
        for span_artist in list(self.chunk_spans_dict.keys()):
            try: span_artist.remove()
            except (ValueError, AttributeError): pass
        self.chunk_spans_dict.clear()
        
        # Clear Junction Lines
        for junction_val, line_list in list(self.junction_lines_by_val.items()):
            for line in line_list:
                 try: line.remove()
                 except (ValueError, AttributeError): pass
        self.junction_lines_by_val.clear()
        
        if hasattr(self, 'all_spans_list'):
            for s in self.all_spans_list:
                try: s.remove()
                except: pass
        self.all_spans_list = []

        if not self.chunks:
            self.fig.canvas.draw_idle()
            return

        # 2. Draw new Spans
        for idx, (start, end) in enumerate(self.chunks):
            if start is None or end is None or start >= end: continue
            
            # Determine color and alpha based on activation status
            if idx in self.selected_chunks_indices:
                color = 'red' if idx % 2 == 0 else 'green'
                alpha = 0.8
            else:
                color = 'gray' # Distinct color for deactivated state
                alpha = 0.4    # Distinct alpha for deactivated state
            
            for ax in self.axes:
                is_main = (ax == self.ax_main)
                # Picker only on main
                span = ax.axvspan(start, end, color=color, alpha=alpha, zorder=1, picker=is_main)
                self.all_spans_list.append(span)
                
                # Only map the main span for click detection
                if is_main:
                    self.chunk_spans_dict[span] = idx

        # 3. Draw new Junction Lines
        added_junction_values = set()
        for i in range(len(self.chunks) - 1):
             junction_val = self.chunks[i][1]
             if self.chunks[i+1][0] == junction_val + 1 and junction_val not in added_junction_values:
                 lines_for_this_junction = []
                 for ax in self.axes:
                     is_main = (ax == self.ax_main)
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
        if event.key == 'enter': self._on_ok(event)
        if event.key == 'space' or event.key == ' ': 
            clicked_chunk_idx = self._find_clicked_chunk_index(event)
            self._create_junction(clicked_chunk_idx, event.xdata)

    def _on_ok(self, event):
        self._disconnect_events()
        if hasattr(self, 'key_press_cid') and self.key_press_cid: self.fig.canvas.mpl_disconnect(self.key_press_cid); self.key_press_cid = None
        if self.dragging_info['active_lines']:
            for line in self.dragging_info['active_lines']: line.set_animated(False)
        self.backgrounds = None
        plt.close(self.fig)

    def _find_clicked_chunk_index(self, event):
        """
        Robustly finds the chunk index based on event.xdata using geometry check.
        """
        if event.inaxes != self.ax_main or event.xdata is None:
            return None
            
        # Iterate over spans on the main axis
        for span_artist, index in self.chunk_spans_dict.items():
            start_x = span_artist.xy[0]
            end_x = start_x + span_artist.get_width()
            
            if start_x <= event.xdata < end_x:
                return index
        return None

    def _find_clicked_junction_value(self, event):
        clicked_val = None
        # Check direct artist click (picker)
        # Note: button_press_event doesn't natively carry 'artist' unless custom triggered,
        # but we keep this check if matplotlib version differs or for completeness.
        if hasattr(event, 'artist') and isinstance(event.artist, mlines.Line2D):
            for val, line_list in self.junction_lines_by_val.items():
                if event.artist == line_list[self.axes.index(self.ax_main)]:
                    clicked_val = val
                    break

        if clicked_val is None:
            if event.xdata is None: return None
            min_dist = float('inf')
            # Use a slightly wider data tolerance
            data_tolerance = (self.ax_main.get_xlim()[1] - self.ax_main.get_xlim()[0]) * 0.01 
            
            for val in self.junction_lines_by_val.keys():
                 dist = abs(event.xdata - val)
                 if dist < data_tolerance and dist < min_dist:
                     min_dist = dist
                     clicked_val = val
        return clicked_val


    def _on_press(self, event):
        if event.inaxes != self.ax_main: return 

        is_double_click = False
        current_time = time.time()
        if event.button == 1:
            if current_time - self._last_click_time < 0.3:
                 if self._click_pos and abs(event.xdata - self._click_pos[0]) < 5 and abs(event.ydata - self._click_pos[1]) < 5:
                       is_double_click = True
            self._last_click_time = current_time
            self._click_pos = (event.xdata, event.ydata) if event.xdata is not None else None

        clicked_junction_val = self._find_clicked_junction_value(event)

        # --- Right Click: Delete ---
        if event.button == 3 and clicked_junction_val is not None:
            self._delete_junction(clicked_junction_val)
            return

        # --- Left Click ---
        elif event.button == 1:
            # 1. Drag Junction (Single click on junction)
            if clicked_junction_val is not None and not is_double_click:
                original_x = clicked_junction_val
                idx1, idx2 = self._find_affected_chunks(original_x)
                if idx1 is not None and idx2 is not None:
                    active_lines = self.junction_lines_by_val.get(original_x)
                    if not active_lines: return

                    self.dragging_info['original_x'] = original_x
                    self.dragging_info['affected_chunks_indices'] = (idx1, idx2)
                    self.dragging_info['active_lines'] = active_lines

                    # Blitting Setup
                    for line in active_lines:
                        line.set_animated(True)
                        line.set_linestyle(':')
                        line.set_color('magenta')
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events() # Ensure draw processes
                    self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
                    return 

            clicked_chunk_idx = self._find_clicked_chunk_index(event)

            # 2. Create Junction (Double click on chunk)
            if clicked_chunk_idx is not None and is_double_click:
                 self._create_junction(clicked_chunk_idx, event.xdata)
                 return

            # 3. Toggle Selection (Single click on chunk)
            elif clicked_chunk_idx is not None and not is_double_click:
                 self._toggle_chunk_selection(clicked_chunk_idx)
                 return


    def _on_motion(self, event):
        if (event.inaxes != self.ax_main or self.backgrounds is None or
            self.dragging_info['active_lines'] is None or event.xdata is None):
            return

        idx1, idx2 = self.dragging_info['affected_chunks_indices']
        active_lines = self.dragging_info['active_lines']
        new_x_float = event.xdata

        chunk1_start = self.chunks[idx1][0]; chunk2_end = self.chunks[idx2][1]
        prev_boundary = -np.inf
        if idx1 > 0: prev_boundary = self.chunks[idx1 - 1][1]
        next_boundary = np.inf
        if idx2 < len(self.chunks) - 1: next_boundary = self.chunks[idx2 + 1][0]
        
        min_allowed_x = max(chunk1_start + self.MIN_CHUNK_WIDTH, prev_boundary + 1)
        max_allowed_x = min(chunk2_end - self.MIN_CHUNK_WIDTH, next_boundary -1)
        
        constrained_x = np.clip(new_x_float, min_allowed_x, max_allowed_x)
        new_x_int = int(round(constrained_x))
        if new_x_int <= chunk1_start or new_x_int >= chunk2_end: return 

        for bg, ax in zip(self.backgrounds, self.axes):
            self.fig.canvas.restore_region(bg)
        for line in active_lines:
            line.set_xdata([new_x_int, new_x_int])
        for ax, line in zip(self.axes, active_lines):
            ax.draw_artist(line)
        for ax in self.axes:
            self.fig.canvas.blit(ax.bbox)


    def _on_release(self, event):
        if self.backgrounds is None or self.dragging_info['active_lines'] is None or event.button != 1:
            return

        original_x = self.dragging_info['original_x']
        idx1, idx2 = self.dragging_info['affected_chunks_indices']
        active_lines = self.dragging_info['active_lines']
        final_x_float = float(active_lines[0].get_xdata()[0])

        for line in active_lines:
            line.set_animated(False)
        self.backgrounds = None

        chunk1_start = self.chunks[idx1][0]; chunk2_end = self.chunks[idx2][1]
        prev_boundary = -np.inf
        if idx1 > 0: prev_boundary = self.chunks[idx1 - 1][1]
        next_boundary = np.inf
        if idx2 < len(self.chunks) - 1: next_boundary = self.chunks[idx2 + 1][0]
        min_allowed_x = max(chunk1_start + self.MIN_CHUNK_WIDTH, prev_boundary + 1)
        max_allowed_x = min(chunk2_end - self.MIN_CHUNK_WIDTH, next_boundary -1)
        
        constrained_x = np.clip(final_x_float, min_allowed_x, max_allowed_x)
        final_x_int = int(round(constrained_x))

        needs_update = False
        if (final_x_int > chunk1_start and final_x_int < chunk2_end and
            final_x_int - chunk1_start >= self.MIN_CHUNK_WIDTH and
            chunk2_end - final_x_int >= self.MIN_CHUNK_WIDTH):
             if final_x_int != original_x:
                 needs_update = True

        self.dragging_info = { 'original_x': None, 'affected_chunks_indices': None, 'active_lines': None }

        if needs_update:
            start1, _ = self.chunks[idx1]
            self.chunks[idx1] = (start1, final_x_int)
            _, end2 = self.chunks[idx2]
            self.chunks[idx2] = (final_x_int + 1, end2) 
            if original_x in self.junction_lines_by_val:
                del self.junction_lines_by_val[original_x]
            self._plot_chunks_and_junctions()
        else:
            self._plot_chunks_and_junctions()


    def _create_junction(self, chunk_index, click_x):
        if click_x is None: return
        if chunk_index < 0 or chunk_index >= len(self.chunks): return
        start, end = self.chunks[chunk_index]
        new_junction_x = int(round(click_x))
        if (new_junction_x <= start or new_junction_x >= end or
            new_junction_x - start < self.MIN_CHUNK_WIDTH or
            end - new_junction_x < self.MIN_CHUNK_WIDTH):
            print(f"Ignored split: {new_junction_x} too close to bounds {start}-{end}")
            return
        new_chunk1 = (start, new_junction_x)
        new_chunk2 = (new_junction_x + 1, end)
        self.chunks = self.chunks[:chunk_index] + [new_chunk1, new_chunk2] + self.chunks[chunk_index+1:]
        
        # Handle Selection Shift
        was_selected = chunk_index in self.selected_chunks_indices
        shifted_selection = set()
        for selected_idx in self.selected_chunks_indices:
            if selected_idx < chunk_index: shifted_selection.add(selected_idx)
            elif selected_idx > chunk_index: shifted_selection.add(selected_idx + 1)
        if was_selected:
            shifted_selection.add(chunk_index)
            shifted_selection.add(chunk_index + 1)
        self.selected_chunks_indices = shifted_selection
        
        self._plot_chunks_and_junctions()


    def _toggle_chunk_selection(self, chunk_index):
        if chunk_index < 0 or chunk_index >= len(self.chunks): return
        if chunk_index in self.selected_chunks_indices:
            self.selected_chunks_indices.remove(chunk_index)
        else:
            self.selected_chunks_indices.add(chunk_index)
        self._plot_chunks_and_junctions()


    def _delete_junction(self, junction_x):
        idx1, idx2 = self._find_affected_chunks(junction_x)
        if idx1 is None or idx2 is None: return
        
        start1, end1 = self.chunks[idx1]; start2, end2 = self.chunks[idx2]
        if end1 != junction_x or start2 != (junction_x + 1):
             print(f"Inconsistent junction data at {junction_x}. Aborting.")
             if junction_x in self.junction_lines_by_val: del self.junction_lines_by_val[junction_x]
             self._plot_chunks_and_junctions()
             return
             
        chunk1_sel = idx1 in self.selected_chunks_indices
        chunk2_sel = idx2 in self.selected_chunks_indices
        new_chunk_should_be_selected = chunk1_sel or chunk2_sel
        
        merged_chunk = (start1, end2)
        self.chunks = self.chunks[:idx1] + [merged_chunk] + self.chunks[idx2+1:]
        
        new_selection = set()
        for selected_idx in self.selected_chunks_indices:
            if selected_idx < idx1: new_selection.add(selected_idx)
            elif selected_idx > idx2: new_selection.add(selected_idx - 1)
        if new_chunk_should_be_selected: new_selection.add(idx1)
        self.selected_chunks_indices = new_selection

        # --- FIX START: Explicitly remove artists before deleting from dictionary ---
        if junction_x in self.junction_lines_by_val:
            for line in self.junction_lines_by_val[junction_x]:
                try:
                    line.remove()
                except (ValueError, AttributeError):
                    pass
            del self.junction_lines_by_val[junction_x]
        # --- FIX END ---
        
        self._plot_chunks_and_junctions()


    def _find_affected_chunks(self, junction_x):
        idx_ending, idx_starting = None, None
        for i, (start, end) in enumerate(self.chunks):
            if end == junction_x: idx_ending = i
            if start == (junction_x + 1): idx_starting = i
            if idx_ending == i and i + 1 < len(self.chunks): 
                if self.chunks[i+1][0] == (junction_x + 1): idx_starting = i + 1; break
        if idx_ending is not None and idx_starting == idx_ending + 1: return idx_ending, idx_starting
        else: return None, None

    def get_final_chunks(self):
        return list(self.chunks)

    def get_selected_chunks(self):
        return [self.chunks[i] for i in sorted(list(self.selected_chunks_indices))]