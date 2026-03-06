import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from PIL import Image, ImageTk

from preprocessing.common import VideoMP4Manager
from preprocessing.forearm_extraction.gui.group_edit_dialog import GroupEditDialog


# --- High DPI Awareness for Windows 11 ---
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class VideoFramesSelector:
    """
    A GUI to view a video and select frame groups for averaging.

    Supports three selection modes:
    - Single Frame (default): each click adds the current frame as a 1-frame group
      (clicking again removes it — toggle behaviour matching the legacy UX).
    - Range: user sets a start frame and an end frame, then confirms the consecutive
      range as a single group.
    - Custom Group: user accumulates non-consecutive frames one at a time, then
      finalises them as a group.

    For every group the user may override the representative frame (the temporal anchor
    used by downstream consumers) via the "Mark as Representative" button.
    """

    _MODE_SINGLE = "single"
    _MODE_RANGE = "range"
    _MODE_CUSTOM = "custom"

    def __init__(
        self,
        parent: tk.Toplevel,
        video_manager: "VideoMP4Manager",
        last_frame: int,
        title: str = "Frame Selector",
        initial_groups: Optional[List[List[int]]] = None,
        initial_representatives: Optional[List[int]] = None,
    ):
        self.parent = parent
        self.manager = video_manager
        self.total_frames = last_frame + 1
        self.title = title

        # --- Core data structures ---
        self.groups: List[List[int]] = []
        self.group_representatives: List[int] = []
        if initial_groups and initial_representatives:
            self.groups = [list(g) for g in initial_groups]
            self.group_representatives = list(initial_representatives)

        self.proceed_was_clicked = False
        self.current_frame_idx = 0
        self.photo_image = None

        # Range mode state
        self.range_start: Optional[int] = None
        self.range_end: Optional[int] = None

        # Custom group mode state
        self.pending_group: List[int] = []

        # --- Window ---
        self.parent.title(self.title)
        self.parent.minsize(800, 580)
        try:
            self.parent.state("zoomed")      # Windows
        except tk.TclError:
            self.parent.attributes("-zoomed", True)  # Linux / X11

        self._build_ui()
        self._update_image(0)
        self._refresh_groups_listbox()
        self._update_status()
        self._sync_frame_entry(0)
        self._sync_single_rep_entry(0)

        # Keyboard navigation (← / → / Space / Delete)
        self.parent.bind("<Left>", self._on_key_left)
        self.parent.bind("<Right>", self._on_key_right)
        self.parent.bind("<space>", self._on_key_space)
        self.parent.bind("<Delete>", lambda _e: self._remove_selected_group())

        # Sash is set after the WM delivers real (maximised) dimensions
        self.parent.after(100, self._set_initial_sash)

    # =========================================================================
    # UI Construction
    # =========================================================================

    def _build_ui(self) -> None:
        """Construct all UI elements."""
        # Horizontal split: left (video + controls) | right (Frame Groups panel)
        self.paned = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL, sashwidth=5)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ----- Left panel -----
        left_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, minsize=580)

        # Video canvas — fills all available space; redraws on resize
        self.canvas = tk.Canvas(left_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind(
            "<Configure>", lambda _e: self._update_image(self.current_frame_idx)
        )

        # Group indicator strip (sits directly above the slider)
        self.group_indicator = tk.Canvas(
            left_frame, height=16, bg="#1a1a1a", bd=0, highlightthickness=0
        )
        self.group_indicator.pack(fill=tk.X)
        self.group_indicator.bind("<Configure>", lambda _e: self._redraw_group_indicator())

        # Slider
        self.slider = tk.Scale(
            left_frame,
            from_=0,
            to=self.total_frames - 1,
            orient=tk.HORIZONTAL,
            command=self._on_slider_move,
            label="Frame Index",
        )
        self.slider.pack(fill=tk.X, pady=(4, 0))

        # Compact navigation row (replaces standalone frame_info_lbl)
        nav_row = ttk.Frame(left_frame)
        nav_row.pack(fill=tk.X, padx=6, pady=(0, 2))
        ttk.Label(nav_row, text="Frame:").pack(side=tk.LEFT)
        self.frame_entry = ttk.Entry(nav_row, width=6, justify=tk.RIGHT)
        self.frame_entry.pack(side=tk.LEFT, padx=(3, 0))
        self.frame_entry.bind("<Return>", self._on_frame_entry_submit)
        self.frame_entry.bind("<KP_Enter>", self._on_frame_entry_submit)
        ttk.Label(nav_row, text=f"/ {self.total_frames - 1}").pack(side=tk.LEFT, padx=(3, 0))

        # Mode selection
        mode_frame = ttk.LabelFrame(left_frame, text="Selection Mode", padding=8)
        mode_frame.pack(fill=tk.X, padx=5, pady=4)

        self.mode_var = tk.StringVar(value=self._MODE_SINGLE)
        for text, value in [
            ("Single Frame", self._MODE_SINGLE),
            ("Range", self._MODE_RANGE),
            ("Custom Group", self._MODE_CUSTOM),
        ]:
            ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=12)

        # Action area — one sub-frame per mode, shown/hidden on mode change
        self.action_outer = ttk.LabelFrame(left_frame, text="Actions", padding=8)
        self.action_outer.pack(fill=tk.X, padx=5, pady=4)

        self._build_single_actions(self.action_outer)
        self._build_range_actions(self.action_outer)
        self._build_custom_actions(self.action_outer)

        # Status bar + Save button
        bottom_bar = ttk.Frame(left_frame)
        bottom_bar.pack(fill=tk.X, padx=5, pady=5)

        self.status_lbl = ttk.Label(bottom_bar, text="")
        self.status_lbl.pack(side=tk.LEFT)

        ttk.Button(
            bottom_bar,
            text="Save & Proceed",
            command=self._on_proceed,
        ).pack(side=tk.RIGHT, padx=5)

        # ----- Right panel: Frame Groups -----
        right_frame = ttk.LabelFrame(self.paned, text="Frame Groups", padding=8)
        self.paned.add(right_frame, minsize=100)

        lb_container = ttk.Frame(right_frame)
        lb_container.pack(fill=tk.BOTH, expand=True)

        self.groups_listbox = tk.Listbox(
            lb_container, selectmode=tk.SINGLE, activestyle="dotbox", font=("Consolas", 9)
        )
        lb_scroll = ttk.Scrollbar(
            lb_container, orient=tk.VERTICAL, command=self.groups_listbox.yview
        )
        self.groups_listbox.configure(yscrollcommand=lb_scroll.set)
        self.groups_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.groups_listbox.bind("<<ListboxSelect>>", self._on_group_select)
        self.groups_listbox.bind("<Double-Button-1>", self._on_group_double_click)

        ttk.Button(
            right_frame, text="Remove Selected Group", command=self._remove_selected_group
        ).pack(fill=tk.X, pady=(8, 4))

        self.mark_rep_btn = ttk.Button(
            right_frame,
            text="Mark as Representative",
            command=self._mark_as_representative,
            state=tk.DISABLED,
        )
        self.mark_rep_btn.pack(fill=tk.X, pady=(0, 4))

        self.groups_summary_lbl = ttk.Label(right_frame, text="Groups defined: 0")
        self.groups_summary_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Show the default mode panel
        self._on_mode_change()

    def _build_single_actions(self, parent: ttk.LabelFrame) -> None:
        self.single_widgets = ttk.Frame(parent)
        self.btn_single_toggle = ttk.Button(
            self.single_widgets, text="Add as Group", command=self._single_add_or_remove
        )
        self.btn_single_toggle.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.single_widgets, text="Rep. frame:").pack(side=tk.LEFT, padx=(8, 2))
        self.single_rep_entry = ttk.Entry(self.single_widgets, width=6, justify=tk.RIGHT)
        self.single_rep_entry.pack(side=tk.LEFT, padx=(0, 3))

    def _build_range_actions(self, parent: ttk.LabelFrame) -> None:
        self.range_widgets = ttk.Frame(parent)

        self.btn_set_start = ttk.Button(
            self.range_widgets, text="Set Start", command=self._range_set_start
        )
        self.btn_set_start.pack(side=tk.LEFT, padx=3)

        self.range_start_lbl = ttk.Label(self.range_widgets, text="Start: —", width=12)
        self.range_start_lbl.pack(side=tk.LEFT, padx=3)

        self.btn_set_end = ttk.Button(
            self.range_widgets, text="Set End", command=self._range_set_end
        )
        self.btn_set_end.pack(side=tk.LEFT, padx=3)

        self.range_end_lbl = ttk.Label(self.range_widgets, text="End: —", width=12)
        self.range_end_lbl.pack(side=tk.LEFT, padx=3)

        ttk.Label(self.range_widgets, text="Rep. frame:").pack(side=tk.LEFT, padx=(8, 2))
        self.range_rep_entry = ttk.Entry(self.range_widgets, width=6, justify=tk.RIGHT)
        self.range_rep_entry.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            self.range_widgets, text="Add Range as Group", command=self._range_add_group
        ).pack(side=tk.LEFT, padx=6)

    def _build_custom_actions(self, parent: ttk.LabelFrame) -> None:
        self.custom_widgets = ttk.Frame(parent)

        ttk.Button(
            self.custom_widgets, text="Add Frame to Group", command=self._custom_add_frame
        ).pack(side=tk.LEFT, padx=3)

        self.pending_lbl = ttk.Label(self.custom_widgets, text="0 frames pending", width=18)
        self.pending_lbl.pack(side=tk.LEFT, padx=6)

        ttk.Label(self.custom_widgets, text="Rep. frame:").pack(side=tk.LEFT, padx=(8, 2))
        self.custom_rep_entry = ttk.Entry(self.custom_widgets, width=6, justify=tk.RIGHT)
        self.custom_rep_entry.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_finalize = ttk.Button(
            self.custom_widgets,
            text="Finalize Group",
            command=self._custom_finalize_group,
            state=tk.DISABLED,
        )
        self.btn_finalize.pack(side=tk.LEFT, padx=3)

        ttk.Button(
            self.custom_widgets, text="Clear", command=self._custom_clear_pending
        ).pack(side=tk.LEFT, padx=3)

    def _set_initial_sash(self) -> None:
        """Position the sash so the Frame Groups panel takes 1/5 of the window width.

        Called via after() to ensure the window manager has delivered the real
        maximised dimensions before the sash is placed. Retries every 100 ms
        while the paned window is still in its initial un-expanded state.
        """
        total = self.paned.winfo_width()
        if total > 680:  # 680 = sum of minsizes (580 left + 100 right)
            self.paned.sash_place(0, (total * 4) // 5, 0)
        else:
            self.parent.after(100, self._set_initial_sash)

    # =========================================================================
    # Mode switching
    # =========================================================================

    def _on_mode_change(self) -> None:
        """Show the correct action widget set for the active mode."""
        self.single_widgets.pack_forget()
        self.range_widgets.pack_forget()
        self.custom_widgets.pack_forget()

        mode = self.mode_var.get()
        if mode == self._MODE_SINGLE:
            self.single_widgets.pack(fill=tk.X)
        elif mode == self._MODE_RANGE:
            self.range_widgets.pack(fill=tk.X)
        else:
            self.custom_widgets.pack(fill=tk.X)

        self._update_status()

    # =========================================================================
    # Slider / video display
    # =========================================================================

    def _on_slider_move(self, value: str) -> None:
        idx = int(value)
        self.current_frame_idx = idx
        self._update_image(idx)
        self._update_status()
        self._update_mark_rep_btn()
        self._sync_frame_entry(idx)
        self._sync_single_rep_entry(idx)

    def _update_image(self, frame_idx: int) -> None:
        try:
            frame = self.manager[frame_idx]
            if self.manager.color_format.name == "BGR":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw, _ = frame.shape
            cw = self.canvas.winfo_width() or 680
            ch = self.canvas.winfo_height() or 420
            scale = min(cw / fw, ch / fh)
            display_w = max(1, int(fw * scale))
            display_h = max(1, int(fh * scale))
            frame_resized = cv2.resize(frame, (display_w, display_h))
            image = Image.fromarray(frame_resized)
            self.photo_image = ImageTk.PhotoImage(image)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo_image, anchor=tk.CENTER)
        except Exception as exc:
            print(f"Error displaying frame {frame_idx}: {exc}")

    # =========================================================================
    # Single Frame mode
    # =========================================================================

    def _single_add_or_remove(self) -> None:
        """Toggle: add current frame as a 1-frame group, or remove if it already exists.

        If the frame is claimed by a different (multi-frame) group, does nothing silently.
        """
        idx = self.current_frame_idx
        existing = next((i for i, g in enumerate(self.groups) if g == [idx]), None)
        if existing is not None:
            # Toggle off: remove the exact 1-frame group
            self.groups.pop(existing)
            self.group_representatives.pop(existing)
        elif idx not in self._claimed_frames():
            # Read representative from entry; fall back to idx if invalid
            rep = idx
            try:
                entry_val = int(self.single_rep_entry.get().strip())
                if 0 <= entry_val < self.total_frames:
                    rep = entry_val
            except ValueError:
                pass
            self.groups.append([idx])
            self.group_representatives.append(rep)
        # else: claimed by a different group — silently skip
        self._refresh_groups_listbox()
        self._update_status()

    # =========================================================================
    # Range mode
    # =========================================================================

    def _range_set_start(self) -> None:
        self.range_start = self.current_frame_idx
        self.range_start_lbl.config(text=f"Start: {self.range_start:04d}")
        # Default rep = start frame; user can override before clicking Add
        self.range_rep_entry.delete(0, tk.END)
        self.range_rep_entry.insert(0, str(self.range_start))

    def _range_set_end(self) -> None:
        self.range_end = self.current_frame_idx
        self.range_end_lbl.config(text=f"End: {self.range_end:04d}")

    def _range_add_group(self) -> None:
        if self.range_start is None or self.range_end is None:
            messagebox.showwarning(
                "Incomplete Range",
                "Please set both a start and an end frame before adding a range.",
                parent=self.parent,
            )
            return
        if self.range_end < self.range_start:
            messagebox.showwarning(
                "Invalid Range",
                f"End frame ({self.range_end}) must be ≥ start frame ({self.range_start}).",
                parent=self.parent,
            )
            return
        group = list(range(self.range_start, self.range_end + 1))
        overlap = set(group) & self._claimed_frames()
        if overlap:
            messagebox.showwarning(
                "Overlap Detected",
                "The range overlaps with existing group(s) at frame(s): "
                + ", ".join(str(f) for f in sorted(overlap)),
                parent=self.parent,
            )
            return
        # Determine representative from entry; fall back to range_start if invalid
        rep = self.range_start
        try:
            entry_val = int(self.range_rep_entry.get().strip())
            if 0 <= entry_val < self.total_frames:
                rep = entry_val
        except ValueError:
            pass
        self.groups.append(group)
        self.group_representatives.append(rep)
        # Reset markers and entry
        self.range_start = None
        self.range_end = None
        self.range_start_lbl.config(text="Start: —")
        self.range_end_lbl.config(text="End: —")
        self.range_rep_entry.delete(0, tk.END)
        self._refresh_groups_listbox()
        self._update_status()

    # =========================================================================
    # Custom Group mode
    # =========================================================================

    def _custom_add_frame(self) -> None:
        idx = self.current_frame_idx
        if idx in self._claimed_frames() or idx in self.pending_group:
            return  # silently skip already-claimed or already-pending frames
        self.pending_group.append(idx)
        self._refresh_pending_label()

    def _custom_finalize_group(self) -> None:
        if not self.pending_group:
            return
        deduped = sorted(set(self.pending_group))
        # Read representative from entry; fall back to first added frame if invalid
        rep = self.pending_group[0]
        try:
            entry_val = int(self.custom_rep_entry.get().strip())
            if 0 <= entry_val < self.total_frames:
                rep = entry_val
        except ValueError:
            pass
        self.groups.append(deduped)
        self.group_representatives.append(rep)
        self._custom_clear_pending()
        self._refresh_groups_listbox()
        self._update_status()

    def _custom_clear_pending(self) -> None:
        self.pending_group = []
        self._refresh_pending_label()

    def _refresh_pending_label(self) -> None:
        n = len(self.pending_group)
        self.pending_lbl.config(text=f"{n} frame{'s' if n != 1 else ''} pending")
        self.btn_finalize.config(state=tk.NORMAL if n > 0 else tk.DISABLED)
        if n == 1:
            # Auto-populate rep entry with the first added frame
            self.custom_rep_entry.delete(0, tk.END)
            self.custom_rep_entry.insert(0, str(self.pending_group[0]))
        elif n == 0:
            self.custom_rep_entry.delete(0, tk.END)

    # =========================================================================
    # Groups listbox
    # =========================================================================

    @staticmethod
    def _group_display_text(group: List[int], representative: int, index: int) -> str:
        """Human-readable label for one group entry."""
        n = len(group)
        if n == 1:
            return f"Group {index + 1}: frame {group[0]:04d}"
        is_consecutive = group == list(range(group[0], group[-1] + 1))
        style = "range" if is_consecutive else "custom"
        return (
            f"Group {index + 1}: frames {group[0]:04d}–{group[-1]:04d} "
            f"({n} frames) [{style}] [rep: {representative:04d}]"
        )

    def _refresh_groups_listbox(self) -> None:
        """Rebuild the listbox from current groups/representatives."""
        sel = self.groups_listbox.curselection()
        self.groups_listbox.delete(0, tk.END)
        for i, (group, rep) in enumerate(zip(self.groups, self.group_representatives)):
            self.groups_listbox.insert(tk.END, self._group_display_text(group, rep, i))
        # Restore selection if still valid
        if sel:
            prev_idx = sel[0]
            if prev_idx < len(self.groups):
                self.groups_listbox.selection_set(prev_idx)
        n_groups = len(self.groups)
        n_frames = sum(len(g) for g in self.groups)
        self.groups_summary_lbl.config(
            text=f"Groups defined: {n_groups} | Frames covered: {n_frames}"
        )
        self._update_mark_rep_btn()
        self._redraw_group_indicator()

    def _on_group_select(self, _event=None) -> None:
        self._update_mark_rep_btn()

    def _on_group_double_click(self, _event=None) -> None:
        """Open the group-edit dialog for the double-clicked group."""
        sel = self.groups_listbox.curselection()
        if not sel:
            return
        idx = sel[0]

        # Frames claimed by every group *except* the one being edited
        other_claimed = {
            fid
            for i, g in enumerate(self.groups)
            if i != idx
            for fid in g
        }

        dialog = GroupEditDialog(
            parent=self.parent,
            group_index=idx,
            group=self.groups[idx],
            representative=self.group_representatives[idx],
            total_frames=self.total_frames,
            other_claimed_frames=other_claimed,
        )
        self.parent.wait_window(dialog.window)

        if dialog.confirmed:
            self.groups[idx] = dialog.result_group
            self.group_representatives[idx] = dialog.result_representative
            self._refresh_groups_listbox()
            self._update_status()

    def _remove_selected_group(self) -> None:
        sel = self.groups_listbox.curselection()
        if not sel:
            messagebox.showinfo(
                "No Selection", "Select a group in the list to remove.", parent=self.parent
            )
            return
        idx = sel[0]
        self.groups.pop(idx)
        self.group_representatives.pop(idx)
        self._refresh_groups_listbox()
        self._update_status()

    # =========================================================================
    # Mark as Representative
    # =========================================================================

    def _update_mark_rep_btn(self) -> None:
        """Enable the button whenever a group is selected."""
        sel = self.groups_listbox.curselection()
        if sel:
            self.mark_rep_btn.config(state=tk.NORMAL)
        else:
            self.mark_rep_btn.config(state=tk.DISABLED)

    def _mark_as_representative(self) -> None:
        sel = self.groups_listbox.curselection()
        if not sel:
            return
        group_idx = sel[0]
        self.group_representatives[group_idx] = self.current_frame_idx
        self._refresh_groups_listbox()
        # Restore selection after refresh
        self.groups_listbox.selection_set(group_idx)
        self._update_mark_rep_btn()

    # =========================================================================
    # Status
    # =========================================================================

    def _update_status(self) -> None:
        idx = self.current_frame_idx

        if self.mode_var.get() == self._MODE_SINGLE:
            already_added = any(g == [idx] for g in self.groups)
            self.btn_single_toggle.config(
                text="Remove Group" if already_added else "Add as Group"
            )

        n_groups = len(self.groups)
        n_frames = sum(len(g) for g in self.groups)
        self.status_lbl.config(
            text=f"Frame {idx} | Groups: {n_groups} | Frames covered: {n_frames}"
        )

    # =========================================================================
    # Frame entry navigation
    # =========================================================================

    def _on_frame_entry_submit(self, _event=None) -> None:
        """Jump to the frame number typed in the entry box and press Enter."""
        text = self.frame_entry.get().strip()
        try:
            idx = int(text)
        except ValueError:
            return  # non-numeric input — no-op
        idx = max(0, min(self.total_frames - 1, idx))
        self.slider.set(idx)  # fires _on_slider_move for full update
        self.parent.focus_set()  # restore keyboard navigation focus

    def _sync_frame_entry(self, idx: int) -> None:
        """Keep the frame entry in sync with the slider.

        Skipped when the entry currently has keyboard focus to avoid overwriting
        in-progress user input.
        """
        if self.parent.focus_get() is not self.frame_entry:
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(idx))

    def _sync_single_rep_entry(self, idx: int) -> None:
        """Keep the Single Frame rep entry in sync with the slider.

        Skipped when the entry has focus so the user can type without interruption.
        """
        if self.parent.focus_get() is not self.single_rep_entry:
            self.single_rep_entry.delete(0, tk.END)
            self.single_rep_entry.insert(0, str(idx))

    # =========================================================================
    # Keyboard navigation
    # =========================================================================

    def _on_key_left(self, _event=None) -> None:
        """Step one frame back; no-op when the frame entry has focus."""
        if self.parent.focus_get() is self.frame_entry:
            return
        new_idx = max(0, self.current_frame_idx - 1)
        self.slider.set(new_idx)

    def _on_key_right(self, _event=None) -> None:
        """Step one frame forward; no-op when the frame entry has focus."""
        if self.parent.focus_get() is self.frame_entry:
            return
        new_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
        self.slider.set(new_idx)

    def _on_key_space(self, _event=None) -> None:
        """Mode-aware action for the current frame — no-op when the entry has focus.

        Single Frame : toggle add/remove (same as the button).
        Range        : first press sets Start; second press sets End; if both are
                       already set, resets and sets a new Start.
        Custom Group : add the current frame to the pending buffer.
        """
        if self.parent.focus_get() is self.frame_entry:
            return
        mode = self.mode_var.get()
        if mode == self._MODE_SINGLE:
            self._single_add_or_remove()
        elif mode == self._MODE_RANGE:
            if self.range_start is None or self.range_end is not None:
                # No start yet, or both already set — begin a fresh range
                self._range_set_start()
            else:
                # Start is set, end is not — complete the pair
                self._range_set_end()
        elif mode == self._MODE_CUSTOM:
            self._custom_add_frame()

    # =========================================================================
    # Group indicator strip
    # =========================================================================

    def _redraw_group_indicator(self) -> None:
        """Draw a 1 px red vertical line for each frame belonging to any group."""
        self.group_indicator.delete("all")
        canvas_width = self.group_indicator.winfo_width()
        canvas_height = self.group_indicator.winfo_height()
        if canvas_width <= 1 or self.total_frames <= 1:
            return
        for group in self.groups:
            for fid in group:
                x = int(fid / (self.total_frames - 1) * (canvas_width - 1))
                self.group_indicator.create_line(
                    x, 0, x, canvas_height, fill="#ff3333", width=1
                )

    # =========================================================================
    # Frame exclusivity helper
    # =========================================================================

    def _claimed_frames(self) -> set:
        """Return the union of all frame indices across all committed groups."""
        return {fid for group in self.groups for fid in group}

    # =========================================================================
    # Proceed
    # =========================================================================

    def _on_proceed(self) -> None:
        self.proceed_was_clicked = True
        self.parent.destroy()


# =============================================================================
# Multi-video coordinator
# =============================================================================

class MultiVideoFramesSelector:
    """
    Manages frame-group selection across multiple video files.

    After the user validates, results are available via `all_selected_groups`:
        {
            "/full/path/to/video.mp4": {
                "groups": [[42], [67, 68, 69]],
                "representatives": [42, 67],
            },
            ...
        }
    """

    def __init__(
        self,
        parent: tk.Tk,
        video_paths,
        initial_groups: Optional[Dict[str, dict]] = None,
    ):
        self.parent = parent
        self.video_paths = [str(p) for p in video_paths]
        # basename → full path lookup
        self.video_paths_dict: Dict[str, str] = {
            os.path.basename(p): p for p in self.video_paths
        }

        # All group data keyed by full path
        self.all_selected_groups: Dict[str, dict] = {}
        if initial_groups:
            for basename, group_data in initial_groups.items():
                if basename in self.video_paths_dict:
                    full_path = self.video_paths_dict[basename]
                    self.all_selected_groups[full_path] = {
                        "groups": [list(g) for g in group_data.get("groups", [])],
                        "representatives": list(group_data.get("representatives", [])),
                    }
                else:
                    print(f"Warning: basename '{basename}' from initial_groups not found.")

        self.selectors_opened = {fn: False for fn in self.video_paths_dict}
        self.validated = False

        self.selected_video = tk.StringVar()
        self.status_labels: Dict[str, tuple] = {}

        # --- Window ---
        self.parent.title("Multi-Video Frame Group Selector")

        main_frame = ttk.Frame(self.parent, padding=15)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Select a video to define frame groups:").grid(
            row=0, column=0, pady=(0, 10), sticky=tk.W
        )

        self.video_list_frame = ttk.LabelFrame(main_frame, text="Videos | Status", padding=10)
        self.video_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self._create_video_list_ui()

        ttk.Button(
            main_frame,
            text="Select Frame Groups for this Video",
            command=self.open_frame_selector,
        ).grid(row=2, column=0, pady=15, ipady=5, sticky=tk.EW)

        ttk.Button(
            main_frame,
            text="Validate Selections and Exit",
            command=self.validate_and_close,
        ).grid(row=3, column=0, pady=(5, 0), ipady=5, sticky=tk.EW)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        self._update_status_display()

        self.parent.update_idletasks()
        fixed_width = 800
        required_height = min(self.parent.winfo_reqheight(), 800)
        self.parent.geometry(f"{fixed_width}x{required_height}")
        self.parent.minsize(fixed_width, 400)

    def _create_video_list_ui(self) -> None:
        style = ttk.Style()
        style.configure("GroupStatus.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))

        canvas = tk.Canvas(self.video_list_frame)
        scrollbar = ttk.Scrollbar(self.video_list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, filename in enumerate(self.video_paths_dict):
            item_frame = ttk.Frame(scrollable_frame)
            item_frame.grid(row=i, column=0, sticky=tk.W, pady=2)

            ttk.Radiobutton(
                item_frame,
                text=filename,
                variable=self.selected_video,
                value=filename,
            ).grid(row=0, column=0, sticky=tk.W)

            opened_lbl = ttk.Label(item_frame, text="", style="GroupStatus.TLabel", width=8)
            opened_lbl.grid(row=0, column=1, padx=(10, 0))

            groups_lbl = ttk.Label(item_frame, text="", style="GroupStatus.TLabel", width=12)
            groups_lbl.grid(row=0, column=2)

            self.status_labels[filename] = (opened_lbl, groups_lbl)

        if self.video_paths_dict:
            self.selected_video.set(next(iter(self.video_paths_dict)))

    def _update_status_display(self) -> None:
        for filename, (opened_lbl, groups_lbl) in self.status_labels.items():
            opened_lbl.config(text="[Viewed]" if self.selectors_opened.get(filename) else "")
            video_path = self.video_paths_dict[filename]
            data = self.all_selected_groups.get(video_path, {})
            n_groups = len(data.get("groups", []))
            groups_lbl.config(text=f"[{n_groups} Group{'s' if n_groups != 1 else ''}]" if n_groups > 0 else "")

    def open_frame_selector(self) -> None:
        """Open the single-video frame group selector in a modal window."""
        video_filename = self.selected_video.get()
        if not video_filename:
            messagebox.showwarning("No Video Selected", "Please select a video from the list.")
            return

        self.selectors_opened[video_filename] = True
        video_path = self.video_paths_dict[video_filename]

        try:
            video_manager = VideoMP4Manager(video_path)
        except FileNotFoundError as exc:
            messagebox.showerror("File Not Found", str(exc))
            return

        # Retrieve existing groups/representatives for this video (if any)
        existing = self.all_selected_groups.get(video_path, {})
        initial_groups = existing.get("groups", [])
        initial_representatives = existing.get("representatives", [])

        selector_window = tk.Toplevel(self.parent)
        selector_window.grab_set()

        selector = VideoFramesSelector(
            parent=selector_window,
            video_manager=video_manager,
            last_frame=video_manager.total_frames - 1,
            title=f"Frame Group Selector: {video_filename}",
            initial_groups=initial_groups,
            initial_representatives=initial_representatives,
        )

        self.parent.wait_window(selector_window)

        if selector.proceed_was_clicked:
            if selector.groups:
                self.all_selected_groups[video_path] = {
                    "groups": selector.groups,
                    "representatives": selector.group_representatives,
                }
                n = len(selector.groups)
                print(
                    f"Stored {n} group(s) for '{video_filename}': "
                    + str([g for g in selector.groups])
                )
            elif video_path in self.all_selected_groups:
                del self.all_selected_groups[video_path]
                print(f"Cleared all groups for '{video_filename}'.")
        else:
            print(f"Selection cancelled for '{video_filename}'.")

        self._update_status_display()

    def validate_and_close(self) -> None:
        """Close the main window and flag that the process is complete."""
        self.validated = True
        self.parent.destroy()


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    temp_dir = Path("./temp_videos")
    temp_dir.mkdir(exist_ok=True)

    dummy_vid_path = temp_dir / "test_video_1.mp4"
    if not dummy_vid_path.exists():
        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(dummy_vid_path), fourcc, 30.0, (width, height))
        for i in range(90):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (i * 2, 255 - i * 2, 100)
            cv2.putText(
                frame, f"Frame {i}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
            )
            out.write(frame)
        out.release()

    root = tk.Tk()
    app = MultiVideoFramesSelector(root, video_paths=[str(dummy_vid_path)])
    root.mainloop()

    if app.validated:
        print("\n✅ Selections Validated!")
        print(app.all_selected_groups)
    else:
        print("\n❌ Window closed without validation.")
