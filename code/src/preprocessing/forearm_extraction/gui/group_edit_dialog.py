import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Set, Tuple


class GroupEditDialog:
    """
    Modal popup for editing a single frame group.

    Allows the user to change the set of frame IDs that belong to the group
    and to re-assign the representative frame.

    Usage::

        dialog = GroupEditDialog(parent, group_index, group, representative,
                                 total_frames, other_claimed_frames)
        parent.wait_window(dialog.window)
        if dialog.confirmed:
            new_group = dialog.result_group
            new_rep   = dialog.result_representative

    Args:
        parent: Parent widget (used for grab and positioning).
        group_index: 0-based index of the group being edited (for the title).
        group: Current list of frame IDs in the group.
        representative: Current representative frame ID.
        total_frames: Total number of frames in the video (upper bound check).
        other_claimed_frames: Frame IDs that belong to *other* groups and
            therefore cannot be assigned to this group.
    """

    def __init__(
        self,
        parent: tk.BaseWidget,
        group_index: int,
        group: List[int],
        representative: int,
        total_frames: int,
        other_claimed_frames: Set[int],
    ) -> None:
        self.parent = parent
        self.group_index = group_index
        self.total_frames = total_frames
        self.other_claimed_frames = other_claimed_frames

        self.confirmed = False
        self.result_group: List[int] = list(group)
        self.result_representative: int = representative

        self.window = tk.Toplevel(parent)
        self.window.title(f"Edit Group {group_index + 1}")
        self.window.resizable(False, False)
        self.window.grab_set()

        self._build_ui(group, representative)

        self.window.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - self.window.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - self.window.winfo_height() // 2
        self.window.geometry(f"+{px}+{py}")

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self, group: List[int], representative: int) -> None:
        outer = ttk.Frame(self.window, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # --- Group type (read-only info row) ---------------------------------
        n = len(group)
        if n == 1:
            type_str = "Single frame"
        elif group == list(range(group[0], group[-1] + 1)):
            type_str = f"Range  ({n} frames,  {group[0]:04d} – {group[-1]:04d})"
        else:
            type_str = f"Custom  ({n} frames)"

        info_row = ttk.Frame(outer)
        info_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(info_row, text="Type:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(info_row, text=f"  {type_str}").pack(side=tk.LEFT)

        # --- Frames editor ---------------------------------------------------
        ttk.Label(outer, text="Frames  (comma-separated integers):").pack(anchor=tk.W)

        text_frame = ttk.Frame(outer)
        text_frame.pack(fill=tk.X, pady=(2, 12))

        self.frames_text = tk.Text(
            text_frame, height=4, width=52, wrap=tk.WORD, font=("Consolas", 9)
        )
        text_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.frames_text.yview)
        self.frames_text.configure(yscrollcommand=text_scroll.set)
        self.frames_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.frames_text.insert("1.0", ", ".join(str(f) for f in sorted(group)))

        # --- Representative frame --------------------------------------------
        rep_row = ttk.Frame(outer)
        rep_row.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(rep_row, text="Representative frame:").pack(side=tk.LEFT)
        self.rep_entry = ttk.Entry(rep_row, width=8, justify=tk.RIGHT)
        self.rep_entry.insert(0, str(representative))
        self.rep_entry.pack(side=tk.LEFT, padx=(8, 0))

        # --- Validation feedback ---------------------------------------------
        self.feedback_lbl = ttk.Label(outer, text="", foreground="red", wraplength=400)
        self.feedback_lbl.pack(anchor=tk.W, pady=(0, 10))

        # --- Buttons ---------------------------------------------------------
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="OK", command=self._on_ok).pack(side=tk.RIGHT)

        self.window.bind("<Return>", lambda _e: self._on_ok())
        self.window.bind("<KP_Enter>", lambda _e: self._on_ok())
        self.window.bind("<Escape>", lambda _e: self._on_cancel())

    # =========================================================================
    # Validation
    # =========================================================================

    def _parse_and_validate(self) -> Optional[Tuple[List[int], int]]:
        """
        Parses and validates the current entry values.

        Returns:
            ``(sorted_unique_frames, representative)`` if valid, or ``None``
            while setting an explanatory message on ``self.feedback_lbl``.
        """
        # -- Frames -----------------------------------------------------------
        raw = self.frames_text.get("1.0", tk.END).strip()
        if not raw:
            self.feedback_lbl.config(text="Frames list cannot be empty.")
            return None

        try:
            parsed = [int(t.strip()) for t in raw.split(",") if t.strip()]
        except ValueError:
            self.feedback_lbl.config(
                text="All values must be integers separated by commas."
            )
            return None

        if not parsed:
            self.feedback_lbl.config(text="Frames list cannot be empty.")
            return None

        out_of_range = [f for f in parsed if not (0 <= f < self.total_frames)]
        if out_of_range:
            sample = ", ".join(str(f) for f in out_of_range[:10])
            self.feedback_lbl.config(
                text=f"Frame(s) out of range [0 – {self.total_frames - 1}]: {sample}"
            )
            return None

        unique_frames = sorted(set(parsed))
        overlap = set(unique_frames) & self.other_claimed_frames
        if overlap:
            sample = ", ".join(str(f) for f in sorted(overlap)[:10])
            self.feedback_lbl.config(
                text=f"Frame(s) already claimed by another group: {sample}"
            )
            return None

        # -- Representative ---------------------------------------------------
        try:
            rep = int(self.rep_entry.get().strip())
        except ValueError:
            self.feedback_lbl.config(text="Representative frame must be an integer.")
            return None

        if not (0 <= rep < self.total_frames):
            self.feedback_lbl.config(
                text=f"Representative frame {rep} is out of range [0 – {self.total_frames - 1}]."
            )
            return None

        return unique_frames, rep

    # =========================================================================
    # Button handlers
    # =========================================================================

    def _on_ok(self) -> None:
        result = self._parse_and_validate()
        if result is None:
            return
        self.result_group, self.result_representative = result
        self.confirmed = True
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.window.destroy()
