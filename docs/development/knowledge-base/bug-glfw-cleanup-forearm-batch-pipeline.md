# Bug: GLFW Cleanup Warnings in Forearm Batch Pipeline

**Date Reported:** 2026-02-22
**Severity:** Low (warnings only, does not prevent completion)
**Status:** Fixed
**Related Plan:** `docs/development/plans/active/forearm-batch-glfw-cleanup-fix.md`
**Related Script:** `code/scripts/preprocess_pipeline_extract_forearm_manual.py`

---

## Symptom

When `_execute_all_batches()` finishes processing all batches in the forearm extraction pipeline, repeated GLFW warnings flood the terminal:

```
[Open3D WARNING] GLFW Error: The GLFW library is not initialized
[error] GLFW error: The GLFW library is not initialized
```

This repeats 6+ times, cluttering the terminal output. The script completes successfully; the warnings are spurious.

### When It Occurs

- **Trigger:** End of `_execute_all_batches()` after all mesh visualization windows have been closed
- **Frequency:** Every run that includes mesh visualization (`define_forearm_mesh(..., show=True)`)
- **Platform:** Observed on WSL2/Linux

---

## Investigation

### Root Cause Analysis

The GLFW warnings originate from Open3D's visualization cleanup sequence:

1. Each call to `define_forearm_mesh(..., show=True)` opens an Open3D visualization window
2. The window uses GLFW for rendering and event handling
3. When the user closes the window, Open3D's `draw_geometries_with_key_callbacks()` returns
4. **BUT:** GLFW context is not properly cleaned up between iterations or at script exit
5. At program termination, the dangling GLFW context attempts cleanup operations and fails silently, emitting warnings

### Why the Initial Fix Failed

The attempted fix was:
```python
def _execute_all_batches(batches: List[FrameBatch]) -> None:
    # ... process batches ...

    # Clean up Open3D/GLFW context to suppress spurious warnings at exit
    try:
        import open3d as o3d
        o3d.visualization.draw_geometries([])
    except Exception:
        pass
```

**Problem:** Calling `draw_geometries([])` with an empty list does not actually initialize/reset GLFW; it's a no-op that returns immediately without touching the GLFW context. The GLFW library was already partially torn down by the visualization windows, so this call doesn't help.

---

## Known Alternatives

### Option 1: Suppress visualization in batch mode (Most practical)
Set `show=False` in the batch pipeline since batch processing doesn't need interactive windows.

**Pros:**
- Eliminates GLFW initialization entirely
- Faster processing (no window management overhead)
- Clean output

**Cons:**
- Lose visual inspection during processing

### Option 2: Force GLFW shutdown between iterations
After each `define_forearm_mesh()` call, explicitly cleanup:
```python
try:
    from glfw import _glfw as glfw_lib
    if glfw_lib.glfwTerminate:
        glfw_lib.glfwTerminate()
except Exception:
    pass
```

**Pros:**
- Keeps visualization enabled
- May prevent context corruption

**Cons:**
- Fragile (relies on internal GLFW bindings)
- Requires testing on multiple platforms
- Could cause crashes if Open3D still holds references

### Option 3: Use Open3D's non-blocking visualization
Switch to `open3d.visualization.Visualizer` (non-blocking) instead of `draw_geometries()`.

**Pros:**
- More control over GLFW lifecycle
- Can properly close visualizer before exit

**Cons:**
- Significant refactoring of `define_forearm_mesh()`
- Requires rewriting visualization interaction loop

### Option 4: Redirect stderr during visualization
Suppress GLFW warnings without fixing root cause:
```python
import sys
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()
# ... run visualizations ...
sys.stderr = old_stderr
```

**Pros:**
- Minimal code change

**Cons:**
- Hides legitimate errors
- Poor engineering practice

---

## Recommended Fix

**Option 1:** Set `show=False` in batch mode.

**Rationale:**
- Batch processing is typically unattended/automated; interactive windows interrupt the flow
- Clean solution that eliminates the root cause rather than patching symptoms
- No fragility or platform-specific code
- Fastest processing

**Implementation:**
```python
# In execute_frame_batch()
def execute_frame_batch(batch: FrameBatch, interactive: bool = True) -> None:
    # ... existing code ...
    print(f"  🕸️  [4/4] Building mesh")
    define_forearm_mesh(
        source=batch.normals_ply,
        output_path=batch.mesh_obj,
        show=interactive  # ← Use parameter instead of hardcoded True
    )
    print(f"         → {batch.mesh_obj.name}\n")

# In run_session() or _execute_all_batches()
_execute_all_batches(batches, interactive=False)  # Batch mode: no visualization
```

---

## References

- **Open3D visualization:** `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py:151-161`
- **Batch pipeline:** `code/scripts/preprocess_pipeline_extract_forearm_manual.py:289-303`
- **Related WSL2 issue:** Platform-specific GLFW behavior on virtualized Linux

---

## Testing Plan

After fix implementation:

1. **With `show=False`:**
   - [ ] Run batch pipeline end-to-end
   - [ ] Verify no GLFW warnings appear
   - [ ] Confirm all meshes are saved correctly

2. **With `show=True` (single-session interactive mode):**
   - [ ] Manually run single-session workflow
   - [ ] Verify visualization windows open and close properly
   - [ ] Note any GLFW warnings (expected, but acceptable for interactive use)

---
