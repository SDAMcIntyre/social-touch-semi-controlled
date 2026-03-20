"""
Diagnostic script: isolate the FEngine/OpenGL wglCreateContextAttribs failure.

The error:
    wglCreateContextAttribs() failed, whdc=...
    Windows error code: 87

means the GPU driver rejects the OpenGL context version that Open3D's
Filament renderer requests.  Common causes:
  - Remote Desktop (RDP) — RDP replaces the GPU driver with a software one
    that caps at OpenGL 1.1
  - Old / missing GPU drivers
  - VM without GPU passthrough
  - Integrated GPU that lacks OpenGL 4.1+

This script tests operations in isolation, from harmless to the exact call
that triggers the crash, so you can see which step fails.

Run:  python code/scripts/_3_preprocessing/_3_forearm_extraction/debug_opengl.py
"""

import os
import sys
import platform
import traceback
import ctypes


# ── Helpers ──────────────────────────────────────────────────────
def banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def step(label, func):
    """Run *func*; print OK / FAILED with full traceback."""
    print(f"\n  [{label}] ... ", end="", flush=True)
    try:
        result = func()
        print("OK")
        if result is not None:
            for line in str(result).splitlines():
                print(f"    {line}")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════
# PHASE 0 — System / GPU information
# ═══════════════════════════════════════════════════════════════════
banner("Phase 0: System & GPU information")

print(f"  Python       : {sys.version}")
print(f"  Platform     : {platform.platform()}")
print(f"  Architecture : {platform.machine()}")
print(f"  CWD          : {os.getcwd()}")

# Detect RDP session (most common cause on Windows)
def _check_rdp():
    msgs = []
    # Method 1: check SM_REMOTESESSION
    try:
        user32 = ctypes.windll.user32
        SM_REMOTESESSION = 0x1000
        if user32.GetSystemMetrics(SM_REMOTESESSION) != 0:
            msgs.append("SM_REMOTESESSION=1 → running inside Remote Desktop")
    except Exception:
        pass
    # Method 2: check sessionname env var
    session = os.environ.get("SESSIONNAME", "")
    if session:
        msgs.append(f"SESSIONNAME={session}")
        if "RDP" in session.upper():
            msgs.append("⚠ Session name contains 'RDP' — likely a Remote Desktop session")
    if not msgs:
        msgs.append("No RDP indicators detected (local console session)")
    return "\n".join(msgs)

step("RDP / remote session check", _check_rdp)

# Query OpenGL version via ctypes (without Open3D)
def _query_opengl_version():
    """Try to read GL_VERSION from the system OpenGL driver directly."""
    try:
        opengl32 = ctypes.windll.opengl32
        # We can't call glGetString without a context, but we can check
        # if the DLL loads at all.
        return "opengl32.dll loaded successfully (driver-level DLL present)"
    except Exception as exc:
        return f"opengl32.dll load failed: {exc}"

step("OpenGL driver DLL", _query_opengl_version)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — Pure imports (no rendering context)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 1: Pure imports (should NOT trigger OpenGL)")

step("import numpy",       lambda: __import__("numpy"))
step("import cv2",         lambda: __import__("cv2"))
step("import open3d",      lambda: __import__("open3d"))
step("import pyk4a",       lambda: __import__("pyk4a"))

# Check Open3D version — some versions init Filament eagerly
def _o3d_version():
    import open3d as o3d
    return f"Open3D {o3d.__version__}"
step("open3d version", _o3d_version)


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — Open3D geometry only (no rendering)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 2: Open3D geometry (no rendering, no GUI)")

def _o3d_geometry():
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    pcd = pcd.voxel_down_sample(0.05)
    return f"PointCloud with {len(pcd.points)} points after voxel downsample"

step("create + voxel_down_sample PointCloud", _o3d_geometry)

def _o3d_bbox_crop():
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(500, 3))
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([0.2, 0.2, 0.2]),
        max_bound=np.array([0.8, 0.8, 0.8]),
    )
    cropped = pcd.crop(bbox)
    return f"Cropped to {len(cropped.points)} points"

step("AxisAlignedBoundingBox crop", _o3d_bbox_crop)

def _o3d_dbscan():
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(200, 3))
    labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=5))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return f"DBSCAN found {n_clusters} clusters"

step("cluster_dbscan", _o3d_dbscan)

def _o3d_io_write_read():
    import open3d as o3d
    import numpy as np
    import tempfile
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(50, 3))
    tmp = os.path.join(tempfile.gettempdir(), "_debug_o3d_test.ply")
    o3d.io.write_point_cloud(tmp, pcd)
    pcd2 = o3d.io.read_point_cloud(tmp)
    os.remove(tmp)
    return f"Write/read PLY OK ({len(pcd2.points)} points round-tripped)"

step("write + read PLY file", _o3d_io_write_read)


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — OpenGL / Filament context (this is the suspected failure)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 3: Open3D rendering context (triggers Filament / OpenGL)")
print("  ⚠  If the FEngine error appears, it will be in THIS phase.\n")

def _o3d_vis_basic():
    """Create an offscreen Visualizer — triggers OpenGL context."""
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=64, height=64)
    vis.destroy_window()
    return "Offscreen Visualizer created + destroyed OK"

step("Visualizer (offscreen, visible=False)", _o3d_vis_basic)

def _o3d_gui_init():
    """Initialize the Open3D GUI Application — triggers Filament."""
    import open3d.visualization.gui as gui
    app = gui.Application.instance
    app.initialize()
    return "gui.Application initialized OK"

step("gui.Application.initialize()", _o3d_gui_init)

def _o3d_rendering():
    """Import the rendering module — may trigger Filament on some versions."""
    import open3d.visualization.rendering as rendering
    return f"rendering module loaded: {rendering.__name__}"

step("open3d.visualization.rendering import", _o3d_rendering)


# ═══════════════════════════════════════════════════════════════════
# PHASE 4 — tkinter (used by ArmSegmentation for screen size)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 4: tkinter screen resolution detection")

def _tkinter_screen():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return f"Screen resolution: {w}x{h}"

step("tkinter screen size", _tkinter_screen)


# ═══════════════════════════════════════════════════════════════════
# PHASE 5 — pyk4a MKV playback (quick open/close, no real file needed)
# ═══════════════════════════════════════════════════════════════════
banner("Phase 5: pyk4a availability")

def _pyk4a_check():
    from pyk4a import PyK4APlayback
    return "PyK4APlayback importable — Azure Kinect SDK binding OK"

step("pyk4a.PyK4APlayback import", _pyk4a_check)


# ═══════════════════════════════════════════════════════════════════
# PHASE 6 — preprocessing package barrel imports
# ═══════════════════════════════════════════════════════════════════
banner("Phase 6: Preprocessing package imports")

step(
    "preprocessing.common (KinectMKV, KinectFrame, PointCloudDataHandler)",
    lambda: __import__(
        "preprocessing.common",
        fromlist=["KinectMKV", "KinectFrame", "PointCloudDataHandler"],
    ),
)

step(
    "preprocessing.forearm_extraction (barrel import)",
    lambda: __import__(
        "preprocessing.forearm_extraction",
        fromlist=["ForearmFrameParametersFileHandler", "ArmSegmentation"],
    ),
)

step(
    "preprocessing.forearm_extraction.depth_averaging",
    lambda: __import__(
        "preprocessing.forearm_extraction.depth_averaging",
        fromlist=["FrameDepthAverager"],
    ),
)


# ═══════════════════════════════════════════════════════════════════
# PHASE 7 — Workaround hints
# ═══════════════════════════════════════════════════════════════════
banner("Phase 7: Environment variable hints for workarounds")

print("""
  If Phase 3 failed, try these workarounds on the other machine:

  1. UPDATE GPU DRIVERS — most common fix.

  2. If running via Remote Desktop (RDP), that's the problem.
     RDP replaces the GPU with a software renderer (OpenGL 1.1).
     → Connect via the physical console, or use a VNC-like tool
       that preserves the real GPU (e.g., AnyDesk, Parsec).

  3. Force Open3D to use a CPU software renderer:
       set OPEN3D_CPU_RENDERING=true
     Then re-run this script to see if it helps.

  4. Set Mesa/software OpenGL (if Mesa is installed):
       set LIBGL_ALWAYS_SOFTWARE=1
       set MESA_GL_VERSION_OVERRIDE=4.1

  5. If interactive GUI is not needed, the script can be patched
     to skip Open3D visualization entirely (geometry-only mode).
""")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
banner("Done — review output above")
print("  The FEngine/wglCreateContextAttribs error should appear in Phase 3.")
print("  If Phases 0-2 and 4-6 pass, the core pipeline can work if")
print("  visualization is disabled or an OpenGL workaround is applied.")
