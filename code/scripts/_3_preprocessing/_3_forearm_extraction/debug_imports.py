"""
Diagnostic script: bisect imports to find which library triggers the OpenGL error.

Run:  python code/scripts/_3_preprocessing/_3_forearm_extraction/debug_imports.py

The FEngine/OpenGL error is suspected to come from Open3D's Filament renderer,
which initialises an OpenGL context at import time.
"""

import sys
import traceback


def try_import(label, import_func):
    print(f"\n{'='*60}")
    print(f"  Importing: {label}")
    print(f"{'='*60}")
    try:
        import_func()
        print(f"  OK — {label} imported without error")
    except Exception:
        print(f"  FAILED — {label}")
        traceback.print_exc()


# ── Individual libraries (most-likely culprit first) ──

try_import("open3d (base)", lambda: __import__("open3d"))

try_import("open3d.visualization.gui", lambda: __import__("open3d.visualization.gui"))

try_import("open3d.visualization.rendering", lambda: __import__("open3d.visualization.rendering"))

try_import("pyvista", lambda: __import__("pyvista"))

try_import("pyvistaqt", lambda: __import__("pyvistaqt"))

try_import("cv2", lambda: __import__("cv2"))

# ── Barrel imports from preprocessing packages ──

print(f"\n{'#'*60}")
print(f"  Now testing preprocessing barrel imports")
print(f"{'#'*60}")

try_import(
    "preprocessing.common (KinectMKV, KinectFrame, PointCloudDataHandler)",
    lambda: __import__(
        "preprocessing.common",
        fromlist=["KinectMKV", "KinectFrame", "PointCloudDataHandler"],
    ),
)

try_import(
    "preprocessing.forearm_extraction (full barrel import)",
    lambda: __import__(
        "preprocessing.forearm_extraction",
        fromlist=["*"],
    ),
)

print(f"\n{'='*60}")
print("  Done — review output above for the FEngine/OpenGL error")
print(f"{'='*60}")
