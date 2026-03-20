# Open3D Filament/OpenGL context creation failure on Windows

## Problem

Running `extract_participant_forearm.py` on a second Windows computer fails with:

```
FEngine (64 bits) created at ... (threading is enabled)
FEngine resolved backend: OpenGL
wglCreateContextAttribs() failed, whdc=...
Windows error code: 87. (null). Investigate
```

The script works fine on the primary development machine.

## Environment comparison

|                  | Working machine     | Failing machine     |
|------------------|---------------------|---------------------|
| Open3D           | 0.19.0              | 0.19.0              |
| NVIDIA driver    | 591.86              | 560.94              |
| GPU              | —                   | RTX 2070 Super      |
| CPU              | —                   | AMD Ryzen 5 3600    |
| Integrated GPU   | —                   | None                |

## Analysis

- Error 87 = `ERROR_INVALID_PARAMETER` — the OpenGL context attributes requested by Open3D's Filament rendering engine are rejected by driver 560.94.
- The RTX 2070 Super hardware supports OpenGL fine; this is a driver-level incompatibility.
- Same Open3D version on both machines, so the driver version gap (560 vs 591) is the likely cause.

## What was tried

1. **`OPEN3D_CPU_RENDERING=true` via VS Code launch.json** — did not resolve the issue. This environment variable may not be supported by Open3D 0.19.0 (needs verification).

## Still to investigate

- Verify which environment variables Open3D 0.19 actually supports for rendering backend control.
- Whether updating the NVIDIA driver from 560.94 to 591+ resolves the issue.
- Whether forcing Vulkan backend instead of OpenGL is possible.
- Whether downgrading Open3D to 0.18.x (which may use a different rendering backend) helps.
- Check Open3D GitHub issues for this specific `wglCreateContextAttribs` error on Windows.

## Affected script

- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`
- Uses `ArmSegmentation` (interactive mode) which requires Open3D's Filament-based GUI via `rendering.Open3DScene`.
