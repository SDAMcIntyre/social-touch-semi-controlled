# Open3D Filament/OpenGL context creation failure on Windows

**Status:** Superseded — see
[issue-open3d-filament-opengl-crash.md](issue-open3d-filament-opengl-crash.md)
for the consolidated investigation.

---

## Summary

This note originally documented the Filament crash on a secondary machine
(RTX 2070 Super, driver 560.94). The same crash now occurs on the primary
development machine (RTX 4070 Ti SUPER, driver 610.47).

Root cause: NVIDIA drivers 560.94 and 610.47 both reject the OpenGL
context attributes that Open3D's embedded Filament renderer requests via
`wglCreateContextAttribs()`. Driver 591.86 was the last known working
version.

The legacy GLFW-based `Visualizer` (`wglCreateContext`) works on all
machines. Only the Filament GUI path is broken.

See the consolidated document for the full list of attempted fixes and
the planned fallback approach.
