"""Entry point for the DAG Config Launcher GUI.

Launch with::

    python code/scripts/launch_dag_config_gui.py
"""

import sys
from pathlib import Path

# CuPy import guard — must precede any preprocessing imports (project convention)
try:
    import cupy  # noqa: F401
except Exception:
    pass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from utils.gui.dag_launcher.launcher_config import parse_launcher_config
from utils.gui.dag_launcher.launcher_window import LauncherWindow

# High-DPI awareness — must be set BEFORE QApplication is constructed, otherwise
# showMaximized() on Windows with display scaling renders a ~half-screen window
# with a title bar drawn at the unscaled size.
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def main() -> None:
    project_root = Path(".").resolve()
    launcher_yaml = project_root / "configs" / "launcher.yaml"

    if not launcher_yaml.is_file():
        print(f"Error: launcher config not found at {launcher_yaml}")
        sys.exit(1)

    try:
        entries = parse_launcher_config(launcher_yaml, project_root)
    except Exception as exc:
        print(f"Error: failed to parse {launcher_yaml}: {exc}")
        sys.exit(1)

    configs_dir = project_root / "configs"
    app = QApplication(sys.argv)
    window = LauncherWindow(entries, configs_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
