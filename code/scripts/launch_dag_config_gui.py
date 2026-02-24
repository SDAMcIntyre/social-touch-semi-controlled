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

from PyQt5.QtWidgets import QApplication

from utils.gui.dag_launcher.launcher_window import LauncherWindow


def main() -> None:
    configs_dir = Path("configs")
    if not configs_dir.is_dir():
        print(f"Error: configs directory not found at {configs_dir.resolve()}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = LauncherWindow(configs_dir)
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
