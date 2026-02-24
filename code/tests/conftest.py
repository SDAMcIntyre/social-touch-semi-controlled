"""conftest.py — package stubs for unit-test environments.

``preprocessing/stickers_analysis/__init__.py`` eagerly imports many
modules that depend on hardware SDKs (pyk4a, open3d, …) which are not
available in a plain unit-test environment.

Injecting a minimal stub for that package into ``sys.modules`` before
any test module is collected prevents the heavy ``__init__.py`` from
running, while still allowing the genuine sub-package files (roi, xyz,
common, …) to load normally via the filesystem.

``preprocessing/__init__.py`` is empty so it does not need a stub.
"""

import sys
import types
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"


def _stub_package(dotted: str) -> None:
    """Inject a bare stub package into sys.modules, skipping its __init__.py.

    The stub's ``__path__`` is set to the real source directory so Python's
    importer can still discover and load genuine sub-packages and modules
    beneath it.
    """
    if dotted in sys.modules:
        return
    pkg_dir = _SRC / Path(*dotted.split("."))
    mod = types.ModuleType(dotted)
    mod.__path__ = [str(pkg_dir)]
    mod.__package__ = dotted
    sys.modules[dotted] = mod


# Stub the one heavyweight package root whose __init__.py pulls in
# pyk4a / open3d / PyQt5 and other SDK dependencies.
_stub_package("preprocessing.stickers_analysis")

# utils/__init__.py eagerly imports PipelineMonitor, DagConfigHandler,
# TaskExecutor, and signal-processing helpers that pull in openpyxl,
# PyYAML, and other optional deps.  Stub the root so individual modules
# (e.g. utils.pipeline.dag_config_model) can still be imported directly.
_stub_package("utils")
_stub_package("utils.pipeline")
_stub_package("utils.pipeline.monitoring")
