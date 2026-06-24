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


# Stub heavyweight package roots whose __init__.py pull in
# pyk4a / open3d / PyQt5 and other SDK dependencies.
_stub_package("preprocessing.stickers_analysis")

# forearm_extraction/__init__.py imports open3d (normals_estimation),
# PyQt5 (curation GUI), and pyk4a transitively.  Stub the root so the
# model and data-access sub-packages can still be imported directly.
_stub_package("preprocessing.forearm_extraction")

# utils/__init__.py eagerly imports PipelineMonitor, DagConfigHandler,
# TaskExecutor, and signal-processing helpers that pull in openpyxl,
# PyYAML, and other optional deps.  Stub the root so individual modules
# (e.g. utils.pipeline.dag_config_model) can still be imported directly.
_stub_package("utils")
_stub_package("utils.pipeline")
_stub_package("utils.pipeline.monitoring")

# analysis/pipeline/__init__.py imports session_discovery (requires
# primary_processing / KinectConfig) and stage_runner (requires utils).
# Neither is available in the unit-test environment.  Stub the package so
# individual modules (e.g. analysis.pipeline.shared_constants) can still
# be imported directly without running the heavyweight __init__.
_stub_package("analysis.pipeline")
