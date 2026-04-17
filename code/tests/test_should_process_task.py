"""Unit tests for refresh_output_mtimes() in utils.should_process_task."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# utils/__init__.py is stubbed out in conftest.py; import the module directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Ensure the stub created by conftest covers utils.should_process_task so that
# the direct module import below works without the heavy utils __init__.
_mod_name = "utils.should_process_task"
if _mod_name not in sys.modules:
    _mod_path = _SRC / "utils" / "should_process_task.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location(_mod_name, _mod_path)
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[_mod_name] = _mod
    spec.loader.exec_module(_mod)

from utils.should_process_task import refresh_output_mtimes  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mtime(p: Path) -> float:
    return p.stat().st_mtime


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefreshOutputMtimes:
    """Tests for refresh_output_mtimes()."""

    def test_single_path_updates_mtime(self, tmp_path: Path) -> None:
        """A single existing Path has its mtime brought forward after touch."""
        f = tmp_path / "output.bin"
        f.write_bytes(b"data")

        # Force the stored mtime to be clearly in the past.
        past = time.time() - 10
        import os
        os.utime(f, (past, past))

        before = _mtime(f)
        time.sleep(0.01)  # ensure clock advances
        refresh_output_mtimes(f)
        after = _mtime(f)

        assert after > before, "mtime was not refreshed for single Path"

    def test_list_of_paths_updates_all_mtimes(self, tmp_path: Path) -> None:
        """All paths in a list have their mtimes brought forward."""
        files = [tmp_path / f"out_{i}.bin" for i in range(3)]
        for f in files:
            f.write_bytes(b"x")

        import os
        past = time.time() - 10
        for f in files:
            os.utime(f, (past, past))

        befores = [_mtime(f) for f in files]
        time.sleep(0.01)
        refresh_output_mtimes(files)
        afters = [_mtime(f) for f in files]

        for i, (before, after) in enumerate(zip(befores, afters)):
            assert after > before, f"mtime not refreshed for files[{i}]"

    def test_missing_path_is_noop_no_raise(self, tmp_path: Path) -> None:
        """A path that does not exist is silently skipped — no exception raised."""
        missing = tmp_path / "does_not_exist.bin"
        assert not missing.exists()

        # Must not raise.
        refresh_output_mtimes(missing)

    def test_missing_path_in_list_is_noop_no_raise(self, tmp_path: Path) -> None:
        """Missing paths inside a list are silently skipped; existing ones are still touched."""
        existing = tmp_path / "exists.bin"
        existing.write_bytes(b"y")
        missing = tmp_path / "absent.bin"

        import os
        past = time.time() - 10
        os.utime(existing, (past, past))
        before = _mtime(existing)

        time.sleep(0.01)
        refresh_output_mtimes([existing, missing])  # must not raise

        assert _mtime(existing) > before, "existing file mtime not refreshed when list contains a missing path"

    def test_readonly_path_logs_and_continues_no_raise(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A PermissionError on touch() triggers a log message and does not raise.

        We use mock.patch to inject a PermissionError because on Windows,
        chmod(S_IREAD) does not reliably block Path.touch() as it does on Unix.
        """
        f = tmp_path / "readonly.bin"
        f.write_bytes(b"z")

        with patch.object(Path, "touch", side_effect=PermissionError("access denied")):
            refresh_output_mtimes(f)  # must not raise

        captured = capsys.readouterr()
        assert "permission denied" in captured.out.lower(), (
            "Expected a permission-denied log line; got: " + repr(captured.out)
        )
