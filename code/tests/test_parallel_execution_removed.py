"""Guard tests for the removed ``parallel_execution`` batch path.

Phase 2 of ``docs/development/plans/active/remove-prefect-orchestration.md``
deleted the dead parallel branches from the four batch runners.  The
``parallel_execution`` YAML key is still *read* so existing DAG configs stay
valid, but setting it to ``true`` must now fail loudly instead of silently
processing zero sessions (the pre-change postprocess behaviour) or crashing
with ``AttributeError`` (the pre-change preprocess/primary behaviour).

Two layers of assertion:

1. :func:`test_parallel_guard_is_first_statement` parses the sources with
   ``ast``.  It needs no third-party dependency and therefore runs in the
   stubbed unit-test environment described in ``conftest.py``.
2. :func:`test_parallel_true_raises_not_implemented` actually calls each batch
   runner with ``parallel=True``.  The four entry scripts import the real
   ``utils`` / ``primary_processing`` packages, which ``conftest.py``
   deliberately stubs out, so this must run in a clean child interpreter.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "code" / "scripts"
SRC_DIR = REPO_ROOT / "code" / "src"

PLAN_REFERENCE = "docs/development/plans/active/remove-prefect-orchestration.md"

# (module name, batch-runner function name)
BATCH_RUNNERS = [
    ("merging_pipeline_neuron_to_kinect_auto", "run_batch_processing"),
    ("preprocess_workflow_kinect_auto", "run_batch_processing"),
    ("primary_workflow_kinect_auto", "run_batch_primary"),
    ("postprocess_workflow_kinect_auto", "run_batch_postprocessing"),
]

_PROBE = r'''
import inspect, json, sys, traceback

TARGETS = %(targets)r

out = {}
for mod_name, fn_name in TARGETS:
    try:
        mod = __import__(mod_name)
        fn = getattr(mod, fn_name)
        fn = getattr(fn, "fn", fn)  # unwrap a prefect @flow if one is still present
        kwargs = {
            "block_files": [],
            "project_data_root": None,
            "dag_config_path": None,
            "parallel": True,
        }
        params = inspect.signature(fn).parameters
        for extra in ("monitor_queue", "report_file_path"):
            if extra in params:
                kwargs[extra] = None
        try:
            fn(**kwargs)
            out[mod_name] = {"raised": None}
        except NotImplementedError as exc:
            out[mod_name] = {"raised": "NotImplementedError", "message": str(exc)}
        except BaseException as exc:
            out[mod_name] = {"raised": type(exc).__name__, "message": str(exc)}
    except BaseException:
        out[mod_name] = {"import_error": traceback.format_exc(limit=3)}

sys.stdout.write("<<<JSON>>>" + json.dumps(out) + "<<<END>>>")
'''


def _find_function(module_path: Path, func_name: str) -> ast.FunctionDef:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"{func_name} not found in {module_path}")


@pytest.mark.parametrize("module_name, func_name", BATCH_RUNNERS)
def test_parallel_guard_is_first_statement(module_name: str, func_name: str) -> None:
    """Each batch runner rejects ``parallel=True`` before doing any work."""
    func = _find_function(SCRIPTS_DIR / f"{module_name}.py", func_name)

    assert "parallel" in {a.arg for a in func.args.args}, (
        f"{module_name}.{func_name} must keep reading the parallel_execution "
        "parameter so existing DAG configs stay valid"
    )

    body = list(func.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # skip the docstring

    assert body, f"{module_name}.{func_name} has an empty body"
    guard = body[0]
    assert isinstance(guard, ast.If), (
        f"the first statement of {module_name}.{func_name} must be the "
        "parallel_execution guard, so nothing is processed before it fires"
    )
    assert isinstance(guard.test, ast.Name) and guard.test.id == "parallel"
    assert not guard.orelse, "the guard must not have an else branch"
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)

    raised = guard.body[0].exc
    assert isinstance(raised, ast.Call)
    assert isinstance(raised.func, ast.Name) and raised.func.id == "NotImplementedError"

    message = " ".join(
        arg.value for arg in raised.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    )
    assert "parallel_execution" in message
    assert PLAN_REFERENCE in message, (
        "the error message must point at the plan that removed the feature"
    )


@pytest.fixture(scope="module")
def parallel_guard_probe():
    """Call all four batch runners with ``parallel=True`` in a clean interpreter."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), str(SCRIPTS_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # The entry scripts print emoji banners at import time; force UTF-8 both
    # ways so decoding the child's output cannot fail on a cp1252 console.
    env["PYTHONIOENCODING"] = "utf-8"
    code = _PROBE % {"targets": BATCH_RUNNERS}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    stdout = completed.stdout or ""
    if "<<<JSON>>>" not in stdout:
        pytest.fail(
            "probe interpreter produced no result\n"
            f"exit={completed.returncode}\nstdout={stdout}\nstderr={completed.stderr}"
        )
    payload = stdout.split("<<<JSON>>>", 1)[1].split("<<<END>>>", 1)[0]
    return json.loads(payload)


@pytest.mark.parametrize("module_name, func_name", BATCH_RUNNERS)
def test_parallel_true_raises_not_implemented(
    parallel_guard_probe, module_name: str, func_name: str
) -> None:
    result = parallel_guard_probe[module_name]

    if "import_error" in result:
        pytest.skip(
            f"{module_name} could not be imported in this environment "
            f"(heavy SDK dependency missing):\n{result['import_error']}"
        )

    assert result["raised"] == "NotImplementedError", (
        f"{module_name}.{func_name}(parallel=True) raised {result['raised']!r} "
        f"instead of NotImplementedError: {result.get('message')}"
    )
    assert "parallel_execution" in result["message"]
    assert PLAN_REFERENCE in result["message"]
