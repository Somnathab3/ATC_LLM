import sys
from pathlib import Path
import types
import builtins
import importlib

import pytest

# Import the CLI module under test
import cli as cli_mod


# ---------- helpers ----------

def _parse_and_run(parser, argv):
    """Helper to parse args and call the bound function."""
    args = parser.parse_args(argv)
    assert hasattr(args, "func"), f"No func attached for argv={argv}"
    return args.func(args)


# ---------- unit tests for utilities ----------

def test_setup_logging_levels(caplog):
    cli_mod.setup_logging(verbose=False)
    assert caplog.text == ""  # logger configured, but no log yet
    cli_mod.setup_logging(verbose=True)  # should not error


def test_argv_context_restores_sys_argv():
    original = list(sys.argv)
    new = ["prog", "verify-llm"]
    with cli_mod.argv_context(new):
        assert sys.argv == new
    assert sys.argv == original


def test_validate_path_exists(tmp_path):
    missing = tmp_path / "nope"
    assert cli_mod.validate_path_exists(str(missing), "data") is False
    present = tmp_path
    assert cli_mod.validate_path_exists(str(present), "dir") is True


# ---------- safe_import behavior (monkeypatched) ----------

@pytest.fixture(autouse=True)
def patch_safe_import(monkeypatch, dummy_main_factory):
    """
    Route all safe_import calls to stubs matching module_name.
    Individual tests can override specific modules if they need custom behavior.
    """
    def _stub(module_name: str, description: str = "module"):
        # Provide a default main for any known module name the CLI can request
        known = {
            "repo_healthcheck":  dummy_main_factory(returncode=0)[0],
            "complete_llm_demo": dummy_main_factory(returncode=0)[0],
            "batch_scat_llm_processor": dummy_main_factory(returncode=0)[0],
            "production_batch_processor": dummy_main_factory(returncode=0)[0],
            "demo_baseline_vs_llm": dummy_main_factory(returncode=0)[0],
            "verify_llm_communication": dummy_main_factory(returncode=0)[0],
            "visualize_conflicts": dummy_main_factory(returncode=0)[0],
        }
        if module_name in known:
            return known[module_name]
        return None
    monkeypatch.setattr(cli_mod, "safe_import", _stub)
    return _stub


# ---------- parser wiring ----------

def test_create_parser_has_all_subcommands():
    p = cli_mod.create_parser()
    # Smoke-check help builds without error
    p.format_help()
    # Ensure top-level commands exist by parsing their help
    for argv in [
        ["health-check"],
        ["simulate", "basic"],
        ["simulate", "scat", "--scat-dir", "X"],
        ["batch", "production"],
        ["compare"],
        ["test"],
        ["server"],
        ["verify-llm"],
        ["visualize", "--data-file", "X"],
    ]:
        try:
            p.parse_args(argv)
        except SystemExit:
            pytest.fail(f"Argparse failed for {argv}")


# ---------- command: health-check ----------

def test_health_check_success():
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["-v", "health-check"])
    assert rc == 0


# ---------- command: simulate basic ----------

def test_simulate_basic_invokes_demo(monkeypatch, dummy_main_factory):
    main, calls = dummy_main_factory(returncode=0)
    monkeypatch.setattr(cli_mod, "safe_import", lambda m, d="": main if m=="complete_llm_demo" else None)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["simulate", "basic", "--aircraft", "7", "--duration", "15", "--llm-model", "llama3.1:8b"])
    assert rc == 0
    assert calls["count"] == 1


# ---------- command: simulate scat ----------

def test_simulate_scat_missing_dir_fails():
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["simulate", "scat", "--scat-dir", "does/not/exist"])
    assert rc == 1

def test_simulate_scat_success(monkeypatch, dummy_main_factory, tmp_dir):
    main, calls = dummy_main_factory(returncode=0)
    monkeypatch.setattr(cli_mod, "safe_import", lambda m, d="": main if m=="batch_scat_llm_processor" else None)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, [
        "-v", "simulate", "scat",
        "--scat-dir", str(tmp_dir),
        "--max-flights", "3",
        "--scenarios-per-flight", "2",
        "--output-dir", "Output",
        "--llm-model", "llama3.1:8b"
    ])
    assert rc == 0
    assert calls["count"] == 1
    # argv_context should have set a script-like argv for the delegated main
    assert calls["argv_at_call"][0].endswith("batch_scat_llm_processor.py")


# ---------- command: batch production ----------

def test_batch_production_missing_dir_fails():
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["batch", "production", "--scat-dir", "nope"])
    assert rc == 1

def test_batch_production_success(monkeypatch, dummy_main_factory, tmp_dir):
    main, calls = dummy_main_factory(returncode=0)
    monkeypatch.setattr(cli_mod, "safe_import", lambda m, d="": main if m=="production_batch_processor" else None)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, [
        "-v", "batch", "production",
        "--scat-dir", str(tmp_dir),
        "--max-flights", "10",
        "--scenarios-per-flight", "5",
        "--output-dir", "Output",
        "--skip-checks"
    ])
    assert rc == 0
    assert calls["count"] == 1
    assert calls["argv_at_call"][0].endswith("production_batch_processor.py")
    assert "--skip-checks" in calls["argv_at_call"]


# ---------- command: compare ----------

def test_compare_missing_path_fails():
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["compare", "--scat-path", "nope"])
    assert rc == 1

def test_compare_success(monkeypatch, dummy_main_factory, tmp_dir, tmp_path):
    main, calls = dummy_main_factory(returncode=0)
    monkeypatch.setattr(cli_mod, "safe_import", lambda m, d="": main if m=="demo_baseline_vs_llm" else None)
    out = tmp_path / "out.json"
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, [
        "compare",
        "--scat-path", str(tmp_dir),
        "--max-flights", "4",
        "--time-window", "45",
        "--output", str(out)
    ])
    assert rc == 0
    assert calls["count"] == 1
    assert calls["argv_at_call"][0].endswith("demo_baseline_vs_llm.py")


# ---------- command: test (pytest wrapper) ----------

def test_test_suite_success(monkeypatch):
    class DummyCompleted:
        def __init__(self, returncode): self.returncode = returncode
    def fake_run(cmd, cwd):
        assert "pytest" in cmd
        return DummyCompleted(0)
    monkeypatch.setattr("subprocess.run", fake_run)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["-v", "test", "--coverage"])
    assert rc == 0

def test_test_suite_no_tests_is_ok(monkeypatch):
    class DummyCompleted:
        def __init__(self, returncode): self.returncode = returncode
    def fake_run(cmd, cwd): return DummyCompleted(5)  # pytest: no tests collected
    monkeypatch.setattr("subprocess.run", fake_run)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["-v", "test", "--test-pattern", "nothing*"])
    assert rc == 0  # treated as non-error

def test_test_suite_failure_bubbles_code(monkeypatch):
    class DummyCompleted:
        def __init__(self, returncode): self.returncode = returncode
    def fake_run(cmd, cwd): return DummyCompleted(1)
    monkeypatch.setattr("subprocess.run", fake_run)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["test"])
    assert rc == 1


# ---------- command: server (uvicorn + FastAPI app) ----------

def test_server_success(monkeypatch, inject_module):
    # Fake 'src.api.service:app' module and uvicorn.run
    svc = inject_module("src.api.service")
    svc.app = object()
    runs = {"count": 0, "args": None, "kwargs": None}
    def fake_run(app_path, host, port, reload, log_level):
        runs["count"] += 1
        runs["args"] = (app_path,)
        runs["kwargs"] = dict(host=host, port=port, reload=reload, log_level=log_level)
    uvicorn_mod = inject_module("uvicorn")
    uvicorn_mod.run = fake_run

    p = cli_mod.create_parser()
    # Use debug to exercise reload=True path; verbose to set log_level
    rc = _parse_and_run(p, ["-v", "server", "--host", "0.0.0.0", "--port", "9000", "--debug"])
    # uvicorn.run should have been called once
    assert rc == 0
    assert runs["count"] == 1
    assert runs["args"][0] == "src.api.service:app"
    assert runs["kwargs"]["host"] == "0.0.0.0"
    assert runs["kwargs"]["port"] == 9000
    assert runs["kwargs"]["reload"] is True
    assert runs["kwargs"]["log_level"] == "debug"

def test_server_missing_service_module(monkeypatch):
    # Ensure import fails by not injecting src.api.service
    # but we still need uvicorn import to succeed up to that point
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["server"])
    assert rc == 1


# ---------- command: verify-llm ----------

def test_verify_llm_invokes_script_argv(monkeypatch, dummy_main_factory):
    captured = {"argv": None}
    def side_effect():
        import sys as _s
        captured["argv"] = list(_s.argv)
    main, _ = dummy_main_factory(returncode=0, side_effect=side_effect)
    monkeypatch.setattr(cli_mod, "safe_import", lambda m, d="": main if m=="verify_llm_communication" else None)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["-v", "verify-llm", "--model", "llama3.1:8b"])
    assert rc == 0
    assert captured["argv"][0].endswith("verify_llm_communication.py")


# ---------- command: visualize ----------

def test_visualize_missing_file_fails():
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["visualize", "--data-file", "nope.json"])
    assert rc == 1

def test_visualize_success(monkeypatch, dummy_main_factory, tmp_file):
    main, calls = dummy_main_factory(returncode=0)
    monkeypatch.setattr(cli_mod, "safe_import", lambda m, d="": main if m=="visualize_conflicts" else None)
    p = cli_mod.create_parser()
    rc = _parse_and_run(p, ["-v", "visualize", "--data-file", str(tmp_file)])
    assert rc == 0
    assert calls["count"] == 1
    assert calls["argv_at_call"][0].endswith("visualize_conflicts.py")
    assert "--data-file" in calls["argv_at_call"]
    assert str(tmp_file) in calls["argv_at_call"]
