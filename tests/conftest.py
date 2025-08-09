import types
import sys
from pathlib import Path
import pytest

@pytest.fixture
def dummy_main_factory():
    """
    Returns a factory to create stub 'main' callables that record calls.
    Usage:
        main, calls = dummy_main_factory(returncode=0)
        main(); assert calls["count"] == 1
    """
    def _factory(returncode=0, side_effect=None):
        calls = {"count": 0, "argv_at_call": None}
        def _main():
            calls["count"] += 1
            import sys as _sys
            calls["argv_at_call"] = list(_sys.argv)
            if side_effect:
                side_effect()
            return returncode
        return _main, calls
    return _factory

@pytest.fixture
def inject_module(monkeypatch):
    """
    Insert a dummy module into sys.modules under a given name.
    Returns the module so tests can populate attributes.
    """
    def _inject(name: str):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod
    return _inject

@pytest.fixture
def tmp_file(tmp_path: Path):
    f = tmp_path / "data.json"
    f.write_text('{"ok": true}')
    return f

@pytest.fixture
def tmp_dir(tmp_path: Path):
    d = tmp_path / "scat_dir"
    d.mkdir()
    return d
