#!/usr/bin/env python3
"""
Pytest tests for CLI modernization sanity checks.
These tests prevent regressions in the unified CLI interface.
"""

import subprocess
import sys
import pytest
import json
import os


def test_console_entry_point_help():
    """Test that atc-llm --help shows unified subcommands."""
    result = subprocess.run([
        "atc-llm", "--help"
    ], capture_output=True, text=True, shell=True)
    
    assert result.returncode == 0, f"atc-llm --help failed: {result.stderr}"
    
    # Check for main command structure
    assert "ATC LLM-BlueSky CDR System" in result.stdout
    assert "Unified Command Line Interface" in result.stdout
    
    # Check for all expected subcommands
    expected_subcommands = [
        "health-check",
        "simulate", 
        "batch",
        "metrics",
        "report",
        "verify-llm",
        "visualize"
    ]
    
    for cmd in expected_subcommands:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"


def test_shim_deprecation_warnings():
    """Test that all shim scripts show deprecation warnings."""
    shim_scripts = [
        "bin/production_batch_processor.py",
        "bin/batch_scat_llm_processor.py",
        "bin/scat_baseline_cli.py", 
        "bin/verify_llm_communication.py",
        "bin/visualize_conflicts.py",
        "bin/repo_healthcheck.py"
    ]
    
    for script in shim_scripts:
        result = subprocess.run([
            sys.executable, script, "--help"
        ], capture_output=True, text=True)
        
        # Should show deprecation warning in stderr
        warning_shown = (
            "DeprecationWarning" in result.stdout or 
            "DeprecationWarning" in result.stderr or
            "DEPRECATED" in result.stdout
        )
        
        assert warning_shown, f"Shim {script} missing deprecation warning"
        assert result.returncode == 0, f"Shim {script} should still work: {result.stderr}"


def test_dump_config_functionality():
    """Test that --dump-config returns valid JSON configuration."""
    result = subprocess.run([
        "atc-llm", "--dump-config"
    ], capture_output=True, text=True, shell=True)
    
    assert result.returncode == 0, f"--dump-config failed: {result.stderr}"
    
    try:
        config = json.loads(result.stdout)
        
        # Check for expected configuration keys
        expected_keys = [
            "llm_model_name",
            "llm_enabled", 
            "lookahead_time_min",
            "min_horizontal_separation_nm",
            "min_vertical_separation_ft",
            "ollama_base_url"
        ]
        
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"
            
    except json.JSONDecodeError as e:
        pytest.fail(f"--dump-config returned invalid JSON: {e}")


def test_environment_variable_override():
    """Test that environment variables override configuration."""
    test_value = "test-model-override"
    
    env = os.environ.copy()
    env["ATC_LLM_LLM_MODEL_NAME"] = test_value
    
    result = subprocess.run([
        "atc-llm", "--dump-config"
    ], capture_output=True, text=True, env=env, shell=True)
    
    assert result.returncode == 0, f"Environment override test failed: {result.stderr}"
    
    try:
        config = json.loads(result.stdout)
        assert config["llm_model_name"] == test_value, \
            f"Environment override failed. Expected {test_value}, got {config['llm_model_name']}"
    except json.JSONDecodeError as e:
        pytest.fail(f"Environment override test returned invalid JSON: {e}")


def test_arg_passthrough():
    """Test that shim arguments pass through correctly."""
    # Test a shim that should accept arguments
    result = subprocess.run([
        sys.executable, "bin/scat_baseline_cli.py", "--duration-min", "45", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, "Argument passthrough failed"
    
    # Should show the simulate subcommand help (not main help)
    assert "simulate" in result.stdout
    assert "--duration-min" in result.stdout
    assert "--dt-min" in result.stdout


def test_no_duplicate_entry_points():
    """Test that there aren't competing CLI entry points."""
    # Check that we don't have bin/atc-llm.py or similar
    import pathlib
    bin_dir = pathlib.Path("bin")
    
    if bin_dir.exists():
        conflicting_files = [
            bin_dir / "atc-llm.py",
            bin_dir / "atc_llm.py", 
            bin_dir / "main.py"
        ]
        
        for file_path in conflicting_files:
            assert not file_path.exists(), f"Found conflicting CLI entry point: {file_path}"


if __name__ == "__main__":
    # Run the tests directly if called as script
    pytest.main([__file__, "-v"])


def test_cli_subcommands_smoke():
    """Smoke test for all CLI subcommands to ensure they respond to --help."""
    subcommands = [
        "health-check",
        "simulate", 
        "batch",
        "metrics",
        "report",
        "verify-llm",
        "visualize"
    ]
    
    for cmd in subcommands:
        result = subprocess.run([
            "atc-llm", cmd, "--help"
        ], capture_output=True, text=True, shell=True)
        
        assert result.returncode == 0, f"Subcommand {cmd} --help failed: {result.stderr}"
        assert "usage:" in result.stdout.lower(), f"Subcommand {cmd} missing usage info"


def test_shim_delegation_integration():
    """Test that shims delegate correctly to CLI subcommands."""
    test_cases = [
        ("bin/production_batch_processor.py", "batch"),
        ("bin/scat_baseline_cli.py", "simulate"),
        ("bin/verify_llm_communication.py", "verify-llm"),
        ("bin/visualize_conflicts.py", "visualize"),
        ("bin/repo_healthcheck.py", "health-check"),
    ]
    
    for shim_script, expected_subcommand in test_cases:
        # Test that shim responds to --help
        result = subprocess.run([
            sys.executable, shim_script, "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Shim {shim_script} failed"
        
        # Should contain usage line with expected subcommand
        assert f"atc-llm {expected_subcommand}" in result.stdout, \
            f"Shim {shim_script} not delegating to {expected_subcommand}"
