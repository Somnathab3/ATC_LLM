#!/usr/bin/env python3
"""
Test CLI modernization tasks 2-5:
- Task 2: Legacy script shimming with deprecation warnings
- Task 3: Argparse consolidation 
- Task 4: Centralized configuration
- Task 5: Console entry point

Run with: python test_cli_modernization.py
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def test_shim_deprecation_warnings():
    """Test that legacy shim scripts show deprecation warnings"""
    print("Testing shim deprecation warnings...")
    
    shim_scripts = [
        "bin/production_batch_processor.py",
        "bin/batch_scat_llm_processor.py", 
        "bin/scat_baseline_cli.py",
        "bin/verify_llm_communication.py",
        "bin/visualize_conflicts.py",
        "bin/repo_healthcheck.py"
    ]
    
    for script in shim_scripts:
        if Path(script).exists():
            print(f"  Testing {script}...")
            result = subprocess.run([
                sys.executable, script, "--help"
            ], capture_output=True, text=True)
            
            if "DeprecationWarning" in result.stdout or "DeprecationWarning" in result.stderr:
                print(f"    ✅ Shows deprecation warning")
            else:
                print(f"    ❌ Missing deprecation warning")
                print(f"    Stdout: {result.stdout[:100]}...")
                print(f"    Stderr: {result.stderr[:100]}...")
        else:
            print(f"  ⚠️  {script} not found")

def test_argparse_consolidation():
    """Test that cli.py has consolidated argparse with --dump-config"""
    print("\nTesting argparse consolidation...")
    
    # Test --dump-config option
    result = subprocess.run([
        sys.executable, "cli.py", "--dump-config"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            config = json.loads(result.stdout)
            print("  ✅ --dump-config works and returns valid JSON")
            print(f"    Sample config keys: {list(config.keys())[:5]}")
        except json.JSONDecodeError:
            print("  ❌ --dump-config returns invalid JSON")
            print(f"    Output: {result.stdout[:200]}")
    else:
        print(f"  ❌ --dump-config failed: {result.stderr}")

def test_centralized_configuration():
    """Test that configuration supports environment variables"""
    print("\nTesting centralized configuration...")
    
    # Set an environment variable and test override
    env = os.environ.copy()
    env["ATC_LLM_LLM_MODEL_NAME"] = "test-model-from-env"
    
    result = subprocess.run([
        sys.executable, "cli.py", "--dump-config"
    ], capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        try:
            config = json.loads(result.stdout)
            if config.get("llm_model_name") == "test-model-from-env":
                print("  ✅ Environment variable override works")
            else:
                print(f"  ❌ Environment override failed. Got: {config.get('llm_model_name')}")
        except json.JSONDecodeError:
            print("  ❌ Invalid JSON from --dump-config")
    else:
        print(f"  ❌ Configuration test failed: {result.stderr}")

def test_console_entry_point():
    """Test that console entry point 'atc-llm' works"""
    print("\nTesting console entry point...")
    
    result = subprocess.run([
        "atc-llm", "--help"
    ], capture_output=True, text=True, shell=True)
    
    if result.returncode == 0:
        if "ATC LLM-BlueSky CDR System" in result.stdout:
            print("  ✅ Console entry point 'atc-llm' works")
        else:
            print("  ❌ Console entry point returns unexpected output")
            print(f"    Output: {result.stdout[:200]}")
    else:
        print(f"  ❌ Console entry point failed: {result.stderr}")
        print("    Note: Run 'pip install -e .' if not installed")

def main():
    """Run all CLI modernization tests"""
    print("=== CLI Modernization Test Suite ===")
    print("Testing Tasks 2-5 implementation...")
    
    test_shim_deprecation_warnings()
    test_argparse_consolidation() 
    test_centralized_configuration()
    test_console_entry_point()
    
    print("\n=== Test Summary ===")
    print("Tasks 2-5 validation complete.")
    print("✅ = Working correctly")
    print("❌ = Needs attention")
    print("⚠️  = Not found/skipped")

if __name__ == "__main__":
    main()
