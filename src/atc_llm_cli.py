#!/usr/bin/env python3
"""
Entry point for ATC-LLM console script.
This module imports and runs the main CLI function.
"""

import sys
import os
from pathlib import Path

def console_main():
    """Console script entry point."""
    # Try multiple ways to find and import the CLI module
    
    # Method 1: Try direct import (if installed properly)
    try:
        from cli import main
        return main()
    except ImportError:
        pass
    
    # Method 2: Try adding project root to path
    try:
        # Find the project root by looking for cli.py
        current_dir = Path(__file__).parent.parent  # Start from src parent
        
        # Look for cli.py in current directory and parent directories
        for potential_root in [current_dir, current_dir.parent]:
            cli_file = potential_root / "cli.py"
            if cli_file.exists():
                sys.path.insert(0, str(potential_root))
                from cli import main
                return main()
        
        # Method 3: Try relative to the installed script location
        script_dir = Path(sys.argv[0]).parent.parent  # Go up from Scripts dir
        for potential_root in [script_dir, script_dir.parent]:
            cli_file = potential_root / "ATC_LLM" / "cli.py"
            if cli_file.exists():
                sys.path.insert(0, str(potential_root / "ATC_LLM"))
                from cli import main
                return main()
                
    except (ImportError, Exception):
        pass
    
    # Method 4: Try finding ATC_LLM directory in common locations
    try:
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_root = Path(sys.prefix).parent  # Go up from .venv
            atc_llm_dir = venv_root / "ATC_LLM"
            if atc_llm_dir.exists() and (atc_llm_dir / "cli.py").exists():
                sys.path.insert(0, str(atc_llm_dir))
                from cli import main
                return main()
    except (ImportError, Exception):
        pass
    
    # If all methods fail, provide helpful error message
    print("Error: Could not locate the ATC-LLM CLI module.")
    print("This usually means one of the following:")
    print("1. You need to run 'pip install -e .' from the ATC_LLM directory")
    print("2. You need to be in the ATC_LLM directory when running the command")
    print("3. The installation is incomplete or corrupted")
    print()
    print("Try running these commands:")
    print("  cd /path/to/ATC_LLM")
    print("  pip install -e .")
    print("  atc-llm health-check")
    print()
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")
    return 1

if __name__ == "__main__":
    sys.exit(console_main())
