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
    
    # Method 1: Try direct import from project root (if installed properly)
    try:
        # Add project root to path if we can find it
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # src -> ATC_LLM
        
        # Check if cli.py exists in the project root
        cli_file = project_root / "cli.py"
        if cli_file.exists():
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from cli import main
            return main()
            
    except (ImportError, Exception) as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try finding cli.py in current working directory
    try:
        cwd = Path.cwd()
        cli_file = cwd / "cli.py"
        if cli_file.exists():
            if str(cwd) not in sys.path:
                sys.path.insert(0, str(cwd))
            from cli import main
            return main()
    except (ImportError, Exception) as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Try finding ATC_LLM directory in parent directories
    try:
        current_dir = Path.cwd()
        # Go up the directory tree looking for ATC_LLM/cli.py
        for i in range(5):  # Check up to 5 levels up
            potential_cli = current_dir / "cli.py"
            if potential_cli.exists():
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                from cli import main
                return main()
            
            # Also check for ATC_LLM subdirectory
            atc_llm_dir = current_dir / "ATC_LLM"
            if atc_llm_dir.exists():
                potential_cli = atc_llm_dir / "cli.py"
                if potential_cli.exists():
                    if str(atc_llm_dir) not in sys.path:
                        sys.path.insert(0, str(atc_llm_dir))
                    from cli import main
                    return main()
            
            current_dir = current_dir.parent
            if current_dir == current_dir.parent:  # Reached root
                break
                
    except (ImportError, Exception) as e:
        print(f"Method 3 failed: {e}")
    
    # Method 4: Try environment-based detection
    try:
        # Check if we're in a virtual environment and find ATC_LLM
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_root = Path(sys.prefix).parent
            atc_llm_dir = venv_root / "ATC_LLM_01" / "ATC_LLM"
            if atc_llm_dir.exists() and (atc_llm_dir / "cli.py").exists():
                if str(atc_llm_dir) not in sys.path:
                    sys.path.insert(0, str(atc_llm_dir))
                from cli import main
                return main()
    except (ImportError, Exception) as e:
        print(f"Method 4 failed: {e}")
    
    # If all methods fail, provide helpful error message
    print("Error: Could not locate the ATC-LLM CLI module.")
    print("This usually means one of the following:")
    print("1. You need to run 'pip install -e .' from the ATC_LLM directory")
    print("2. You need to be in the ATC_LLM directory when running the command")
    print("3. The installation is incomplete or corrupted")
    print()
    print("Try running these commands:")
    print("  cd F:\\ATC_LLM_01\\ATC_LLM")
    print("  pip install -e .")
    print("  atc-llm health-check")
    print()
    print("Alternative - run directly:")
    print("  cd F:\\ATC_LLM_01\\ATC_LLM")
    print("  python cli.py health-check --verbose --test-llm --test-bluesky --test-scat")
    print()
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Script location: {__file__}")
    return 1

if __name__ == "__main__":
    sys.exit(console_main())
