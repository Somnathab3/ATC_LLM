#!/usr/bin/env python3
"""
Legacy shim for scat_baseline_cli.py
This script redirects to the unified CLI at cli.py
"""

import sys
import warnings
import subprocess
from pathlib import Path


def show_deprecation_warning():
    """Show deprecation warning to users"""
    warnings.warn(
        "\nDeprecationWarning: Use 'python cli.py simulate --scat-dir ...' instead. "
        "This script will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )


def main():
    """Main entry point - redirect to unified CLI"""
    show_deprecation_warning()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    cli_path = script_dir.parent / "cli.py"
    
    # Delegate to unified CLI with 'simulate' subcommand
    cmd = [sys.executable, str(cli_path), "simulate"] + sys.argv[1:]
    
    # Run the unified CLI
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
