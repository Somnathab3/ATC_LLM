#!/usr/bin/env python3
"""
DEPRECATED: Use 'python cli.py visualize' instead.

This script is deprecated and will be removed in a future version.
Please use the unified CLI interface instead.
"""

import sys
import warnings
import subprocess
from pathlib import Path

def main():
    """Deprecated entry point - redirects to unified CLI."""
    warnings.warn(
        "DeprecationWarning: Use 'python cli.py visualize ...' instead. "
        "This script will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Map any existing arguments to the new CLI format
    args = sys.argv[1:]
    
    # Construct the new command
    cli_path = Path(__file__).parent.parent / "cli.py"
    cmd = [sys.executable, str(cli_path), "visualize"] + args
    
    # Execute the new CLI
    return subprocess.run(cmd).returncode

if __name__ == "__main__":
    sys.exit(main())
