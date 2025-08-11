#!/usr/bin/env python3
"""
ATC LLM CLI Entry Point

This module provides the console entry point for the ATC LLM system.
It imports and calls the main CLI function from the root cli.py module.
"""

import sys
from pathlib import Path

def console_main():
    """Console entry point for the atc-llm command."""
    try:
        # Add project root to path to import cli module
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Import the main CLI module from project root
        from cli import main
        return main()
    except ImportError as e:
        print(f"[ERROR] Failed to import CLI module: {e}")
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        print(f"[DEBUG] Project root: {Path(__file__).parent.parent}")
        print(f"[DEBUG] Python path: {sys.path}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error in CLI entry point: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(console_main())