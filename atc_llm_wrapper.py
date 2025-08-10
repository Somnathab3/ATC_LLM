#!/usr/bin/env python3
"""
Wrapper script for ATC-LLM console script entry point.
This ensures the console script works from any directory.
"""
import sys
import os
from pathlib import Path

def main():
    """Main entry point for the atc-llm console script."""
    # Find the project root directory (where cli.py is located)
    # This works whether installed as a package or run from source
    
    # First, try to find cli.py in the same directory as this script
    wrapper_dir = Path(__file__).parent
    cli_path = wrapper_dir / "cli.py"
    
    if cli_path.exists():
        # Add the project root to sys.path
        sys.path.insert(0, str(wrapper_dir))
        
        # Import and run the main function from cli.py
        try:
            import cli
            return cli.main()
        except Exception as e:
            print(f"Error running ATC-LLM: {e}")
            return 1
    else:
        # If cli.py is not found, show usage information
        print("""
ATC-LLM Console Script

This command should be run from the ATC_LLM project directory.

Usage from project directory:
    python cli.py [command] [options]

Available commands:
    health-check     - System health check
    simulate         - Run unified simulation
    batch            - Batch processing operations
    metrics          - Calculate Wolfgang and basic metrics  
    report           - Generate enhanced reporting
    verify-llm       - Verify LLM connectivity
    visualize        - Generate conflict visualizations

For help on any command:
    python cli.py [command] --help

Installation:
    pip install -e .  # Install in development mode
        """)
        return 1

if __name__ == "__main__":
    sys.exit(main())
