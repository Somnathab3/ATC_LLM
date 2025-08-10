#!/usr/bin/env python3
"""
ATC LLM CLI - Command Line Interface for LLM-BlueSky CDR System

This module provides a unified command-line interface for all major functions
of the Air Traffic Control LLM-driven Conflict Detection and Resolution system.

Usage:
    atc-llm --help                     # Show all available commands
    atc-llm simulate --help            # Show simulation options
    atc-llm batch --help               # Show batch processing options
    atc-llm compare --help             # Show comparison options
    atc-llm test --help                # Show testing options
    atc-llm health-check              # Run system health check
    atc-llm start-server              # Start API server
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.cdr.schemas import ConfigurationSettings, MonteCarloParameters


def setup_logging(verbose: bool = False):
    """Configure logging for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_health_check(args: argparse.Namespace) -> int:
    """Run system health check to verify all components are working."""
    setup_logging(args.verbose)
    
    try:
        from repo_healthcheck import main as health_main
        health_main()
        print("[OK] System health check completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return 1


def cmd_simulate_basic(args: argparse.Namespace) -> int:
    """Run basic simulation with generated scenarios."""
    setup_logging(args.verbose)
    
    try:
        from complete_llm_demo import main as demo_main
        
        # Configure for basic simulation
        print(f"[INIT] Starting basic simulation...")
        print(f"   Aircraft: {args.aircraft}")
        print(f"   Duration: {args.duration} minutes")
        print(f"   LLM Model: {args.llm_model}")
        
        demo_main()
        print("[OK] Basic simulation completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Basic simulation failed: {e}")
        return 1


def cmd_simulate_scat(args: argparse.Namespace) -> int:
    """Run simulation with SCAT real-world data."""
    setup_logging(args.verbose)
    
    try:
        from batch_scat_llm_processor import main as batch_main
        
        print(f"[INIT] Starting SCAT simulation...")
        print(f"   SCAT Directory: {args.scat_dir}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Scenarios per Flight: {args.scenarios_per_flight}")
        print(f"   Output Directory: {args.output_dir}")
        
        # Override sys.argv for the batch processor
        original_argv = sys.argv.copy()
        sys.argv = [
            'batch_scat_llm_processor.py',
            '--scat-dir', args.scat_dir,
            '--max-flights', str(args.max_flights),
            '--scenarios-per-flight', str(args.scenarios_per_flight),
            '--output-dir', args.output_dir,
            '--llm-model', args.llm_model
        ]
        if args.verbose:
            sys.argv.append('--verbose')
            
        result = batch_main()
        sys.argv = original_argv
        
        if result == 0:
            print("[OK] SCAT simulation completed successfully!")
        else:
            print("[ERROR] SCAT simulation failed!")
        return result
    except Exception as e:
        print(f"[ERROR] SCAT simulation failed: {e}")
        return 1


def cmd_batch_production(args: argparse.Namespace) -> int:
    """Run production batch processing with all safety checks."""
    setup_logging(args.verbose)
    
    try:
        from production_batch_processor import main as prod_main
        
        print(f"[FACTORY] Starting production batch processing...")
        print(f"   SCAT Directory: {args.scat_dir}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Scenarios per Flight: {args.scenarios_per_flight}")
        
        # Override sys.argv for the production processor
        original_argv = sys.argv.copy()
        sys.argv = [
            'production_batch_processor.py',
            '--scat-dir', args.scat_dir,
            '--max-flights', str(args.max_flights),
            '--scenarios-per-flight', str(args.scenarios_per_flight),
            '--output-dir', args.output_dir
        ]
        if args.skip_checks:
            sys.argv.append('--skip-checks')
        if args.verbose:
            sys.argv.append('--verbose')
            
        result = prod_main()
        sys.argv = original_argv
        
        if result == 0:
            print("[OK] Production batch processing completed successfully!")
        else:
            print("[ERROR] Production batch processing failed!")
        return result
    except Exception as e:
        print(f"[ERROR] Production batch processing failed: {e}")
        return 1


def cmd_compare_baseline_llm(args: argparse.Namespace) -> int:
    """Compare baseline vs LLM performance."""
    setup_logging(args.verbose)
    
    try:
        # Compare functionality moved to test suite
        print("[ERROR] Compare functionality not available. Use test suite instead.")
        return 1
        print(f"[STATS] Starting baseline vs LLM comparison...")
        print(f"   SCAT Path: {args.scat_path}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Time Window: {args.time_window} minutes")
        
        # Override sys.argv for the comparison
        original_argv = sys.argv.copy()
        sys.argv = [
            'demo_baseline_vs_llm.py',
            '--scat-path', args.scat_path,
            '--max-flights', str(args.max_flights),
            '--time-window', str(args.time_window),
            '--output', args.output
        ]
        
        compare_main()
        sys.argv = original_argv
        
        print("[OK] Baseline vs LLM comparison completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        return 1


def cmd_test_suite(args: argparse.Namespace) -> int:
    """Run comprehensive test suite."""
    setup_logging(args.verbose)
    
    try:
        import subprocess
        
        print(f"[EXPERIMENT] Running test suite...")
        
        if args.coverage:
            cmd = ['python', '-m', 'pytest', '--cov=src', '--cov-report=html', '--cov-report=term']
        else:
            cmd = ['python', '-m', 'pytest']
            
        if args.verbose:
            cmd.append('-v')
            
        if args.test_pattern:
            cmd.extend(['-k', args.test_pattern])
            
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("[OK] Test suite completed successfully!")
            if args.coverage:
                print("[STATS] Coverage report generated in htmlcov/")
        else:
            print("[ERROR] Some tests failed!")
            
        return result.returncode
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return 1


def cmd_start_server(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    setup_logging(args.verbose)
    
    try:
        import uvicorn
        from src.api.service import app
        
        print(f"[GLOBAL] Starting API server...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Debug: {args.debug}")
        
        uvicorn.run(
            "src.api.service:app",
            host=args.host,
            port=args.port,
            reload=args.debug,
            log_level="debug" if args.verbose else "info"
        )
        
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        return 1


def cmd_verify_llm(args: argparse.Namespace) -> int:
    """Verify LLM connectivity and functionality."""
    setup_logging(args.verbose)
    
    try:
        # Verification moved to tests directory
        print("[ERROR] LLM verification not available via CLI. Use tests/verify_llm_communication.py directly.")
        return 1
        
        print(f"[BOT] Verifying LLM connectivity...")
        print(f"   Model: {args.model}")
        
        verify_main()
        print("[OK] LLM verification completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] LLM verification failed: {e}")
        return 1


def cmd_visualize_conflicts(args: argparse.Namespace) -> int:
    """Generate conflict visualization plots."""
    setup_logging(args.verbose)
    
    try:
        # Visualization functionality removed
        print("[ERROR] Visualization functionality not available. Use external plotting tools.")
        return 1
        
        print(f"[CHART] Generating conflict visualizations...")
        print(f"   Data file: {args.data_file}")
        print(f"   Output directory: {args.output_dir}")
        
        # Call visualization main function
        viz_main()
        print("[OK] Conflict visualizations generated successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='atc-llm',
        description='ATC LLM-BlueSky CDR System - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atc-llm health-check                                    # Check system health
  atc-llm simulate basic --aircraft 5 --duration 30      # Basic simulation
  atc-llm simulate scat --scat-dir /path/to/scat          # SCAT simulation
  atc-llm batch production --max-flights 10              # Production batch
  atc-llm compare --scat-path /path/to/scat               # Compare performance
  atc-llm test --coverage                                 # Run tests with coverage
  atc-llm server --port 8080                             # Start API server
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Check system health')
    health_parser.set_defaults(func=cmd_health_check)
    
    # Simulation commands
    sim_parser = subparsers.add_parser('simulate', help='Run simulations')
    sim_subparsers = sim_parser.add_subparsers(dest='sim_type', help='Simulation types')
    
    # Basic simulation
    basic_parser = sim_subparsers.add_parser('basic', help='Run basic simulation')
    basic_parser.add_argument('--aircraft', type=int, default=5,
                             help='Number of aircraft (default: 5)')
    basic_parser.add_argument('--duration', type=int, default=30,
                             help='Simulation duration in minutes (default: 30)')
    basic_parser.add_argument('--llm-model', default='llama3.1:8b',
                             help='LLM model to use (default: llama3.1:8b)')
    basic_parser.set_defaults(func=cmd_simulate_basic)
    
    # SCAT simulation
    scat_parser = sim_subparsers.add_parser('scat', help='Run SCAT data simulation')
    scat_parser.add_argument('--scat-dir', required=True,
                            help='Directory containing SCAT files')
    scat_parser.add_argument('--max-flights', type=int, default=3,
                            help='Maximum flights to process (default: 3)')
    scat_parser.add_argument('--scenarios-per-flight', type=int, default=5,
                            help='Scenarios per flight (default: 5)')
    scat_parser.add_argument('--output-dir', default='Output',
                            help='Output directory (default: Output)')
    scat_parser.add_argument('--llm-model', default='llama3.1:8b',
                            help='LLM model to use (default: llama3.1:8b)')
    scat_parser.set_defaults(func=cmd_simulate_scat)
    
    # Batch processing commands
    batch_parser = subparsers.add_parser('batch', help='Run batch processing')
    batch_subparsers = batch_parser.add_subparsers(dest='batch_type', help='Batch types')
    
    # Production batch
    prod_parser = batch_subparsers.add_parser('production', help='Production batch processing')
    prod_parser.add_argument('--scat-dir', default='F:\\SCAT_extracted',
                            help='SCAT directory (default: F:\\SCAT_extracted)')
    prod_parser.add_argument('--max-flights', type=int, default=5,
                            help='Maximum flights to process (default: 5)')
    prod_parser.add_argument('--scenarios-per-flight', type=int, default=5,
                            help='Scenarios per flight (default: 5)')
    prod_parser.add_argument('--output-dir', default='Output',
                            help='Output directory (default: Output)')
    prod_parser.add_argument('--skip-checks', action='store_true',
                            help='Skip prerequisite checks')
    prod_parser.set_defaults(func=cmd_batch_production)
    
    # Comparison commands
    comp_parser = subparsers.add_parser('compare', help='Compare system performance')
    comp_parser.add_argument('--scat-path', default='F:/SCAT_extracted',
                           help='Path to SCAT dataset (default: F:/SCAT_extracted)')
    comp_parser.add_argument('--max-flights', type=int, default=5,
                           help='Maximum flights to process (default: 5)')
    comp_parser.add_argument('--time-window', type=int, default=30,
                           help='Time window in minutes (default: 30)')
    comp_parser.add_argument('--output', default='demo_results.json',
                           help='Output file (default: demo_results.json)')
    comp_parser.set_defaults(func=cmd_compare_baseline_llm)
    
    # Testing commands
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--coverage', action='store_true',
                           help='Generate coverage report')
    test_parser.add_argument('--test-pattern', type=str,
                           help='Run tests matching pattern')
    test_parser.set_defaults(func=cmd_test_suite)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1',
                              help='Server host (default: 127.0.0.1)')
    server_parser.add_argument('--port', type=int, default=8000,
                              help='Server port (default: 8000)')
    server_parser.add_argument('--debug', action='store_true',
                              help='Enable debug mode with auto-reload')
    server_parser.set_defaults(func=cmd_start_server)
    
    # LLM verification command
    llm_parser = subparsers.add_parser('verify-llm', help='Verify LLM connectivity')
    llm_parser.add_argument('--model', default='llama3.1:8b',
                           help='LLM model to test (default: llama3.1:8b)')
    llm_parser.set_defaults(func=cmd_verify_llm)
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Generate conflict visualizations')
    viz_parser.add_argument('--data-file', required=True,
                           help='Data file to visualize')
    viz_parser.add_argument('--output-dir', default='visualizations',
                           help='Output directory (default: visualizations)')
    viz_parser.set_defaults(func=cmd_visualize_conflicts)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if hasattr(args, 'sim_type') and not args.sim_type:
        parser.parse_args(['simulate', '--help'])
        return 1
        
    if hasattr(args, 'batch_type') and not args.batch_type:
        parser.parse_args(['batch', '--help'])
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[WARN]  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
