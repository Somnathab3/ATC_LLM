#!/usr/bin/env python3
"""
SCAT Baseline Builder CLI - Epic C2 Implementation

Command-line tool for building SCAT neighbor baselines.
Creates baseline.jsonl and baseline_paths.geojson for LLM runs.

Usage:
    python scat_baseline_cli.py --root F:/SCAT_extracted --ownship 100000.json --output ./baseline_results
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "bin" else script_dir
sys.path.insert(0, str(project_root))

from src.cdr.scat_baseline import SCATBaselineBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build SCAT neighbor baseline for proximity analysis',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Build baseline for flight 100000.json
  python scat_baseline_cli.py --root F:/SCAT_extracted --ownship 100000.json --output baseline_results

  # Use custom proximity criteria
  python scat_baseline_cli.py --root F:/SCAT_extracted --ownship 100000.json --output baseline_results \\
    --radius-nm 50.0 --altitude-ft 3000.0

  # Analyze more neighbors
  python scat_baseline_cli.py --root F:/SCAT_extracted --ownship 100000.json --output baseline_results \\
    --max-neighbors 100
        """
    )
    
    parser.add_argument('--root', required=True, 
                       help='Root directory containing SCAT files')
    parser.add_argument('--ownship', required=True, 
                       help='Ownship SCAT file (e.g., 100000.json)')
    parser.add_argument('--output', required=True, 
                       help='Output directory for baseline files')
    parser.add_argument('--radius-nm', type=float, default=100.0,
                       help='Horizontal proximity radius in NM (default: 100.0)')
    parser.add_argument('--altitude-ft', type=float, default=5000.0,
                       help='Vertical proximity window in feet (default: 5000.0)')
    parser.add_argument('--min-segment-min', type=float, default=2.0,
                       help='Minimum segment duration in minutes (default: 2.0)')
    parser.add_argument('--max-neighbors', type=int, default=50,
                       help='Maximum number of neighbor files to analyze (default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    scat_root = Path(args.root)
    if not scat_root.exists():
        logger.error(f"SCAT root directory does not exist: {scat_root}")
        return 1
    
    ownship_file = scat_root / args.ownship
    if not ownship_file.exists():
        logger.error(f"Ownship file does not exist: {ownship_file}")
        return 1
    
    output_dir = Path(args.output)
    
    try:
        logger.info("=== SCAT Baseline Builder (Epic C2) ===")
        logger.info(f"SCAT Root: {scat_root}")
        logger.info(f"Ownship: {args.ownship}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Proximity: {args.radius_nm} NM / {args.altitude_ft} ft")
        logger.info(f"Max neighbors: {args.max_neighbors}")
        
        # Create baseline builder
        builder = SCATBaselineBuilder(
            scat_root=str(scat_root),
            proximity_radius_nm=args.radius_nm,
            altitude_window_ft=args.altitude_ft,
            min_segment_duration_min=args.min_segment_min
        )
        
        # Build baseline
        logger.info("Building baseline...")
        baseline = builder.build_baseline(args.ownship, args.max_neighbors)
        
        # Save results
        logger.info("Saving results...")
        builder.save_baseline(baseline, str(output_dir))
        
        # Print summary
        neighbor_count = len(set(seg.neighbor_id for seg in baseline.neighbor_segments))
        total_duration = sum(
            (seg.end_time - seg.start_time).total_seconds() / 60.0 
            for seg in baseline.neighbor_segments
        )
        
        print(f"\\n[OK] SCAT baseline building completed successfully!")
        print(f"   Ownship: {baseline.ownship_id}")
        print(f"   Neighbors analyzed: {baseline.total_neighbors}")
        print(f"   Neighbors with proximity: {neighbor_count}")
        print(f"   Total proximity segments: {len(baseline.neighbor_segments)}")
        print(f"   Total proximity duration: {total_duration:.1f} minutes")
        print(f"   Results saved to: {output_dir}")
        
        # Output file details
        baseline_files = [
            f"baseline_{baseline.ownship_id}.jsonl",
            f"baseline_paths_{baseline.ownship_id}.geojson", 
            f"baseline_summary_{baseline.ownship_id}.json"
        ]
        print(f"\\n   Files created:")
        for filename in baseline_files:
            file_path = output_dir / filename
            if file_path.exists():
                print(f"     ✓ {filename}")
            else:
                print(f"     ✗ {filename} (missing)")
        
        # Show sample segments if any
        if baseline.neighbor_segments:
            print(f"\\n   Sample proximity segments:")
            for i, segment in enumerate(baseline.neighbor_segments[:5]):  # Show first 5
                duration_min = (segment.end_time - segment.start_time).total_seconds() / 60.0
                print(f"     {i+1}. {segment.neighbor_id}: {duration_min:.1f} min, "
                      f"min_dist={segment.min_distance_nm:.1f} NM, "
                      f"min_alt_diff={segment.min_altitude_diff_ft:.0f} ft")
            
            if len(baseline.neighbor_segments) > 5:
                print(f"     ... and {len(baseline.neighbor_segments) - 5} more segments")
        
        print(f"\\n[INFO] Use these baseline files as input for LLM runs:")
        print(f"   baseline_{baseline.ownship_id}.jsonl -> seed intruders")
        print(f"   baseline_paths_{baseline.ownship_id}.geojson -> visualization")
        
        return 0
        
    except Exception as e:
        logger.error(f"Baseline building failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
