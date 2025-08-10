#!/usr/bin/env python3
"""SCAT JSONL Generator - Produce normalized JSONL artifacts from SCAT data.

This script processes SCAT dataset files and produces normalized JSONL artifacts:
- ownship_track.jsonl: Contains ownship trajectory data
- base_intruders.jsonl: Contains intruder aircraft within vicinity

Actions:
1) Use SCATAdapter.load_flight_record() to read flights and synchronized neighbors
2) Apply vicinity filtering: 100 NM radius, ±5000 ft altitude window using KDTree
3) Write JSONL with UTC seconds, lat_deg, lon_deg, alt_ft, hdg_deg, gs_kt

Acceptance criteria:
- Output files appear under Output/scat_simulation/
- Each record validates against schemas.AircraftState (pydantic v2)
- Timing logs show average vicinity query time and <1s per query for N<=10k
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from cdr.scat_adapter import SCATAdapter, SCATFlightRecord
from cdr.schemas import AircraftState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_aircraft_state_record(record: Dict[str, Any]) -> bool:
    """Validate that a JSONL record conforms to custom output schema.
    
    Args:
        record: Dictionary representing an aircraft state record
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Convert custom field names to AircraftState schema field names for validation
        validation_record = record.copy()
        
        # Map custom field names to AircraftState field names
        field_mapping = {
            'lat_deg': 'latitude',
            'lon_deg': 'longitude',
            'alt_ft': 'altitude_ft',
            'hdg_deg': 'heading_deg',
            'gs_kt': 'ground_speed_kt'
        }
        
        for custom_field, schema_field in field_mapping.items():
            if custom_field in validation_record:
                validation_record[schema_field] = validation_record.pop(custom_field)
        
        # Convert timestamp number back to datetime for validation
        if isinstance(validation_record.get('timestamp'), (int, float)):
            from datetime import datetime, timezone
            validation_record['timestamp'] = datetime.fromtimestamp(
                validation_record['timestamp'], tz=timezone.utc
            )
        elif isinstance(validation_record.get('timestamp'), str):
            validation_record['timestamp'] = datetime.fromisoformat(validation_record['timestamp'].replace('Z', '+00:00'))
        
        # Validate against AircraftState schema
        AircraftState(**validation_record)
        return True
    except Exception as e:
        logger.warning(f"Record validation failed: {e}")
        return False


def convert_to_jsonl_record(state: AircraftState) -> Dict[str, Any]:
    """Convert AircraftState to JSONL record format.
    
    Args:
        state: AircraftState object
        
    Returns:
        Dictionary ready for JSONL serialization
    """
    # Convert timestamp to UTC seconds since epoch
    utc_seconds = state.timestamp.timestamp()
    
    record = {
        'timestamp': utc_seconds,
        'aircraft_id': state.aircraft_id,
        'callsign': getattr(state, 'callsign', state.aircraft_id),
        'lat_deg': state.latitude,
        'lon_deg': state.longitude,
        'alt_ft': state.altitude_ft,
        'hdg_deg': state.heading_deg,
        'gs_kt': state.ground_speed_kt,
        'vertical_speed_fpm': state.vertical_speed_fpm,
        'aircraft_type': getattr(state, 'aircraft_type', None),
        'spawn_offset_min': getattr(state, 'spawn_offset_min', 0.0)
    }
    
    return record


def write_jsonl_file(file_path: Path, records: List[Dict[str, Any]], 
                     validate_records: bool = True) -> int:
    """Write records to JSONL file with optional validation.
    
    Args:
        file_path: Output file path
        records: List of record dictionaries
        validate_records: Whether to validate each record
        
    Returns:
        Number of records written
    """
    valid_count = 0
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            if validate_records:
                if not validate_aircraft_state_record(record.copy()):
                    logger.warning(f"Skipping invalid record for {record.get('aircraft_id', 'UNKNOWN')}")
                    continue
            
            json.dump(record, f, separators=(',', ':'))
            f.write('\n')
            valid_count += 1
    
    logger.info(f"Wrote {valid_count} valid records to {file_path}")
    return valid_count


def process_scat_data(scat_path: str, output_dir: Path, ownship_id: str,
                     vicinity_radius_nm: float = 100.0,
                     altitude_window_ft: float = 5000.0,
                     max_flights: int = 50) -> Tuple[str, str]:
    """Process SCAT data and generate normalized JSONL artifacts.
    
    Args:
        scat_path: Path to SCAT extracted dataset
        output_dir: Output directory for JSONL files
        ownship_id: Target ownship aircraft ID
        vicinity_radius_nm: Vicinity search radius in nautical miles
        altitude_window_ft: Altitude window in feet
        max_flights: Maximum number of flights to process
        
    Returns:
        Tuple of (ownship_file_path, intruders_file_path)
    """
    logger.info(f"Processing SCAT data from: {scat_path}")
    logger.info(f"Ownship ID: {ownship_id}")
    logger.info(f"Vicinity parameters: {vicinity_radius_nm} NM, ±{altitude_window_ft} ft")
    
    # Initialize SCAT adapter
    adapter = SCATAdapter(scat_path)
    
    # Load all aircraft states from scenario
    logger.info(f"Loading scenario with up to {max_flights} flights...")
    all_states = adapter.load_scenario(max_flights=max_flights, time_window_minutes=0)
    
    if not all_states:
        raise ValueError("No aircraft states loaded from SCAT data")
    
    logger.info(f"Loaded {len(all_states)} total aircraft states")
    
    # Build spatial index for vicinity searches
    logger.info("Building spatial index for vicinity queries...")
    start_time = time.perf_counter()
    adapter.build_spatial_index(all_states)
    index_time_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(f"Spatial index built in {index_time_ms:.1f}ms")
    
    # Separate ownship and find intruders
    ownship_states = [s for s in all_states if s.aircraft_id == ownship_id]
    if not ownship_states:
        available_ids = sorted(set(s.aircraft_id for s in all_states[:20]))  # Show first 20 IDs
        raise ValueError(f"No states found for ownship {ownship_id}. Available IDs: {available_ids}")
    
    logger.info(f"Found {len(ownship_states)} ownship states for {ownship_id}")
    
    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate ownship track JSONL
    ownship_file = output_dir / "ownship_track.jsonl"
    ownship_records = []
    
    for state in sorted(ownship_states, key=lambda s: s.timestamp):
        record = convert_to_jsonl_record(state)
        ownship_records.append(record)
    
    ownship_count = write_jsonl_file(ownship_file, ownship_records)
    
    # Generate intruders JSONL with vicinity filtering
    intruders_file = output_dir / "base_intruders.jsonl"
    intruder_states = []
    intruder_ids = set()  # Track unique aircraft IDs to avoid duplicates
    
    # Performance tracking
    total_queries = 0
    total_query_time_ms = 0.0
    query_times = []
    
    logger.info("Finding vicinity aircraft for each ownship position...")
    
    # For each ownship position, find vicinity aircraft
    for i, ownship_state in enumerate(sorted(ownship_states, key=lambda s: s.timestamp)):
        query_start = time.perf_counter()
        
        vicinity_aircraft = adapter.find_vicinity_aircraft(
            ownship_state, vicinity_radius_nm, altitude_window_ft
        )
        
        query_time_ms = (time.perf_counter() - query_start) * 1000.0
        total_query_time_ms += query_time_ms
        query_times.append(query_time_ms)
        total_queries += 1
        
        for aircraft in vicinity_aircraft:
            aircraft_key = f"{aircraft.aircraft_id}_{aircraft.timestamp.isoformat()}"
            if aircraft_key not in intruder_ids:
                intruder_ids.add(aircraft_key)
                intruder_states.append(aircraft)
        
        if (i + 1) % 10 == 0:
            avg_time = sum(query_times[-10:]) / min(10, len(query_times))
            logger.info(f"Processed {i + 1}/{len(ownship_states)} ownship positions, "
                       f"avg query time: {avg_time:.2f}ms")
    
    # Convert intruder states to records
    intruder_records = []
    for state in sorted(intruder_states, key=lambda s: (s.aircraft_id, s.timestamp)):
        record = convert_to_jsonl_record(state)
        intruder_records.append(record)
    
    intruder_count = write_jsonl_file(intruders_file, intruder_records)
    
    # Performance summary
    avg_query_time_ms = total_query_time_ms / total_queries if total_queries > 0 else 0
    queries_per_second = 1000.0 / avg_query_time_ms if avg_query_time_ms > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("SCAT JSONL Generation Complete")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Ownship track: {ownship_file} ({ownship_count} records)")
    logger.info(f"Base intruders: {intruders_file} ({intruder_count} records)")
    logger.info(f"Unique intruder aircraft: {len(set(r['aircraft_id'] for r in intruder_records))}")
    logger.info("\nVicinity Query Performance:")
    logger.info(f"  Total queries: {total_queries}")
    logger.info(f"  Average query time: {avg_query_time_ms:.2f}ms")
    logger.info(f"  Query throughput: {queries_per_second:.1f} queries/sec")
    logger.info(f"  Index build time: {index_time_ms:.1f}ms")
    
    # Performance validation
    if avg_query_time_ms > 1000.0:  # > 1 second
        logger.warning(f"⚠️  Average query time {avg_query_time_ms:.2f}ms exceeds 1s threshold!")
    else:
        logger.info(f"✅ Performance target met: {avg_query_time_ms:.2f}ms < 1s per query")
    
    # Get detailed vicinity performance from adapter
    perf_summary = adapter.get_vicinity_performance_summary()
    if perf_summary:
        logger.info("\nDetailed Vicinity Index Performance:")
        for key, value in perf_summary.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("="*60)
    
    return str(ownship_file), str(intruders_file)


def list_available_aircraft(scat_path: str, max_flights: int = 10) -> List[str]:
    """List available aircraft IDs in SCAT dataset.
    
    Args:
        scat_path: Path to SCAT extracted dataset
        max_flights: Maximum number of flights to check
        
    Returns:
        List of available aircraft IDs
    """
    try:
        adapter = SCATAdapter(scat_path)
        states = adapter.load_scenario(max_flights=max_flights, time_window_minutes=0)
        
        aircraft_ids = sorted(set(s.aircraft_id for s in states))
        return aircraft_ids
    except Exception as e:
        logger.error(f"Error listing aircraft: {e}")
        return []


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate normalized JSONL artifacts from SCAT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate JSONL for specific ownship
  python scat_jsonl_generator.py F:/SCAT_extracted AAL123

  # List available aircraft IDs
  python scat_jsonl_generator.py F:/SCAT_extracted --list-aircraft

  # Custom vicinity parameters
  python scat_jsonl_generator.py F:/SCAT_extracted UAL456 --radius 150 --altitude-window 8000
        """
    )
    
    parser.add_argument('scat_path', help='Path to extracted SCAT dataset directory')
    parser.add_argument('ownship_id', nargs='?', help='Target ownship aircraft ID')
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory (default: Output/scat_simulation/)')
    parser.add_argument('--radius', '-r', type=float, default=100.0,
                       help='Vicinity radius in nautical miles (default: 100.0)')
    parser.add_argument('--altitude-window', '-a', type=float, default=5000.0,
                       help='Altitude window in feet (default: 5000.0)')
    parser.add_argument('--max-flights', '-m', type=int, default=50,
                       help='Maximum flights to process (default: 50)')
    parser.add_argument('--list-aircraft', action='store_true',
                       help='List available aircraft IDs and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate SCAT path
    scat_path = Path(args.scat_path)
    if not scat_path.exists():
        logger.error(f"SCAT path does not exist: {scat_path}")
        return 1
    
    # List aircraft mode
    if args.list_aircraft:
        logger.info(f"Listing available aircraft in {scat_path}...")
        aircraft_ids = list_available_aircraft(str(scat_path), max_flights=args.max_flights)
        
        if aircraft_ids:
            logger.info(f"Found {len(aircraft_ids)} aircraft:")
            for i, aircraft_id in enumerate(aircraft_ids, 1):
                print(f"  {i:3d}. {aircraft_id}")
        else:
            logger.warning("No aircraft found in dataset")
        
        return 0
    
    # Validate ownship ID is provided
    if not args.ownship_id:
        logger.error("Ownship ID is required when not using --list-aircraft")
        parser.print_help()
        return 1
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to Output/scat_simulation/
        script_dir = Path(__file__).parent
        output_dir = script_dir / "Output" / "scat_simulation"
    
    try:
        # Generate JSONL artifacts
        ownship_file, intruders_file = process_scat_data(
            scat_path=str(scat_path),
            output_dir=output_dir,
            ownship_id=args.ownship_id,
            vicinity_radius_nm=args.radius,
            altitude_window_ft=args.altitude_window,
            max_flights=args.max_flights
        )
        
        logger.info(f"✅ Successfully generated JSONL artifacts:")
        logger.info(f"   Ownship: {ownship_file}")
        logger.info(f"   Intruders: {intruders_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error generating JSONL artifacts: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
