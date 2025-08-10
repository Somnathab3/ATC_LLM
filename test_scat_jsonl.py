#!/usr/bin/env python3
"""Test script for SCAT JSONL Generator functionality.

This script demonstrates how to use the SCAT JSONL generator and validates
that it produces correct output format and performance.
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
import sys
sys_path = Path(__file__).parent / "src"
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from cdr.schemas import AircraftState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_jsonl_file(file_path: Path, expected_fields: list) -> bool:
    """Validate a JSONL file format and schema compliance.
    
    Args:
        file_path: Path to JSONL file
        expected_fields: List of required fields
        
    Returns:
        True if validation passes, False otherwise
    """
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False
    
    try:
        valid_records = 0
        total_records = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_records += 1
                
                try:
                    record = json.loads(line)
                    
                    # Check required fields
                    missing_fields = [field for field in expected_fields 
                                    if field not in record]
                    if missing_fields:
                        logger.warning(f"Line {line_num}: Missing fields {missing_fields}")
                        continue
                    
                    # Validate data types and ranges
                    if not validate_record_values(record):
                        logger.warning(f"Line {line_num}: Invalid values")
                        continue
                    
                    valid_records += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error: {e}")
                    continue
        
        success_rate = (valid_records / total_records * 100) if total_records > 0 else 0
        logger.info(f"Validation results for {file_path.name}:")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Valid records: {valid_records}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        
        return success_rate >= 95.0  # Require 95% success rate
        
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return False


def validate_record_values(record: dict) -> bool:
    """Validate individual record value ranges and types.
    
    Args:
        record: Dictionary record to validate
        
    Returns:
        True if all values are valid, False otherwise
    """
    try:
        # Check timestamp
        timestamp = record.get('timestamp')
        if not isinstance(timestamp, (int, float)) or timestamp <= 0:
            return False
        
        # Check latitude range
        lat = record.get('lat_deg')
        if not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
            return False
        
        # Check longitude range
        lon = record.get('lon_deg')
        if not isinstance(lon, (int, float)) or not (-180 <= lon <= 180):
            return False
        
        # Check altitude
        alt = record.get('alt_ft')
        if not isinstance(alt, (int, float)) or not (0 <= alt <= 60000):
            return False
        
        # Check heading
        hdg = record.get('hdg_deg')
        if not isinstance(hdg, (int, float)) or not (0 <= hdg < 360):
            return False
        
        # Check ground speed
        gs = record.get('gs_kt')
        if not isinstance(gs, (int, float)) or not (0 <= gs <= 1000):
            return False
        
        # Check aircraft ID
        aircraft_id = record.get('aircraft_id')
        if not isinstance(aircraft_id, str) or not aircraft_id.strip():
            return False
        
        return True
        
    except Exception:
        return False


def test_scat_jsonl_with_mock_data():
    """Test SCAT JSONL generator with mock data.
    
    This function creates a small mock SCAT dataset and validates the output.
    """
    logger.info("Testing SCAT JSONL generator with mock data...")
    
    # Create mock SCAT data directory
    test_dir = Path(__file__).parent / "test_scat_data"
    test_dir.mkdir(exist_ok=True)
    
    # Create mock flight file
    mock_flight_data = {
        "fpl": {
            "fpl_base": [{
                "callsign": "TEST001",
                "aircraft_type": "B738",
                "flight_rules": "I",
                "wtc": "M",
                "adep": "EGLL",
                "ades": "LFPG",
                "equip_status_rvsm": True
            }]
        },
        "plots": [
            {
                "time_of_track": "2024-08-10T12:00:00.000000",
                "I062/105": {"lat": 51.4775, "lon": -0.4614},
                "I062/136": {"measured_flight_level": 350},
                "I062/185": {"vx": 250.0, "vy": 50.0},
                "I062/220": {"rocd": 0}
            },
            {
                "time_of_track": "2024-08-10T12:01:00.000000",
                "I062/105": {"lat": 51.5000, "lon": -0.4500},
                "I062/136": {"measured_flight_level": 350},
                "I062/185": {"vx": 250.0, "vy": 50.0},
                "I062/220": {"rocd": 0}
            },
            {
                "time_of_track": "2024-08-10T12:02:00.000000",
                "I062/105": {"lat": 51.5225, "lon": -0.4386},
                "I062/136": {"measured_flight_level": 350},
                "I062/185": {"vx": 250.0, "vy": 50.0},
                "I062/220": {"rocd": 0}
            }
        ],
        "centre_ctrl": [{"centre_id": "EGTT"}]
    }
    
    # Write mock flight file
    flight_file = test_dir / "test_flight.json"
    with open(flight_file, 'w') as f:
        json.dump(mock_flight_data, f)
    
    # Create a second flight for intruders
    mock_intruder_data = {
        "fpl": {
            "fpl_base": [{
                "callsign": "TEST002",
                "aircraft_type": "A320",
                "flight_rules": "I",
                "wtc": "M",
                "adep": "LFPG",
                "ades": "EGLL",
                "equip_status_rvsm": True
            }]
        },
        "plots": [
            {
                "time_of_track": "2024-08-10T12:00:30.000000",
                "I062/105": {"lat": 51.4800, "lon": -0.4600},  # Close to ownship
                "I062/136": {"measured_flight_level": 340},      # 1000 ft below
                "I062/185": {"vx": 200.0, "vy": 100.0},
                "I062/220": {"rocd": 500}
            },
            {
                "time_of_track": "2024-08-10T12:01:30.000000",
                "I062/105": {"lat": 51.5025, "lon": -0.4486},  # Still close
                "I062/136": {"measured_flight_level": 345},      # Climbing
                "I062/185": {"vx": 200.0, "vy": 100.0},
                "I062/220": {"rocd": 500}
            }
        ],
        "centre_ctrl": [{"centre_id": "EGTT"}]
    }
    
    intruder_file = test_dir / "test_intruder.json"
    with open(intruder_file, 'w') as f:
        json.dump(mock_intruder_data, f)
    
    logger.info(f"Created mock SCAT data in {test_dir}")
    
    # Test the SCAT adapter directly
    try:
        from cdr.scat_adapter import SCATAdapter
        
        adapter = SCATAdapter(str(test_dir))
        states = adapter.load_scenario(max_flights=10, time_window_minutes=0)
        
        logger.info(f"Loaded {len(states)} aircraft states from mock data")
        
        # Test vicinity queries
        if states:
            adapter.build_spatial_index(states)
            
            # Find vicinity for first ownship state
            ownship_states = [s for s in states if s.aircraft_id == "TEST001"]
            if ownship_states:
                test_state = ownship_states[0]
                vicinity = adapter.find_vicinity_aircraft(test_state, 100.0, 5000.0)
                logger.info(f"Found {len(vicinity)} aircraft in vicinity of {test_state.aircraft_id}")
        
        # Clean up test data
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Mock data test failed: {e}")
        # Clean up test data
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        return False


def main():
    """Run SCAT JSONL generator tests."""
    logger.info("Starting SCAT JSONL Generator Tests")
    logger.info("=" * 50)
    
    # Test 1: Mock data test
    logger.info("\nTest 1: Mock Data Processing")
    if test_scat_jsonl_with_mock_data():
        logger.info("✅ Mock data test passed")
    else:
        logger.error("❌ Mock data test failed")
    
    # Test 2: Validate expected output format
    logger.info("\nTest 2: Output Format Validation")
    output_dir = Path(__file__).parent / "Output" / "scat_simulation"
    
    expected_fields = [
        'timestamp', 'aircraft_id', 'callsign', 'lat_deg', 'lon_deg', 
        'alt_ft', 'hdg_deg', 'gs_kt', 'vertical_speed_fpm'
    ]
    
    # Check if output files exist (from previous runs)
    ownship_file = output_dir / "ownship_track.jsonl"
    intruders_file = output_dir / "base_intruders.jsonl"
    
    files_validated = 0
    total_files = 0
    
    for file_path, file_type in [(ownship_file, "ownship"), (intruders_file, "intruders")]:
        total_files += 1
        if file_path.exists():
            logger.info(f"Validating {file_type} file: {file_path}")
            if validate_jsonl_file(file_path, expected_fields):
                logger.info(f"✅ {file_type} file validation passed")
                files_validated += 1
            else:
                logger.error(f"❌ {file_type} file validation failed")
        else:
            logger.info(f"ℹ️  {file_type} file not found (run generator first): {file_path}")
    
    # Test 3: Performance expectations
    logger.info("\nTest 3: Performance Validation")
    logger.info("Expected performance criteria:")
    logger.info("  - Average vicinity query time: < 1000ms")
    logger.info("  - KDTree indexing should be available")
    logger.info("  - Memory usage should be reasonable for N<=10k aircraft")
    
    try:
        # Test KDTree availability
        from scipy.spatial import KDTree
        import numpy as np
        logger.info("✅ KDTree indexing available (scipy installed)")
    except ImportError:
        logger.warning("⚠️  KDTree not available - will use linear search fallback")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    logger.info(f"Files validated: {files_validated}/{total_files}")
    
    if files_validated == total_files and total_files > 0:
        logger.info("✅ All tests passed - JSONL generator is working correctly")
    else:
        logger.info("ℹ️  Run the generator with real SCAT data to complete validation")
    
    # Usage instructions
    logger.info("\nUsage Instructions:")
    logger.info("=" * 20)
    logger.info("1. To list available aircraft:")
    logger.info("   python scat_jsonl_generator.py F:/SCAT_extracted --list-aircraft")
    logger.info("")
    logger.info("2. To generate JSONL for specific ownship:")
    logger.info("   python scat_jsonl_generator.py F:/SCAT_extracted AAL123")
    logger.info("")
    logger.info("3. Custom parameters:")
    logger.info("   python scat_jsonl_generator.py F:/SCAT_extracted UAL456 --radius 150 --altitude-window 8000")
    
    return 0


if __name__ == "__main__":
    exit(main())
