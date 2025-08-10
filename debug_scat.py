#!/usr/bin/env python3
"""Simple SCAT test to debug the vicinity_index issue."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cdr.scat_adapter import SCATAdapter

def test_simple():
    """Simple test to verify SCATAdapter initialization."""
    try:
        # Create a test directory
        test_dir = Path(__file__).parent / "test_simple_scat"
        test_dir.mkdir(exist_ok=True)
        
        # Create a minimal flight file
        import json
        test_flight = {
            "fpl": {"fpl_base": [{"callsign": "TEST", "aircraft_type": "B738"}]},
            "plots": [{"time_of_track": "2024-08-10T12:00:00.000000", 
                      "I062/105": {"lat": 51.0, "lon": 0.0},
                      "I062/136": {"measured_flight_level": 350}}],
            "centre_ctrl": [{"centre_id": "TEST"}]
        }
        
        with open(test_dir / "test.json", 'w') as f:
            json.dump(test_flight, f)
        
        # Test adapter creation
        adapter = SCATAdapter(str(test_dir))
        print(f"SCATAdapter created successfully")
        print(f"Has vicinity_index: {hasattr(adapter, 'vicinity_index')}")
        
        if hasattr(adapter, 'vicinity_index'):
            print(f"vicinity_index type: {type(adapter.vicinity_index)}")
            print(f"vicinity_index methods: {[m for m in dir(adapter.vicinity_index) if not m.startswith('_')]}")
        else:
            print("vicinity_index attribute missing!")
            print(f"Available attributes: {[attr for attr in dir(adapter) if not attr.startswith('_')]}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()
