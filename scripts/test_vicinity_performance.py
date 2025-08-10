#!/usr/bin/env python3
"""Example script demonstrating VicinityIndex performance logging."""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cdr.scat_adapter import SCATAdapter, VicinityIndex
from cdr.schemas import AircraftState
from datetime import datetime, timezone

def create_test_aircraft_states(count: int = 1000):
    """Create test aircraft states for performance testing."""
    import random
    
    states = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(count):
        # Create random positions around a center point (simulating busy airspace)
        center_lat, center_lon = 40.0, -74.0  # Around NYC area
        
        # Random positions within ~200 NM radius
        lat_offset = random.uniform(-3.0, 3.0)  # ~180 NM max
        lon_offset = random.uniform(-3.0, 3.0)
        
        state = AircraftState(
            aircraft_id=f"AC{i:04d}",
            timestamp=base_time,
            latitude=center_lat + lat_offset,
            longitude=center_lon + lon_offset,
            altitude_ft=random.randint(10000, 40000),
            ground_speed_kt=random.randint(200, 500),
            heading_deg=random.randint(0, 359),
            vertical_speed_fpm=random.randint(-2000, 2000),
            callsign=f"FLIGHT{i:04d}"
        )
        states.append(state)
    
    return states

def test_vicinity_performance():
    """Test VicinityIndex performance with different configurations."""
    print("VicinityIndex Performance Test")
    print("=" * 50)
    
    # Create test data
    print("Creating test aircraft states...")
    aircraft_states = create_test_aircraft_states(1000)
    ownship = aircraft_states[0]  # Use first aircraft as ownship
    
    # Test with KDTree (if available)
    print("\n1. Testing with KDTree:")
    vicinity_index_kdtree = VicinityIndex(use_kdtree=True)
    vicinity_index_kdtree.build_index(aircraft_states)
    
    # Run multiple queries for performance statistics
    print("   Running 100 vicinity queries...")
    for i in range(100):
        vicinity_index_kdtree.query_vicinity(ownship, radius_nm=100.0, altitude_window_ft=5000.0)
    
    print("   KDTree Performance Summary:")
    summary = vicinity_index_kdtree.get_performance_summary()
    for key, value in summary.items():
        print(f"     {key}: {value}")
    
    # Test with linear search
    print("\n2. Testing with Linear Search:")
    vicinity_index_linear = VicinityIndex(use_kdtree=False)
    vicinity_index_linear.build_index(aircraft_states)
    
    # Run multiple queries for performance statistics
    print("   Running 100 vicinity queries...")
    for i in range(100):
        vicinity_index_linear.query_vicinity(ownship, radius_nm=100.0, altitude_window_ft=5000.0)
    
    print("   Linear Search Performance Summary:")
    summary = vicinity_index_linear.get_performance_summary()
    for key, value in summary.items():
        print(f"     {key}: {value}")
    
    # Compare performance
    kdtree_avg = vicinity_index_kdtree.performance.avg_query_time_ms
    linear_avg = vicinity_index_linear.performance.avg_query_time_ms
    
    if kdtree_avg > 0 and linear_avg > 0:
        speedup = linear_avg / kdtree_avg
        print(f"\n3. Performance Comparison:")
        print(f"   KDTree avg query time: {kdtree_avg:.2f}ms")
        print(f"   Linear avg query time: {linear_avg:.2f}ms")
        print(f"   KDTree speedup: {speedup:.1f}x faster")
    
    print("\nVicinityIndex performance test completed!")

def test_scat_adapter_integration():
    """Test SCATAdapter integration with VicinityIndex."""
    print("\nSCATAdapter Integration Test")
    print("=" * 50)
    
    # Note: This would require actual SCAT data to run fully
    # For demo purposes, we'll just show the API usage
    
    print("Example usage with SCATAdapter:")
    print("""
    # Initialize SCAT adapter with VicinityIndex
    adapter = SCATAdapter("/path/to/scat/data")
    
    # Load scenario data
    aircraft_states = adapter.load_scenario(max_flights=50)
    
    # Build spatial index with performance tracking
    adapter.build_spatial_index(aircraft_states)
    
    # Find vicinity aircraft with performance logging
    ownship = aircraft_states[0]
    vicinity = adapter.find_vicinity_aircraft(
        ownship, radius_nm=100.0, altitude_window_ft=5000.0
    )
    
    # Get performance metrics
    performance = adapter.get_vicinity_performance_summary()
    print(f"Query performance: {performance}")
    
    # Log performance summary
    adapter.log_vicinity_performance()
    """)
    
    print("Integration test info displayed!")

if __name__ == "__main__":
    print("VicinityIndex Performance Demonstration")
    print("=" * 60)
    
    try:
        test_vicinity_performance()
        test_scat_adapter_integration()
        
    except Exception as e:
        print(f"Error during performance test: {e}")
        import traceback
        traceback.print_exc()
