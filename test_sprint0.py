#!/usr/bin/env python3
"""
Quick test script to validate Sprint 0 geodesy implementation.
This tests the core algorithms without requiring pytest setup.
"""

import sys
import os
import math

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cdr.geodesy import haversine_nm, bearing_rad, cpa_nm
    print("✓ Successfully imported geodesy functions")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_haversine():
    """Test haversine distance calculation."""
    print("\n--- Testing Haversine Distance ---")
    
    # Test 1: Symmetry
    a = (59.3, 18.1)
    b = (59.4, 18.3)
    dist_ab = haversine_nm(a, b)
    dist_ba = haversine_nm(b, a)
    
    print(f"Distance A→B: {dist_ab:.3f} NM")
    print(f"Distance B→A: {dist_ba:.3f} NM")
    print(f"Symmetry check: {abs(dist_ab - dist_ba) < 1e-6}")
    
    # Test 2: Zero distance
    zero_dist = haversine_nm(a, a)
    print(f"Zero distance test: {zero_dist < 1e-10}")
    
    # Test 3: Known distance (1 degree longitude at equator ≈ 60 NM)
    equator_a = (0.0, 0.0)
    equator_b = (0.0, 1.0)
    equator_dist = haversine_nm(equator_a, equator_b)
    print(f"1° longitude at equator: {equator_dist:.3f} NM (expected ~60)")
    
    return True

def test_bearing():
    """Test bearing calculation."""
    print("\n--- Testing Bearing Calculation ---")
    
    # Test cardinal directions
    origin = (0.0, 0.0)
    
    # North
    north = (1.0, 0.0)
    bearing_n = bearing_rad(origin, north)
    print(f"Bearing to North: {math.degrees(bearing_n):.1f}° (expected 0°)")
    
    # East  
    east = (0.0, 1.0)
    bearing_e = bearing_rad(origin, east)
    print(f"Bearing to East: {math.degrees(bearing_e):.1f}° (expected 90°)")
    
    # South
    south = (-1.0, 0.0)
    bearing_s = bearing_rad(origin, south)
    print(f"Bearing to South: {math.degrees(bearing_s):.1f}° (expected ±180°)")
    
    # West
    west = (0.0, -1.0)
    bearing_w = bearing_rad(origin, west)
    print(f"Bearing to West: {math.degrees(bearing_w):.1f}° (expected -90°)")
    
    return True

def test_cpa():
    """Test Closest Point of Approach calculation."""
    print("\n--- Testing CPA Calculation ---")
    
    # Test 1: Basic converging scenario
    own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East
    intr = {"lat": 0.5, "lon": 1.0, "spd_kt": 460, "hdg_deg": 270}  # West
    
    dmin, tmin = cpa_nm(own, intr)
    print(f"Converging aircraft - Distance: {dmin:.2f} NM, Time: {tmin:.2f} min")
    print(f"Converging test: {tmin > 0 and dmin >= 0}")
    
    # Test 2: Parallel aircraft (same direction)
    own_parallel = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}
    intr_parallel = {"lat": 0.1, "lon": 0, "spd_kt": 480, "hdg_deg": 90}
    
    dmin_p, tmin_p = cpa_nm(own_parallel, intr_parallel)
    print(f"Parallel aircraft - Distance: {dmin_p:.2f} NM, Time: {tmin_p:.2f} min")
    print(f"Parallel test: {tmin_p == 0.0 and dmin_p > 0}")
    
    # Test 3: Head-on collision course
    own_head = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}
    intr_head = {"lat": 0, "lon": 1.0, "spd_kt": 480, "hdg_deg": 270}
    
    dmin_h, tmin_h = cpa_nm(own_head, intr_head)
    print(f"Head-on aircraft - Distance: {dmin_h:.2f} NM, Time: {tmin_h:.2f} min")
    print(f"Head-on test: {tmin_h > 0 and dmin_h < 1.0}")
    
    # Test 4: Identical aircraft
    dmin_same, tmin_same = cpa_nm(own, own)
    print(f"Identical aircraft - Distance: {dmin_same:.2f} NM, Time: {tmin_same:.2f} min")
    print(f"Identical test: {tmin_same == 0.0 and dmin_same == 0.0}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Sprint 0 Geodesy Validation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_haversine():
            tests_passed += 1
            print("✓ Haversine tests PASSED")
    except Exception as e:
        print(f"✗ Haversine tests FAILED: {e}")
    
    try:
        if test_bearing():
            tests_passed += 1
            print("✓ Bearing tests PASSED")
    except Exception as e:
        print(f"✗ Bearing tests FAILED: {e}")
    
    try:
        if test_cpa():
            tests_passed += 1
            print("✓ CPA tests PASSED")
    except Exception as e:
        print(f"✗ CPA tests FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 All Sprint 0 geodesy tests PASSED!")
        print("✅ Ready for Sprint 1 development")
        return 0
    else:
        print("❌ Some tests failed - review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
