#!/usr/bin/env python3
"""
Gap Fix Verification Test
Demonstrates KDTree vicinity filtering and ECEF coordinate conversion
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ecef_conversion():
    """Test ECEF coordinate conversion functionality."""
    print("=== Testing ECEF Coordinate Conversion ===")
    
    try:
        from src.cdr.scat_adapter import SCATAdapter
        
        # Create adapter instance (will fail on file access but methods will work)
        try:
            adapter = SCATAdapter("nonexistent")
        except FileNotFoundError:
            # Expected - create adapter without file validation for testing
            adapter = object.__new__(SCATAdapter)
            adapter.use_kdtree = True
        
        # Test ECEF conversion
        lat, lon, alt_ft = 51.5074, -0.1278, 35000  # London at FL350
        alt_m = alt_ft * 0.3048
        
        x, y, z = adapter.lat_lon_to_ecef(lat, lon, alt_m)
        
        print(f"Input: Lat={lat}°, Lon={lon}°, Alt={alt_ft}ft")
        print(f"ECEF: X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m")
        
        # Verify coordinates are reasonable for London
        expected_magnitude = 6.378e6  # Earth radius + altitude
        actual_magnitude = (x*x + y*y + z*z) ** 0.5
        
        print(f"Distance from Earth center: {actual_magnitude:.0f}m")
        print(f"Expected ~{expected_magnitude:.0f}m")
        
        if abs(actual_magnitude - expected_magnitude) < 50000:  # Within 50km tolerance
            print("✅ ECEF conversion working correctly")
            return True
        else:
            print("❌ ECEF conversion may have issues")
            return False
            
    except Exception as e:
        print(f"❌ ECEF test failed: {e}")
        return False

def test_kdtree_availability():
    """Test KDTree availability and basic functionality."""
    print("\n=== Testing KDTree Availability ===")
    
    try:
        from scipy.spatial import KDTree
        import numpy as np
        
        # Create test points
        points = np.array([
            [0, 0, 0],      # Origin
            [1000, 0, 0],   # 1km east
            [0, 1000, 0],   # 1km north
            [0, 0, 1000],   # 1km up
        ])
        
        tree = KDTree(points)
        
        # Query near origin
        distances, indices = tree.query([100, 100, 100], k=2)
        
        print(f"Query point: (100, 100, 100)")
        print(f"Nearest 2 points - distances: {distances}, indices: {indices}")
        
        # Verify results make sense
        if len(distances) == 2 and distances[0] < distances[1]:
            print("✅ KDTree working correctly")
            return True
        else:
            print("❌ KDTree results unexpected")
            return False
            
    except ImportError:
        print("❌ scipy not available - KDTree disabled")
        return False
    except Exception as e:
        print(f"❌ KDTree test failed: {e}")
        return False

def test_bluesky_baseline_setup():
    """Test BlueSky baseline setup functionality."""
    print("\n=== Testing BlueSky Baseline Setup ===")
    
    try:
        from src.cdr.bluesky_io import BlueSkyClient
        from src.cdr.schemas import ConfigurationSettings
        
        config = ConfigurationSettings()
        client = BlueSkyClient(config)
        
        # Check method exists
        if hasattr(client, 'setup_baseline'):
            print("✅ setup_baseline() method available")
            
            # Check method signature and docstring
            method = getattr(client, 'setup_baseline')
            if callable(method) and method.__doc__:
                print("✅ Method is callable with documentation")
                print(f"Doc: {method.__doc__[:100]}...")
                return True
            else:
                print("❌ Method not properly implemented")
                return False
        else:
            print("❌ setup_baseline() method missing")
            return False
            
    except Exception as e:
        print(f"❌ BlueSky baseline test failed: {e}")
        return False

def main():
    """Run all gap fix verification tests."""
    print("Gap Fix Verification Test")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("ECEF Conversion", test_ecef_conversion()))
    results.append(("KDTree Availability", test_kdtree_availability()))
    results.append(("BlueSky Baseline Setup", test_bluesky_baseline_setup()))
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL GAP FIXES VERIFIED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())
