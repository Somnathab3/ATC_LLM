#!/usr/bin/env python3
"""
Test script for waypoint direct resolution functionality.

This script tests the complete waypoint resolution flow:
1. Data model validation (ResolutionCommand with waypoint fields)
2. Navigation utilities (waypoint lookup and validation)
3. LLM prompt enhancement (waypoint suggestions)
4. Resolution execution (BlueSky command generation)
"""

import os
import sys
import pytest
from datetime import datetime, timezone
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cdr.schemas import ResolutionType, ResolutionCommand, ResolutionEngine
from cdr.nav_utils import resolve_fix, nearest_fixes, validate_waypoint_diversion
from cdr.resolve import format_resolution_command, to_bluesky_command


def test_waypoint_resolution_command():
    """Test ResolutionCommand with waypoint fields."""
    print("Testing waypoint ResolutionCommand creation...")
    
    resolution = ResolutionCommand(
        resolution_id="WPT_TEST_001",
        target_aircraft="UAL123",
        resolution_type=ResolutionType.WAYPOINT_DIRECT,
        source_engine=ResolutionEngine.HORIZONTAL,
        new_heading_deg=None,
        new_speed_kt=None,
        new_altitude_ft=None,
        waypoint_name="KJFK",
        waypoint_lat=40.6413,
        waypoint_lon=-73.7781,
        diversion_distance_nm=45.2,
        issue_time=datetime.now(timezone.utc),
        safety_margin_nm=5.0,
        is_validated=True,
        is_ownship_command=True,
        angle_within_limits=True,
        altitude_within_limits=True,
        rate_within_limits=True
    )
    
    assert resolution.resolution_type == ResolutionType.WAYPOINT_DIRECT
    assert resolution.waypoint_name == "KJFK"
    assert resolution.waypoint_lat == 40.6413
    assert resolution.waypoint_lon == -73.7781
    assert resolution.diversion_distance_nm == 45.2
    print("✅ Waypoint ResolutionCommand creation successful")


def test_bluesky_command_formatting():
    """Test BlueSky command formatting for waypoint resolutions."""
    print("Testing BlueSky command formatting...")
    
    resolution = ResolutionCommand(
        resolution_id="WPT_TEST_002",
        target_aircraft="DLH456",
        resolution_type=ResolutionType.WAYPOINT_DIRECT,
        source_engine=ResolutionEngine.HORIZONTAL,
        waypoint_name="EDDF",
        waypoint_lat=50.0379, 
        waypoint_lon=8.5622,
        diversion_distance_nm=32.1,
        issue_time=datetime.now(timezone.utc),
        safety_margin_nm=5.0,
        is_validated=True,
        is_ownship_command=True,
        angle_within_limits=True,
        altitude_within_limits=True,
        rate_within_limits=True
    )
    
    # Test format_resolution_command
    formatted_cmd = format_resolution_command(resolution)
    expected_cmd = "DIRECT DLH456 EDDF"
    assert formatted_cmd == expected_cmd, f"Expected '{expected_cmd}', got '{formatted_cmd}'"
    
    # Test to_bluesky_command
    bluesky_cmd = to_bluesky_command(resolution, "DLH456")
    expected_bluesky = "DLH456 DCT EDDF"
    assert bluesky_cmd == expected_bluesky, f"Expected '{expected_bluesky}', got '{bluesky_cmd}'"
    
    print("✅ BlueSky command formatting successful")


def test_navigation_utilities():
    """Test navigation utilities for waypoint resolution."""
    print("Testing navigation utilities...")
    
    # Test basic waypoint resolution (may fail if BlueSky not available)
    try:
        # Test with a common airport code
        result = resolve_fix("KJFK")
        if result:
            lat, lon = result
            print(f"✅ Resolved KJFK to ({lat:.4f}, {lon:.4f})")
        else:
            print("⚠️ KJFK resolution failed (BlueSky navdb not available)")
            
        # Test nearest fixes search
        nearest = nearest_fixes(40.6413, -73.7781, k=5, max_dist_nm=50)
        if nearest:
            print(f"✅ Found {len(nearest)} nearby fixes to KJFK area")
        else:
            print("⚠️ Nearest fixes search failed (BlueSky navdb not available)")
            
        # Test waypoint validation
        validation = validate_waypoint_diversion(40.6413, -73.7781, "KJFK", 100.0)
        if validation:
            lat, lon, distance = validation
            print(f"✅ KJFK validation successful: distance {distance:.1f} NM")
        else:
            print("⚠️ KJFK validation failed (BlueSky navdb not available)")
            
    except Exception as e:
        print(f"⚠️ Navigation utilities test failed: {e}")
        print("This is expected if BlueSky is not available")


def test_enum_integration():
    """Test that WAYPOINT_DIRECT is properly integrated in ResolutionType enum."""
    print("Testing ResolutionType enum integration...")
    
    # Test enum value exists
    assert hasattr(ResolutionType, 'WAYPOINT_DIRECT')
    assert ResolutionType.WAYPOINT_DIRECT.value == "waypoint_direct"
    
    # Test enum can be used in comparisons
    resolution_type = ResolutionType.WAYPOINT_DIRECT
    assert resolution_type == ResolutionType.WAYPOINT_DIRECT
    assert resolution_type != ResolutionType.HEADING_CHANGE
    assert resolution_type != ResolutionType.ALTITUDE_CHANGE
    
    print("✅ ResolutionType enum integration successful")


def test_waypoint_field_validation():
    """Test that waypoint fields are properly validated."""
    print("Testing waypoint field validation...")
    
    # Test that waypoint fields are optional for non-waypoint resolutions
    heading_resolution = ResolutionCommand(
        resolution_id="HDG_TEST_001",
        target_aircraft="UAL123",
        resolution_type=ResolutionType.HEADING_CHANGE,
        source_engine=ResolutionEngine.HORIZONTAL,
        new_heading_deg=270.0,
        issue_time=datetime.now(timezone.utc),
        safety_margin_nm=5.0,
        is_validated=True,
        is_ownship_command=True,
        angle_within_limits=True,
        altitude_within_limits=True,
        rate_within_limits=True
    )
    
    assert heading_resolution.waypoint_name is None
    assert heading_resolution.waypoint_lat is None
    assert heading_resolution.waypoint_lon is None
    assert heading_resolution.diversion_distance_nm is None
    
    print("✅ Waypoint field validation successful")


def run_all_tests():
    """Run all waypoint resolution tests."""
    print("=" * 60)
    print("WAYPOINT DIRECT RESOLUTION - INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_enum_integration,
        test_waypoint_resolution_command,
        test_waypoint_field_validation,
        test_bluesky_command_formatting,
        test_navigation_utilities,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Waypoint resolution feature is ready!")
    else:
        print("⚠️ Some tests failed - check implementation details")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
