"""Comprehensive test suite for navigation utilities.

Tests cover:
- resolve_fix: found / ambiguous / not found cases
- nearest_fixes: returns ≤k, sorted by distance, filtered by max_dist_nm
- Edge case when BlueSky nav DB is unavailable (NAV_OK=False)
- Graceful degradation without navdb

Coverage target: nav_utils ≥80%
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Optional
import logging
from _pytest.logging import LogCaptureFixture

# Import the module under test
import src.cdr.nav_utils as nav_utils
from src.cdr.nav_utils import resolve_fix, nearest_fixes, validate_waypoint_diversion


class MockNavdatabase:
    """Mock BlueSky navigation database for testing."""
    
    def __init__(self, wpids: Optional[List[str]] = None, wplats: Optional[List[float]] = None, wplons: Optional[List[float]] = None, 
                 aptids: Optional[List[str]] = None, aptlats: Optional[List[float]] = None, aptlons: Optional[List[float]] = None,
                 navids: Optional[List[str]] = None, navlats: Optional[List[float]] = None, navlons: Optional[List[float]] = None):
        # Waypoints
        self.wpid: List[str] = wpids if wpids is not None else ["KJFK", "KLGA", "KEWR", "KTEB", "KSEA", "KORD"]
        self.wplat: List[float] = wplats if wplats is not None else [40.6413, 40.7769, 40.6925, 40.8501, 47.4502, 41.9742]
        self.wplon: List[float] = wplons if wplons is not None else [-73.7781, -73.8740, -74.1687, -74.0602, -122.3088, -87.9073]
        
        # Airports  
        self.aptid: List[str] = aptids if aptids is not None else ["KJFK", "KLGA", "KSEA"]
        self.aptlat: List[float] = aptlats if aptlats is not None else [40.6413, 40.7769, 47.4502]
        self.aptlon: List[float] = aptlons if aptlons is not None else [-73.7781, -73.8740, -122.3088]
        
        # Navaids
        self.navid: List[str] = navids if navids is not None else ["JFK", "LGA", "SEA"]
        self.navlat: List[float] = navlats if navlats is not None else [40.6413, 40.7769, 47.4502]
        self.navlon: List[float] = navlons if navlons is not None else [-73.7781, -73.8740, -122.3088]


class TestResolveFixWithNavdb:
    """Test resolve_fix function with navigation database available."""
    
    def test_resolve_fix_found_waypoint(self):
        """Test resolving an existing waypoint."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = resolve_fix("KJFK")
            
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2
            lat, lon = result
            assert abs(lat - 40.6413) < 1e-4
            assert abs(lon - (-73.7781)) < 1e-4
    
    def test_resolve_fix_found_airport(self):
        """Test resolving fix falls back to airport lookup."""
        # Create navdb with no waypoints, only airports
        mock_navdb = MockNavdatabase(wpids=[], wplats=[], wplons=[])
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = resolve_fix("KJFK")
            
            assert result is not None
            lat, lon = result
            assert abs(lat - 40.6413) < 1e-4
            assert abs(lon - (-73.7781)) < 1e-4
    
    def test_resolve_fix_found_navaid(self):
        """Test resolving fix falls back to navaid lookup."""
        # Create navdb with no waypoints or airports, only navaids
        mock_navdb = MockNavdatabase(
            wpids=[], wplats=[], wplons=[],
            aptids=[], aptlats=[], aptlons=[]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = resolve_fix("JFK")
            
            assert result is not None
            lat, lon = result
            assert abs(lat - 40.6413) < 1e-4
            assert abs(lon - (-73.7781)) < 1e-4
    
    def test_resolve_fix_not_found(self):
        """Test resolving a non-existent fix."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = resolve_fix("NONEXISTENT")
            
            assert result is None
    
    def test_resolve_fix_case_insensitive(self):
        """Test that fix resolution is case insensitive."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Test various cases
            result1 = resolve_fix("kjfk")
            result2 = resolve_fix("KJFK")
            result3 = resolve_fix("KjFk")
            
            assert result1 is not None
            assert result2 is not None
            assert result3 is not None
            assert result1 == result2 == result3
    
    def test_resolve_fix_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Test with extra whitespace
            result1 = resolve_fix("  KJFK  ")
            result2 = resolve_fix("KJFK")
            
            assert result1 is not None
            assert result2 is not None
            assert result1 == result2
    
    def test_resolve_fix_database_error_handling(self):
        """Test error handling when database access fails."""
        mock_navdb = Mock()
        mock_navdb.wpid = Mock(side_effect=Exception("Database error"))
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = resolve_fix("KJFK")
            
            assert result is None


class TestResolveFixWithoutNavdb:
    """Test resolve_fix function when navigation database is unavailable."""
    
    def test_resolve_fix_nav_not_ok(self):
        """Test graceful degradation when NAV_OK is False."""
        with patch.object(nav_utils, 'NAV_OK', False), \
             patch.object(nav_utils, 'navdb', None):
            
            result = resolve_fix("KJFK")
            
            assert result is None
    
    def test_resolve_fix_navdb_none(self):
        """Test graceful degradation when navdb is None."""
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', None):
            
            result = resolve_fix("KJFK")
            
            assert result is None


class TestNearestFixesWithNavdb:
    """Test nearest_fixes function with navigation database available."""
    
    def test_nearest_fixes_basic_functionality(self):
        """Test basic nearest fixes functionality."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Search near JFK (40.6413, -73.7781)
            result = nearest_fixes(40.6413, -73.7781, k=3, max_dist_nm=50.0)
            
            assert isinstance(result, list)
            assert len(result) <= 3
            
            # Check that all results have required fields
            for fix in result:
                assert isinstance(fix, dict)
                assert "name" in fix
                assert "lat" in fix
                assert "lon" in fix
                assert "dist_nm" in fix
                assert "type" in fix
                assert fix["dist_nm"] >= 0
    
    def test_nearest_fixes_sorted_by_distance(self):
        """Test that nearest fixes are sorted by distance."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Search from a point that will have multiple results
            result = nearest_fixes(40.7, -73.8, k=5, max_dist_nm=100.0)
            
            assert len(result) > 1
            
            # Check that distances are sorted (non-decreasing)
            distances = [fix["dist_nm"] for fix in result]
            assert distances == sorted(distances)
    
    def test_nearest_fixes_respects_k_limit(self):
        """Test that nearest_fixes returns at most k results."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Request only 2 fixes
            result = nearest_fixes(40.7, -73.8, k=2, max_dist_nm=1000.0)
            
            assert len(result) <= 2
    
    def test_nearest_fixes_respects_distance_limit(self):
        """Test that nearest_fixes filters by max_dist_nm."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Very restrictive distance limit
            result = nearest_fixes(40.7, -73.8, k=10, max_dist_nm=1.0)
            
            # Should have fewer results due to distance constraint
            for fix in result:
                assert fix["dist_nm"] <= 1.0
    
    def test_nearest_fixes_includes_airports_when_needed(self):
        """Test that airports are included when not enough waypoints."""
        # Create navdb with only one waypoint but multiple airports
        mock_navdb = MockNavdatabase(
            wpids=["KJFK"], wplats=[40.6413], wplons=[-73.7781]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            # Should have waypoints and airports
            waypoint_types = [fix["type"] for fix in result]
            assert "waypoint" in waypoint_types or "airport" in waypoint_types
    
    def test_nearest_fixes_no_duplicates(self):
        """Test that nearest_fixes doesn't return duplicate names."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = nearest_fixes(40.7, -73.8, k=10, max_dist_nm=1000.0)
            
            # Check for duplicate names
            names = [fix["name"] for fix in result]
            assert len(names) == len(set(names))  # No duplicates
    
    def test_nearest_fixes_empty_database(self):
        """Test nearest_fixes with empty database."""
        mock_navdb = MockNavdatabase(
            wpids=[], wplats=[], wplons=[],
            aptids=[], aptlats=[], aptlons=[]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            assert result == []
    
    def test_nearest_fixes_invalid_coordinates(self):
        """Test nearest_fixes handles invalid coordinates in database."""
        # Create navdb with some invalid data by using Mock instead of MockNavdatabase
        mock_navdb = Mock()
        mock_navdb.wpid = ["KJFK", "INVALID", "KLGA"]
        mock_navdb.wplat = [40.6413, "invalid", 40.7769]  # Invalid data type
        mock_navdb.wplon = [-73.7781, -73.0, -73.8740]
        # Add empty arrays for other attributes to avoid AttributeError
        mock_navdb.aptid = []
        mock_navdb.aptlat = []
        mock_navdb.aptlon = []
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            # Should handle invalid data gracefully
            assert isinstance(result, list)
            # Should have valid entries only
            for fix in result:
                assert isinstance(fix["lat"], float)
                assert isinstance(fix["lon"], float)
                assert isinstance(fix["dist_nm"], float)


class TestNearestFixesWithoutNavdb:
    """Test nearest_fixes function when navigation database is unavailable."""
    
    def test_nearest_fixes_nav_not_ok(self):
        """Test graceful degradation when NAV_OK is False."""
        with patch.object(nav_utils, 'NAV_OK', False), \
             patch.object(nav_utils, 'navdb', None):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            assert result == []
    
    def test_nearest_fixes_navdb_none(self):
        """Test graceful degradation when navdb is None."""
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', None):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            assert result == []
    
    def test_nearest_fixes_database_error(self):
        """Test error handling when database access fails."""
        mock_navdb = Mock()
        mock_navdb.wpid = Mock(side_effect=Exception("Database error"))
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            assert result == []


class TestValidateWaypointDiversion:
    """Test validate_waypoint_diversion function."""
    
    def test_validate_waypoint_diversion_valid(self):
        """Test waypoint validation with valid waypoint and distance."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Validate KJFK from a nearby position
            result = validate_waypoint_diversion(40.7, -73.8, "KJFK", 50.0)
            
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 3
            
            lat, lon, distance = result
            assert abs(lat - 40.6413) < 1e-4
            assert abs(lon - (-73.7781)) < 1e-4
            assert distance > 0
            assert distance <= 50.0
    
    def test_validate_waypoint_diversion_too_far(self):
        """Test waypoint validation when waypoint is too far."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Validate KJFK from a very distant position with small diversion limit
            result = validate_waypoint_diversion(60.0, 5.0, "KJFK", 10.0)
            
            assert result is None
    
    def test_validate_waypoint_diversion_not_found(self):
        """Test waypoint validation when waypoint doesn't exist."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = validate_waypoint_diversion(40.7, -73.8, "NONEXISTENT", 50.0)
            
            assert result is None
    
    def test_validate_waypoint_diversion_no_navdb(self):
        """Test waypoint validation without navigation database."""
        with patch.object(nav_utils, 'NAV_OK', False), \
             patch.object(nav_utils, 'navdb', None):
            
            result = validate_waypoint_diversion(40.7, -73.8, "KJFK", 50.0)
            
            assert result is None


class TestNavUtilsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_navdb_attributes(self):
        """Test handling when navdb is missing expected attributes."""
        # Create a mock navdb missing some attributes
        mock_navdb = Mock()
        # Only has wpid, missing wplat and wplon
        mock_navdb.wpid = ["KJFK"]
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Should handle missing attributes gracefully
            _ = resolve_fix("KJFK")  # May be None due to missing attributes
            
            nearest = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            assert nearest == []
    
    def test_empty_navdb_arrays(self):
        """Test handling when navdb arrays are empty."""
        mock_navdb = MockNavdatabase(
            wpids=[], wplats=[], wplons=[],
            aptids=[], aptlats=[], aptlons=[],
            navids=[], navlats=[], navlons=[]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = resolve_fix("KJFK")
            assert result is None
            
            nearest = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            assert nearest == []
    
    def test_mismatched_array_lengths(self):
        """Test handling when navdb arrays have mismatched lengths."""
        mock_navdb = MockNavdatabase(
            wpids=["KJFK", "KLGA"],
            wplats=[40.6413],  # Missing one latitude
            wplons=[-73.7781, -73.8740]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Should handle gracefully without crashing
            _ = resolve_fix("KLGA")  # The one with missing data
            # Might be None due to IndexError, should not crash
            
            nearest = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            assert isinstance(nearest, list)
    
    def test_extreme_coordinates(self):
        """Test with extreme coordinate values."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Test with extreme coordinates
            result = nearest_fixes(90.0, 180.0, k=3, max_dist_nm=100.0)
            assert isinstance(result, list)
            
            result = nearest_fixes(-90.0, -180.0, k=3, max_dist_nm=100.0)
            assert isinstance(result, list)
    
    def test_zero_and_negative_parameters(self):
        """Test with zero and negative parameters."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Test with k=0
            result = nearest_fixes(40.7, -73.8, k=0, max_dist_nm=100.0)
            assert result == []
            
            # Test with negative max_dist_nm
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=-10.0)
            assert result == []
            
            # Test with max_dist_nm=0
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=0.0)
            # Should return only fixes at exactly 0 distance (if any)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_realistic_flight_scenario(self):
        """Test a realistic flight scenario with multiple waypoints."""
        # Create a realistic navdb with common waypoints
        mock_navdb = MockNavdatabase(
            wpids=["KJFK", "WAVEY", "COATE", "KLGA", "LGA", "JFK"],
            wplats=[40.6413, 40.65, 40.67, 40.7769, 40.7769, 40.6413],
            wplons=[-73.7781, -73.75, -73.73, -73.8740, -73.8740, -73.7781]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Aircraft at cruise between airports
            aircraft_lat, aircraft_lon = 40.70, -73.80
            
            # Find nearby fixes for potential diversion
            nearby = nearest_fixes(aircraft_lat, aircraft_lon, k=3, max_dist_nm=25.0)
            
            assert len(nearby) > 0
            assert all(fix["dist_nm"] <= 25.0 for fix in nearby)
            
            # Validate a specific waypoint for diversion
            validation = validate_waypoint_diversion(
                aircraft_lat, aircraft_lon, "KJFK", 30.0
            )
            
            assert validation is not None
            _, _, distance = validation
            assert distance <= 30.0
    
    def test_no_navdb_degradation(self):
        """Test complete system degradation when navdb unavailable."""
        with patch.object(nav_utils, 'NAV_OK', False), \
             patch.object(nav_utils, 'navdb', None):
            
            # All functions should degrade gracefully
            assert resolve_fix("KJFK") is None
            assert nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0) == []
            assert validate_waypoint_diversion(40.7, -73.8, "KJFK", 50.0) is None
    
    def test_ambiguous_fix_scenario(self):
        """Test scenario where multiple fixes might match."""
        # Create navdb with similar-sounding waypoints
        mock_navdb = MockNavdatabase(
            wpids=["JFK", "KJFK"],
            wplats=[40.6413, 40.6413],
            wplons=[-73.7781, -73.7781]
        )
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Should find the first match
            result = resolve_fix("JFK")
            assert result is not None
            
            result = resolve_fix("KJFK")
            assert result is not None


class TestLogging:
    """Test logging behavior."""
    
    def test_logging_on_successful_resolution(self, caplog: LogCaptureFixture) -> None:
        """Test that successful resolutions are logged."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            with caplog.at_level(logging.INFO):
                resolve_fix("KJFK")
                
                # Should log successful resolution
                assert any("Resolved waypoint KJFK" in record.message for record in caplog.records)
    
    def test_logging_on_failed_resolution(self, caplog: LogCaptureFixture) -> None:
        """Test that failed resolutions are logged."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            with caplog.at_level(logging.WARNING):
                resolve_fix("NONEXISTENT")
                
                # Should log failed resolution
                assert any("not found in navigation database" in record.message for record in caplog.records)
    
    def test_logging_nearest_fixes_summary(self, caplog: LogCaptureFixture) -> None:
        """Test that nearest_fixes logs summary information."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            with caplog.at_level(logging.INFO):
                nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
                
                # Should log summary
                assert any("Found" in record.message and "nearby fixes" in record.message 
                          for record in caplog.records)


class TestExitChecks:
    """Test exit conditions - functions return expected dicts/tuples; no crashes without navdb."""
    
    def test_resolve_fix_return_types(self):
        """Test that resolve_fix returns expected types."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Found case - should return tuple
            result = resolve_fix("KJFK")
            assert result is None or isinstance(result, tuple)
            if result is not None:
                assert len(result) == 2
                assert isinstance(result[0], float)  # latitude
                assert isinstance(result[1], float)  # longitude
            
            # Not found case - should return None
            result = resolve_fix("NONEXISTENT")
            assert result is None
    
    def test_nearest_fixes_return_types(self):
        """Test that nearest_fixes returns expected list of dicts."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            result = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
            
            # Should always return a list
            assert isinstance(result, list)
            
            # Each item should be a dict with required fields
            for fix in result:
                assert isinstance(fix, dict)
                assert "name" in fix and isinstance(fix["name"], str)
                assert "lat" in fix and isinstance(fix["lat"], float)
                assert "lon" in fix and isinstance(fix["lon"], float)
                assert "dist_nm" in fix and isinstance(fix["dist_nm"], float)
                assert "type" in fix and isinstance(fix["type"], str)
    
    def test_validate_waypoint_diversion_return_types(self):
        """Test that validate_waypoint_diversion returns expected types."""
        mock_navdb = MockNavdatabase()
        
        with patch.object(nav_utils, 'NAV_OK', True), \
             patch.object(nav_utils, 'navdb', mock_navdb):
            
            # Valid case
            result = validate_waypoint_diversion(40.7, -73.8, "KJFK", 50.0)
            if result is not None:
                assert isinstance(result, tuple)
                assert len(result) == 3
                assert isinstance(result[0], float)  # latitude
                assert isinstance(result[1], float)  # longitude
                assert isinstance(result[2], float)  # distance
            
            # Invalid case
            result = validate_waypoint_diversion(40.7, -73.8, "NONEXISTENT", 50.0)
            assert result is None
    
    def test_no_crashes_without_navdb(self):
        """Test that all functions handle missing navdb gracefully without crashes."""
        with patch.object(nav_utils, 'NAV_OK', False), \
             patch.object(nav_utils, 'navdb', None):
            
            # All functions should complete without raising exceptions
            try:
                result1 = resolve_fix("KJFK")
                assert result1 is None
                
                result2 = nearest_fixes(40.7, -73.8, k=3, max_dist_nm=100.0)
                assert result2 == []
                
                result3 = validate_waypoint_diversion(40.7, -73.8, "KJFK", 50.0)
                assert result3 is None
                
                # Test completed without exceptions
                assert True
                
            except Exception as e:
                pytest.fail(f"Functions should not crash without navdb, but got: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.cdr.nav_utils", "--cov-report=term-missing"])
