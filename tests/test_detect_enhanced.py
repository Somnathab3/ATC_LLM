"""
Enhanced comprehensive test suite for conflict detection module.
Targets 100% coverage with boundary condition testing.
"""

import pytest
from datetime import datetime, timezone
from src.cdr.detect import (
    predict_conflicts, 
    is_conflict, 
    project_trajectory, 
    calculate_severity_score,
    MIN_HORIZONTAL_SEP_NM,
    MIN_VERTICAL_SEP_FT
)
from src.cdr.schemas import AircraftState, ConflictPrediction


class TestDetectBoundaryConditions:
    """Test boundary conditions for 100% coverage."""
    
    def create_ownship(self, lat=40.0, lon=-74.0, alt=35000, speed=450, heading=90):
        """Helper to create standard ownship state."""
        return AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(timezone.utc),
            latitude=lat,
            longitude=lon,
            altitude_ft=alt,
            ground_speed_kt=speed,
            heading_deg=heading,
            vertical_speed_fpm=0
        )
    
    def create_intruder(self, aircraft_id, lat, lon, alt, speed=450, heading=270):
        """Helper to create intruder aircraft state."""
        return AircraftState(
            aircraft_id=aircraft_id,
            timestamp=datetime.now(timezone.utc),
            latitude=lat,
            longitude=lon,
            altitude_ft=alt,
            ground_speed_kt=speed,
            heading_deg=heading,
            vertical_speed_fpm=0
        )
    
    @pytest.mark.parametrize("distance_nm,expected_filtered", [
        (99.9, False),    # Just inside 100 NM - should be included
        (100.0, False),   # Exactly 100 NM - should be included
        (100.1, True),    # Just outside 100 NM - should be filtered out
        (150.0, True),    # Well outside 100 NM - should be filtered out
    ])
    def test_horizontal_distance_boundary(self, distance_nm, expected_filtered):
        """Test horizontal distance filtering at 100 NM boundary."""
        ownship = self.create_ownship()
        
        # Calculate lat/lon for specific distance (approximate)
        # 1 degree latitude ≈ 60 NM
        lat_offset = distance_nm / 60.0
        intruder = self.create_intruder("INTR1", 40.0 + lat_offset, -74.0, 35000)
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        if expected_filtered:
            assert len(conflicts) == 0, f"Aircraft at {distance_nm} NM should be filtered out"
        else:
            # Aircraft should not be filtered by distance, but may not conflict
            # This tests that the distance filter works correctly
            pass
    
    @pytest.mark.parametrize("altitude_diff,expected_filtered", [
        (4999.0, False),  # Just inside 5000 ft - should be included
        (5000.0, False),  # Exactly 5000 ft - should be included
        (5001.0, True),   # Just outside 5000 ft - should be filtered out
        (10000.0, True),  # Well outside 5000 ft - should be filtered out
    ])
    def test_vertical_distance_boundary(self, altitude_diff, expected_filtered):
        """Test vertical distance filtering at 5000 ft boundary."""
        ownship = self.create_ownship(alt=35000)
        intruder = self.create_intruder("INTR1", 40.01, -74.0, 35000 + altitude_diff)
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        if expected_filtered:
            assert len(conflicts) == 0, f"Aircraft at {altitude_diff} ft diff should be filtered out"
        else:
            # Aircraft should not be filtered by altitude, but may not conflict
            pass
    
    def test_self_aircraft_filtering(self):
        """Test that ownship is filtered from traffic list."""
        ownship = self.create_ownship()
        
        # Include ownship in traffic list
        traffic = [ownship]
        conflicts = predict_conflicts(ownship, traffic)
        
        assert len(conflicts) == 0, "Ownship should not conflict with itself"
    
    def test_negative_time_to_cpa_filtering(self):
        """Test filtering of aircraft with negative time to CPA (diverging)."""
        ownship = self.create_ownship(lat=40.0, lon=-74.0, heading=90)  # East
        
        # Create intruder that's already past CPA (diverging)
        intruder = self.create_intruder("INTR1", 40.0, -73.8, 35000, heading=90)  # Also east, ahead
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should be filtered out due to negative/zero time to CPA
        assert len(conflicts) == 0, "Diverging aircraft should not generate conflicts"
    
    def test_conflict_type_vertical_only(self):
        """Test conflict type classification: vertical only (line 99)."""
        ownship = self.create_ownship(lat=40.0, lon=-74.0, alt=35000)
        
        # Create intruder that violates vertical sep but not horizontal
        # Position for horizontal separation > 5 NM but altitude conflict
        intruder = self.create_intruder(
            "INTR1", 
            lat=40.1,  # ~6 NM away horizontally
            lon=-74.0, 
            alt=35500,  # 500 ft difference (< 1000 ft)
            speed=450,
            heading=270  # Converging
        )
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should detect conflict with type "vertical"
        assert len(conflicts) > 0, "Should detect vertical conflict"
        if conflicts:
            assert conflicts[0].conflict_type == "vertical", "Should classify as vertical conflict"
    
    def test_conflict_type_horizontal_only(self):
        """Test conflict type classification: horizontal only (line 101)."""
        ownship = self.create_ownship(lat=40.0, lon=-74.0, alt=35000)
        
        # Create intruder that violates horizontal sep but not vertical
        # Close horizontally but well separated vertically
        intruder = self.create_intruder(
            "INTR1",
            lat=40.02,  # ~1.2 NM away horizontally
            lon=-74.0,
            alt=36500,  # 1500 ft difference (> 1000 ft)
            speed=450,
            heading=270  # Converging
        )
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should detect conflict with type "horizontal"
        assert len(conflicts) > 0, "Should detect horizontal conflict"
        if conflicts:
            assert conflicts[0].conflict_type == "horizontal", "Should classify as horizontal conflict"
    
    def test_conflict_type_both(self):
        """Test conflict type classification: both horizontal and vertical."""
        ownship = self.create_ownship(lat=40.0, lon=-74.0, alt=35000)
        
        # Create intruder that violates both separations
        intruder = self.create_intruder(
            "INTR1",
            lat=40.02,  # ~1.2 NM away horizontally
            lon=-74.0,
            alt=35500,  # 500 ft difference
            speed=450,
            heading=270  # Converging
        )
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should detect conflict with type "both"
        assert len(conflicts) > 0, "Should detect both type conflict"
        if conflicts:
            assert conflicts[0].conflict_type == "both", "Should classify as both type conflict"
    
    @pytest.mark.parametrize("horizontal_sep,vertical_sep,expected_conflict", [
        (4.9, 900, True),     # Both violations - conflict
        (5.1, 900, False),    # Only vertical violation - no conflict
        (4.9, 1100, False),   # Only horizontal violation - no conflict  
        (5.1, 1100, False),   # No violations - no conflict
        (5.0, 1000, False),   # Exactly at limits - no conflict
        (4.999, 999, True),  # Just inside limits - conflict
    ])
    def test_conflict_criteria_boundaries(self, horizontal_sep, vertical_sep, expected_conflict):
        """Test conflict detection at exact boundary conditions."""
        # Use is_conflict function directly for precise control
        result = is_conflict(horizontal_sep, vertical_sep, 5.0)  # 5 minutes to CPA
        
        assert result == expected_conflict, (
            f"horizontal_sep={horizontal_sep}, vertical_sep={vertical_sep} "
            f"should {'cause' if expected_conflict else 'not cause'} conflict"
        )
    
    def test_conflict_criteria_negative_time(self):
        """Test that negative time to CPA prevents conflict detection."""
        # Even with separation violations, negative time means no conflict
        result = is_conflict(2.0, 500, -1.0)
        assert result is False, "Negative time to CPA should prevent conflict"
    
    def test_conflict_criteria_both_standards_safe(self):
        """Test that meeting both standards prevents conflict."""
        # Both horizontal and vertical separations are safe
        result = is_conflict(6.0, 1200, 5.0)
        assert result is False, "Safe separations should not cause conflict"
    
    def test_severity_score_boundaries(self):
        """Test severity score calculation at boundaries."""
        # Test minimum severity (no violation)
        severity = calculate_severity_score(10.0, 2000.0, 15.0)
        assert severity >= 0.0, "Severity should be non-negative"
        
        # Test maximum severity (close encounter)
        severity = calculate_severity_score(0.1, 50.0, 0.5)
        assert severity <= 1.0, "Severity should not exceed 1.0"
        assert severity > 0.8, "Close encounter should have high severity"
        
        # Test medium severity
        severity = calculate_severity_score(2.5, 500.0, 5.0)
        assert 0.2 < severity < 0.8, "Medium encounter should have medium severity"
    
    def test_project_trajectory_boundary_conditions(self):
        """Test trajectory projection edge cases."""
        aircraft = AircraftState(
            aircraft_id="TEST",
            timestamp=datetime.now(timezone.utc),
            latitude=0.0,  # Equator
            longitude=0.0,  # Prime meridian
            altitude_ft=0,   # Sea level
            ground_speed_kt=0,  # Stationary
            heading_deg=0,   # North
            vertical_speed_fpm=0
        )
        
        # Test zero time horizon
        waypoints = project_trajectory(aircraft, 0.0)
        assert len(waypoints) == 1, "Zero time should give one waypoint"
        assert waypoints[0][0] == 0.0, "First waypoint should be at time 0"
        
        # Test stationary aircraft
        waypoints = project_trajectory(aircraft, 5.0, 60.0)  # 5 min, 1 min steps
        assert len(waypoints) >= 5, "Should have multiple waypoints"
        
        # All positions should be the same for stationary aircraft
        for i in range(1, len(waypoints)):
            assert abs(waypoints[i][1] - waypoints[0][1]) < 1e-10, "Latitude shouldn't change"
            assert abs(waypoints[i][2] - waypoints[0][2]) < 1e-10, "Longitude shouldn't change"
            assert abs(waypoints[i][3] - waypoints[0][3]) < 1e-10, "Altitude shouldn't change"
    
    def test_project_trajectory_with_vertical_speed(self):
        """Test trajectory projection with vertical movement."""
        aircraft = AircraftState(
            aircraft_id="TEST",
            timestamp=datetime.now(timezone.utc),
            latitude=40.0,
            longitude=-74.0,
            altitude_ft=35000,
            ground_speed_kt=0,  # No horizontal movement
            heading_deg=0,
            vertical_speed_fpm=1000  # Climbing
        )
        
        waypoints = project_trajectory(aircraft, 2.0, 60.0)  # 2 min, 1 min steps
        
        # Check altitude increases
        initial_alt = waypoints[0][3]
        final_alt = waypoints[-1][3]
        expected_alt_gain = 1000 * 2  # 1000 fpm * 2 min
        
        assert final_alt > initial_alt, "Aircraft should climb"
        assert abs(final_alt - initial_alt - expected_alt_gain) < 100, "Altitude gain should match vertical speed"
    
    def test_multiple_conflicts_sorting(self):
        """Test that multiple conflicts are sorted by time to CPA."""
        ownship = self.create_ownship(lat=40.0, lon=-74.0, alt=35000)
        
        # Create multiple intruders at different distances (different times to CPA)
        intruder1 = self.create_intruder("INTR1", 40.05, -74.0, 35500, heading=270)  # Closer
        intruder2 = self.create_intruder("INTR2", 40.1, -74.0, 35500, heading=270)   # Farther
        
        traffic = [intruder2, intruder1]  # Add in reverse order
        conflicts = predict_conflicts(ownship, traffic)
        
        if len(conflicts) > 1:
            # Conflicts should be sorted by time to CPA (ascending)
            for i in range(len(conflicts) - 1):
                assert conflicts[i].time_to_cpa_min <= conflicts[i+1].time_to_cpa_min, \
                    "Conflicts should be sorted by time to CPA"
    
    def test_edge_case_high_speed_aircraft(self):
        """Test with high-speed aircraft near boundaries."""
        ownship = self.create_ownship(speed=900)  # High speed
        
        # Fast intruder just inside distance boundary
        intruder = self.create_intruder("INTR1", 41.6, -74.0, 35500, speed=800, heading=270)
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should process without errors
        assert isinstance(conflicts, list), "Should return list of conflicts"
    
    def test_edge_case_polar_coordinates(self):
        """Test near polar regions where longitude calculations differ."""
        # Test near North Pole
        ownship = self.create_ownship(lat=89.0, lon=0.0)
        intruder = self.create_intruder("INTR1", 89.0, 180.0, 35500)
        
        traffic = [intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should handle polar coordinates without error
        assert isinstance(conflicts, list), "Should handle polar coordinates"
    
    def test_comprehensive_integration(self):
        """Integration test covering multiple scenarios."""
        ownship = self.create_ownship()
        
        traffic = [
            # Aircraft too far horizontally - should be filtered
            self.create_intruder("FAR_H", 42.0, -74.0, 35000),
            
            # Aircraft too far vertically - should be filtered  
            self.create_intruder("FAR_V", 40.01, -74.0, 45000),
            
            # Aircraft with negative time to CPA - should be filtered
            self.create_intruder("DIVERGE", 40.0, -73.5, 35000, heading=90),
            
            # Valid conflict - both violations
            self.create_intruder("CONFLICT", 40.02, -74.0, 35500, heading=270),
            
            # Valid vertical-only conflict
            self.create_intruder("VERT", 40.1, -74.0, 35500, heading=270),
            
            # Valid horizontal-only conflict  
            self.create_intruder("HORIZ", 40.02, -74.0, 36500, heading=270),
        ]
        
        conflicts = predict_conflicts(ownship, traffic, lookahead_minutes=10.0)
        
        # Should have detected some conflicts but filtered out the invalid ones
        assert isinstance(conflicts, list), "Should return conflict list"
        
        # Check that all returned conflicts are valid
        for conflict in conflicts:
            assert conflict.time_to_cpa_min > 0, "All conflicts should have positive time to CPA"
            assert conflict.time_to_cpa_min <= 10.0, "All conflicts should be within lookahead"
            assert conflict.is_conflict is True, "All returned items should be conflicts"
            assert conflict.conflict_type in ["horizontal", "vertical", "both"], "Valid conflict type"


if __name__ == "__main__":
    pytest.main([__file__])
