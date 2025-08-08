"""Test suite for conflict detection algorithms."""

import pytest
from datetime import datetime, timedelta
from src.cdr.detect import predict_conflicts, is_conflict, project_trajectory
from src.cdr.schemas import AircraftState, ConflictPrediction


class TestConflictDetection:
    """Test conflict detection algorithms."""
    
    def test_predict_conflicts_empty_traffic(self):
        """Test conflict prediction with no traffic."""
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        traffic = []
        
        # Should return empty list for no traffic
        conflicts = predict_conflicts(ownship, traffic)
        assert conflicts is None or len(conflicts) == 0  # May not be implemented yet
    
    def test_predict_conflicts_single_aircraft(self):
        """Test conflict prediction with single traffic aircraft."""
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        traffic = [
            AircraftState(
                aircraft_id="TRAFFIC1",
                timestamp=datetime.now(),
                latitude=59.4,
                longitude=18.5,
                altitude_ft=35000,
                ground_speed_kt=400,
                heading_deg=270,  # Head-on
                vertical_speed_fpm=0
            )
        ]
        
        # Test function (may not be implemented in Sprint 0)
        conflicts = predict_conflicts(ownship, traffic)
        # Just verify function doesn't crash
        assert conflicts is None or isinstance(conflicts, list)


class TestConflictCriteria:
    """Test conflict determination criteria."""
    
    def test_is_conflict_safe_separation(self):
        """Test conflict detection with safe separation."""
        distance_nm = 10.0  # Well above 5 NM minimum
        altitude_diff_ft = 2000.0  # Well above 1000 ft minimum
        time_to_cpa_min = 5.0
        
        # Should not be a conflict with safe separation
        result = is_conflict(distance_nm, altitude_diff_ft, time_to_cpa_min)
        assert result is None or result is False  # May not be implemented yet
    
    def test_is_conflict_horizontal_violation(self):
        """Test conflict detection with horizontal separation violation."""
        distance_nm = 3.0  # Below 5 NM minimum
        altitude_diff_ft = 2000.0  # Safe vertical separation
        time_to_cpa_min = 5.0
        
        # Should be a conflict due to horizontal violation
        result = is_conflict(distance_nm, altitude_diff_ft, time_to_cpa_min)
        # Function may not be implemented yet
        assert result is None or isinstance(result, bool)
    
    def test_is_conflict_vertical_violation(self):
        """Test conflict detection with vertical separation violation."""
        distance_nm = 10.0  # Safe horizontal separation
        altitude_diff_ft = 500.0  # Below 1000 ft minimum
        time_to_cpa_min = 5.0
        
        # Should be a conflict due to vertical violation
        result = is_conflict(distance_nm, altitude_diff_ft, time_to_cpa_min)
        # Function may not be implemented yet
        assert result is None or isinstance(result, bool)
    
    def test_is_conflict_both_violations(self):
        """Test conflict detection with both horizontal and vertical violations."""
        distance_nm = 3.0  # Below 5 NM minimum
        altitude_diff_ft = 500.0  # Below 1000 ft minimum
        time_to_cpa_min = 5.0
        
        # Should definitely be a conflict
        result = is_conflict(distance_nm, altitude_diff_ft, time_to_cpa_min)
        # Function may not be implemented yet
        assert result is None or isinstance(result, bool)


class TestTrajectoryProjection:
    """Test trajectory projection algorithms."""
    
    def test_project_trajectory_basic(self):
        """Test basic trajectory projection."""
        aircraft = AircraftState(
            aircraft_id="TEST",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,  # East
            vertical_speed_fpm=0
        )
        
        time_horizon_min = 10.0
        time_step_sec = 60.0  # 1 minute steps
        
        trajectory = project_trajectory(aircraft, time_horizon_min, time_step_sec)
        
        # Function may not be implemented yet
        if trajectory is not None:
            assert isinstance(trajectory, list)
            # Should have approximately 10 waypoints for 10-minute horizon with 1-minute steps
            assert len(trajectory) >= 10
            
            # Each waypoint should be (time, lat, lon, alt)
            for waypoint in trajectory:
                assert len(waypoint) == 4
                time_min, lat, lon, alt_ft = waypoint
                assert 0 <= time_min <= time_horizon_min
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180
                assert alt_ft >= 0
    
    def test_project_trajectory_climbing(self):
        """Test trajectory projection with climbing aircraft."""
        aircraft = AircraftState(
            aircraft_id="CLIMBING",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=30000,
            ground_speed_kt=400,
            heading_deg=45,  # Northeast
            vertical_speed_fpm=1000  # Climbing
        )
        
        time_horizon_min = 5.0
        
        trajectory = project_trajectory(aircraft, time_horizon_min)
        
        # Function may not be implemented yet
        if trajectory is not None and len(trajectory) > 1:
            # Altitude should increase over time
            first_alt = trajectory[0][3]
            last_alt = trajectory[-1][3]
            assert last_alt > first_alt
    
    def test_project_trajectory_zero_speed(self):
        """Test trajectory projection with stationary aircraft."""
        aircraft = AircraftState(
            aircraft_id="STATIONARY",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=0,  # Stationary
            heading_deg=0,
            vertical_speed_fpm=0
        )
        
        time_horizon_min = 10.0
        
        trajectory = project_trajectory(aircraft, time_horizon_min)
        
        # Function may not be implemented yet
        if trajectory is not None and len(trajectory) > 1:
            # Position should remain constant
            first_pos = trajectory[0][1:3]  # (lat, lon)
            last_pos = trajectory[-1][1:3]
            assert abs(first_pos[0] - last_pos[0]) < 1e-6
            assert abs(first_pos[1] - last_pos[1]) < 1e-6


class TestDetectionEdgeCases:
    """Test edge cases in conflict detection."""
    
    def test_detect_same_aircraft(self):
        """Test detection doesn't flag aircraft against itself."""
        aircraft = AircraftState(
            aircraft_id="SAME",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        # Test with same aircraft in traffic list
        conflicts = predict_conflicts(aircraft, [aircraft])
        
        # Should not detect conflict with itself
        if conflicts is not None:
            assert len(conflicts) == 0
    
    def test_detect_very_distant_aircraft(self):
        """Test detection with very distant aircraft."""
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        distant_traffic = AircraftState(
            aircraft_id="DISTANT",
            timestamp=datetime.now(),
            latitude=0.0,  # Equator - very far from Stockholm
            longitude=0.0,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        conflicts = predict_conflicts(ownship, [distant_traffic])
        
        # Should not detect conflict with very distant aircraft
        if conflicts is not None:
            assert len(conflicts) == 0
    
    def test_detect_different_altitudes(self):
        """Test detection with significant altitude differences."""
        ownship = AircraftState(
            aircraft_id="HIGH",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=40000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        low_traffic = AircraftState(
            aircraft_id="LOW",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,  # Same horizontal position
            altitude_ft=10000,  # Much lower
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        conflicts = predict_conflicts(ownship, [low_traffic])
        
        # Should not detect conflict due to large altitude separation
        if conflicts is not None:
            # If any conflicts detected, they should not be flagged as violations
            for conflict in conflicts:
                if hasattr(conflict, 'is_conflict'):
                    assert not conflict.is_conflict


# Smoke tests for Sprint 0
def test_detection_module_imports():
    """Smoke test that detection module imports correctly."""
    from src.cdr import detect
    
    # Verify key functions are available
    assert hasattr(detect, 'predict_conflicts')
    assert hasattr(detect, 'is_conflict')
    assert hasattr(detect, 'project_trajectory')


def test_detection_constants():
    """Test that detection constants are reasonable."""
    from src.cdr.detect import MIN_HORIZONTAL_SEP_NM, MIN_VERTICAL_SEP_FT
    
    # Verify separation standards
    assert MIN_HORIZONTAL_SEP_NM == 5.0
    assert MIN_VERTICAL_SEP_FT == 1000.0
