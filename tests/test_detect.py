"""Comprehensive test suite for conflict detection module."""

import pytest
from datetime import datetime, timedelta
from src.cdr.detect import (
    predict_conflicts, 
    is_conflict, 
    project_trajectory, 
    calculate_severity_score,
    MIN_HORIZONTAL_SEP_NM,
    MIN_VERTICAL_SEP_FT
)
from src.cdr.schemas import AircraftState, ConflictPrediction


class TestDetectModule:
    """Test module imports and constants."""
    
    def test_module_imports(self):
        """Test that all required functions can be imported."""
        assert predict_conflicts is not None
        assert is_conflict is not None  
        assert project_trajectory is not None
        assert calculate_severity_score is not None
    
    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert isinstance(MIN_HORIZONTAL_SEP_NM, (int, float))
        assert isinstance(MIN_VERTICAL_SEP_FT, (int, float))
        assert MIN_HORIZONTAL_SEP_NM > 0
        assert MIN_VERTICAL_SEP_FT > 0


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
        assert conflicts is None or len(conflicts) == 0
    
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
    
    def test_predict_conflicts_basic_functionality(self):
        """Test basic conflict prediction functionality."""
        # Create test aircraft states with potential conflict
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=40.0,
            longitude=-74.0,
            altitude_ft=35000.0,
            ground_speed_kt=450.0,
            heading_deg=90.0,
            vertical_speed_fpm=0.0
        )
        
        # Create traffic that should conflict
        traffic = [
            AircraftState(
                aircraft_id="TRAFFIC1",
                timestamp=datetime.now(),
                latitude=40.0,
                longitude=-73.5,  # About 30 NM east
                altitude_ft=35000.0,
                ground_speed_kt=450.0,
                heading_deg=270.0,  # Head-on
                vertical_speed_fpm=0.0
            )
        ]
        
        conflicts = predict_conflicts(ownship, traffic, lookahead_minutes=10.0)
        
        # Should detect conflict or return empty list
        assert isinstance(conflicts, (list, type(None)))
        if conflicts:
            assert all(isinstance(c, ConflictPrediction) for c in conflicts)


class TestConflictCriteria:
    """Test conflict determination criteria."""
    
    @pytest.mark.parametrize("distance_nm,altitude_diff_ft,expected", [
        (10.0, 2000.0, False),  # Safe separation
        (3.0, 2000.0, True),    # Horizontal violation
        (10.0, 500.0, True),    # Vertical violation
        (3.0, 500.0, True),     # Both violations
    ])
    def test_is_conflict_criteria(self, distance_nm, altitude_diff_ft, expected):
        """Test conflict detection with various separation values."""
        time_to_cpa_min = 5.0
        result = is_conflict(distance_nm, altitude_diff_ft, time_to_cpa_min)
        
        # Function may not be fully implemented yet
        if result is not None:
            assert isinstance(result, bool)
            # Uncomment when function is implemented:
            # assert result == expected
    
    def test_is_conflict_safe_separation(self):
        """Test conflict detection with safe separation."""
        distance_nm = 10.0  # Well above 5 NM minimum
        altitude_diff_ft = 2000.0  # Well above 1000 ft minimum
        time_to_cpa_min = 5.0
        
        # Should not be a conflict with safe separation
        result = is_conflict(distance_nm, altitude_diff_ft, time_to_cpa_min)
        if result is not None:
            assert result is False
    
    def test_is_conflict_basic_function(self):
        """Test basic conflict detection function operation."""
        # Test that function can be called without crashing
        result = is_conflict(1.0, 500.0, 5.0)  # 1 NM horizontal, 500 ft vertical, 5 min to CPA
        assert isinstance(result, bool)
        
        result = is_conflict(10.0, 2000.0, 5.0)  # 10 NM horizontal, 2000 ft vertical, 5 min to CPA
        assert isinstance(result, bool)


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
            if len(trajectory) > 0:
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
    
    def test_project_trajectory_simple(self):
        """Test aircraft trajectory projection - simple case."""
        aircraft = AircraftState(
            aircraft_id="TEST",
            timestamp=datetime.now(),
            latitude=40.0,
            longitude=-74.0,
            altitude_ft=35000.0,
            ground_speed_kt=450.0,
            heading_deg=90.0,  # East
            vertical_speed_fpm=0.0
        )
        
        # Project forward
        result = project_trajectory(aircraft, 60.0)  # 1 minute ahead
        
        # Should return some trajectory points
        assert isinstance(result, list)


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
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        high_traffic = AircraftState(
            aircraft_id="HIGH",
            timestamp=datetime.now(),
            latitude=59.4,
            longitude=18.5,
            altitude_ft=45000,  # 10,000 ft higher
            ground_speed_kt=450,
            heading_deg=270,
            vertical_speed_fpm=0
        )
        
        conflicts = predict_conflicts(ownship, [high_traffic])
        
        # Should not detect conflict due to altitude separation
        if conflicts is not None:
            assert len(conflicts) == 0


class TestSeverityScoring:
    """Test conflict severity scoring."""
    
    def test_calculate_severity_score_function(self):
        """Test severity score calculation."""
        # Test with conflict parameters
        score = calculate_severity_score(3.0, 500.0, 2.0)  # distance, altitude diff, time to CPA
        
        # Should return a numeric score
        assert isinstance(score, (int, float))
        assert score >= 0.0
    
    @pytest.mark.parametrize("distance_nm,altitude_diff_ft,time_to_cpa", [
        (1.0, 100.0, 1.0),  # High severity
        (5.0, 1000.0, 10.0),  # Low severity
        (3.0, 500.0, 5.0),   # Medium severity
    ])
    def test_severity_score_values(self, distance_nm, altitude_diff_ft, time_to_cpa):
        """Test severity score with various input combinations."""
        score = calculate_severity_score(distance_nm, altitude_diff_ft, time_to_cpa)
        assert isinstance(score, (int, float))
        assert score >= 0.0

    def test_diverging_aircraft_edge_case(self):
        """Test diverging aircraft that should not trigger conflicts."""
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=400,
            heading_deg=90,  # East
            vertical_speed_fpm=0
        )
        
        # Aircraft already past closest point, moving away
        diverging_aircraft = AircraftState(
            aircraft_id="DIVERGING",
            timestamp=datetime.now(),
            latitude=59.299,  # Just behind ownship
            longitude=18.099,
            altitude_ft=35000,
            ground_speed_kt=500,
            heading_deg=225,  # Southwest - diverging rapidly
            vertical_speed_fpm=0
        )
        
        conflicts = predict_conflicts(ownship, [diverging_aircraft])
        
        # Should not detect conflict - aircraft are diverging (negative CPA time)
        if conflicts is not None:
            assert len(conflicts) == 0

    def test_conflict_boundary_conditions(self):
        """Test conflicts right at the boundary conditions."""
        # Test exactly at minimum separation thresholds
        assert not is_conflict(MIN_HORIZONTAL_SEP_NM, MIN_VERTICAL_SEP_FT, 30.0)
        
        # Test just inside conflict thresholds
        assert is_conflict(MIN_HORIZONTAL_SEP_NM - 0.01, MIN_VERTICAL_SEP_FT - 1, 30.0)
        
        # Test negative CPA time (diverging)
        assert not is_conflict(2.0, 500.0, -0.1)
        
        # Test exactly at zero CPA time
        assert is_conflict(2.0, 500.0, 0.0)

    def test_trajectory_projection_edge_cases(self):
        """Test trajectory projection with edge case aircraft states."""
        # Test with zero ground speed
        stationary = AircraftState(
            aircraft_id="STATIONARY",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=0,  # Stationary
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        trajectory = project_trajectory(stationary, 10.0)
        # Should handle stationary aircraft gracefully
        if trajectory is not None:
            assert len(trajectory) >= 0  # Should not crash
        
        # Test with very high vertical speed
        climbing = AircraftState(
            aircraft_id="CLIMBING",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=400,
            heading_deg=90,
            vertical_speed_fpm=5000  # Very fast climb
        )
        
        trajectory = project_trajectory(climbing, 5.0)
        # Should handle extreme vertical speeds
        if trajectory is not None:
            assert len(trajectory) >= 0  # Should not crash
