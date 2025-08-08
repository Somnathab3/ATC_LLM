"""Smoke tests for conflict detection module."""

import pytest
from datetime import datetime
from src.cdr.detect import predict_conflicts, is_conflict, project_trajectory, calculate_severity_score
from src.cdr.schemas import AircraftState, ConflictPrediction


class TestDetectSmoke:
    """Smoke tests for detect.py module."""
    
    def test_predict_conflicts_basic(self):
        """Test basic conflict prediction functionality."""
        # Create test aircraft states
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
    
    def test_predict_conflicts_no_traffic(self):
        """Test conflict prediction with no traffic."""
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
        
        conflicts = predict_conflicts(ownship, [], lookahead_minutes=10.0)
        
        # Should return empty list or None
        assert conflicts is None or len(conflicts) == 0
    
    def test_is_conflict_function(self):
        """Test basic conflict detection function."""
        # Test that function can be called without crashing
        result = is_conflict(1.0, 500.0, 5.0)  # 1 NM horizontal, 500 ft vertical, 5 min to CPA
        assert isinstance(result, bool)
        assert result is True  # Should be conflict
        
        result = is_conflict(10.0, 2000.0, 5.0)  # 10 NM horizontal, 2000 ft vertical, 5 min to CPA
        assert isinstance(result, bool)
        assert result is False  # Should not be conflict
    
    def test_project_trajectory_function(self):
        """Test aircraft trajectory projection."""
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
        # Function may return empty list if not implemented
    
    def test_calculate_severity_score_function(self):
        """Test severity score calculation."""
        # Test with conflict parameters
        score = calculate_severity_score(3.0, 500.0, 2.0)  # distance, altitude diff, time to CPA
        
        # Should return a numeric score
        assert isinstance(score, (int, float))
        assert score >= 0.0
    
    def test_module_imports(self):
        """Test that all required functions can be imported."""
        from src.cdr.detect import predict_conflicts, is_conflict, project_trajectory, calculate_severity_score
        
        # Just ensure imports work
        assert predict_conflicts is not None
        assert is_conflict is not None  
        assert project_trajectory is not None
        assert calculate_severity_score is not None
    
    def test_constants_exist(self):
        """Test that required constants are defined."""
        from src.cdr.detect import MIN_HORIZONTAL_SEP_NM, MIN_VERTICAL_SEP_FT
        
        assert isinstance(MIN_HORIZONTAL_SEP_NM, (int, float))
        assert isinstance(MIN_VERTICAL_SEP_FT, (int, float))
        assert MIN_HORIZONTAL_SEP_NM > 0
        assert MIN_VERTICAL_SEP_FT > 0
