"""Integration test for enhanced CPA and minimum separation verification.

This test verifies that the enhanced CPA functions integrate correctly
with the existing conflict detection system.
"""

import pytest
from datetime import datetime
from src.cdr.schemas import AircraftState
from src.cdr.enhanced_cpa import (
    calculate_enhanced_cpa, check_minimum_separation, 
    calculate_adaptive_cadence
)
from src.cdr.detect import predict_conflicts_enhanced

def create_aircraft(aircraft_id: str, lat: float, lon: float, alt_ft: float, 
                   speed_kt: float, heading_deg: float) -> AircraftState:
    """Helper to create aircraft state for testing."""
    return AircraftState(
        aircraft_id=aircraft_id,
        timestamp=datetime.now(),
        latitude=lat,
        longitude=lon,
        altitude_ft=alt_ft,
        ground_speed_kt=speed_kt,
        heading_deg=heading_deg,
        vertical_speed_fpm=0.0,
        aircraft_type="B737",
        spawn_offset_min=0.0
    )

class TestEnhancedCPA:
    """Test enhanced CPA calculations."""
    
    def test_head_on_collision_cpa(self):
        """Test CPA for head-on collision scenario."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)   # East
        intruder = create_aircraft("INTRUDER", 40.0, -73.5, 35000, 480, 270) # West
        
        cpa_result = calculate_enhanced_cpa(ownship, intruder)
        
        # Should converge (head-on)
        assert cpa_result.time_to_cpa_min > 0
        assert cpa_result.is_converging
        assert cpa_result.distance_at_cpa_nm < 5.0  # Should be close approach
        assert cpa_result.confidence > 0.5  # Should have reasonable confidence
    
    def test_parallel_traffic_cpa(self):
        """Test CPA for parallel traffic."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)    # East
        intruder = create_aircraft("INTRUDER", 40.1, -74.0, 37000, 480, 90)  # East, parallel
        
        cpa_result = calculate_enhanced_cpa(ownship, intruder)
        
        # Parallel traffic should not be converging
        assert not cpa_result.is_converging or cpa_result.convergence_rate_nm_min >= -0.1
        assert cpa_result.confidence > 0.5

class TestMinimumSeparation:
    """Test minimum separation verification."""
    
    def test_safe_separation(self):
        """Test aircraft with safe separation."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        intruder = create_aircraft("INTRUDER", 40.2, -73.9, 37000, 480, 90)  # Safe distance
        
        min_sep = check_minimum_separation(ownship, intruder)
        
        assert not min_sep.is_conflict
        assert min_sep.horizontal_sep_nm > 5.0  # Above minimum
        assert min_sep.vertical_sep_ft > 1000.0  # Above minimum
        assert min_sep.margin_horizontal_nm > 0
        assert min_sep.margin_vertical_ft > 0
    
    def test_horizontal_violation(self):
        """Test horizontal separation violation."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        intruder = create_aircraft("INTRUDER", 40.0, -73.97, 35000, 480, 90)  # Too close horizontally
        
        min_sep = check_minimum_separation(ownship, intruder)
        
        assert min_sep.horizontal_violation
        assert min_sep.vertical_violation  # Same altitude
        assert min_sep.is_conflict  # Both violations = conflict
        assert min_sep.horizontal_sep_nm < 5.0
        assert min_sep.vertical_sep_ft < 1000.0

class TestAdaptiveCadence:
    """Test adaptive polling cadence calculations."""
    
    def test_no_traffic_sparse_interval(self):
        """Test sparse interval with no traffic."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        traffic = []
        
        interval = calculate_adaptive_cadence(ownship, traffic, [])
        
        assert interval == 5.0  # Should use sparse interval
    
    def test_distant_traffic_sparse_interval(self):
        """Test sparse interval with distant traffic."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        distant_traffic = create_aircraft("DISTANT", 41.0, -72.0, 37000, 480, 180)
        traffic = [distant_traffic]
        
        interval = calculate_adaptive_cadence(ownship, traffic, [])
        
        assert interval >= 2.0  # Should be normal or sparse
    
    def test_close_traffic_urgent_interval(self):
        """Test urgent interval with close traffic."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        close_traffic = create_aircraft("CLOSE", 40.1, -73.9, 35000, 480, 270)  # Close and converging
        traffic = [close_traffic]
        
        interval = calculate_adaptive_cadence(ownship, traffic, [])
        
        assert interval <= 2.0  # Should be urgent or normal
    
    def test_imminent_threat_interval(self):
        """Test imminent interval with very close threat."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        imminent_threat = create_aircraft("THREAT", 40.0, -73.98, 35000, 480, 270)  # Very close
        traffic = [imminent_threat]
        
        interval = calculate_adaptive_cadence(ownship, traffic, [])
        
        assert interval <= 1.0  # Should be urgent or imminent

class TestEnhancedConflictDetection:
    """Test enhanced conflict detection integration."""
    
    def test_enhanced_detection_with_conflicts(self):
        """Test enhanced detection finds conflicts correctly."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        threat = create_aircraft("THREAT", 40.0, -73.9, 35000, 480, 270)  # Converging
        safe_traffic = create_aircraft("SAFE", 40.2, -74.0, 37000, 480, 90)  # Safe
        
        traffic = [threat, safe_traffic]
        
        conflicts, recommended_interval = predict_conflicts_enhanced(
            ownship, traffic, lookahead_minutes=10.0, use_adaptive_cadence=True
        )
        
        # Should detect the threat but not the safe traffic
        assert len(conflicts) >= 1
        threat_conflicts = [c for c in conflicts if c.intruder_id == "THREAT"]
        assert len(threat_conflicts) == 1
        
        # Should recommend urgent interval due to close threat
        assert recommended_interval <= 2.0
    
    def test_enhanced_detection_no_conflicts(self):
        """Test enhanced detection with no conflicts."""
        ownship = create_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
        safe_traffic = [
            create_aircraft("SAFE1", 40.2, -74.0, 37000, 480, 90),  # Safe parallel
            create_aircraft("SAFE2", 41.0, -72.0, 37000, 480, 180),  # Distant
        ]
        
        conflicts, recommended_interval = predict_conflicts_enhanced(
            ownship, safe_traffic, lookahead_minutes=10.0, use_adaptive_cadence=True
        )
        
        # Should detect no conflicts
        assert len(conflicts) == 0
        
        # Should recommend longer interval for safe scenario
        assert recommended_interval >= 2.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
