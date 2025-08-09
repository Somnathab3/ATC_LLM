"""Test suite for geodesy calculations."""

import pytest
import math
from src.cdr.geodesy import haversine_nm, bearing_rad, cpa_nm, cross_track_distance_nm


class TestHaversine:
    """Test haversine distance calculations."""
    
    def test_haversine_symmetry(self):
        """Test that distance calculation is symmetric."""
        a = (59.3, 18.1)
        b = (59.4, 18.3)
        assert abs(haversine_nm(a, b) - haversine_nm(b, a)) < 1e-6
    
    def test_haversine_zero_distance(self):
        """Test distance between identical points is zero."""
        a = (59.3, 18.1)
        assert haversine_nm(a, a) < 1e-10
    
    def test_haversine_known_distance(self):
        """Test against known distance calculation."""
        # Stockholm to Gothenburg (approximately)
        stockholm = (59.3293, 18.0686)
        gothenburg = (57.7089, 11.9746)
        
        distance = haversine_nm(stockholm, gothenburg)
        
        # Expected distance is approximately 214.3 nautical miles (great-circle distance)
        assert distance == pytest.approx(214.3, abs=2.0)
    
    def test_haversine_equator_distance(self):
        """Test distance calculation along equator."""
        # 1 degree of longitude at equator ~= 60 NM
        a = (0.0, 0.0)
        b = (0.0, 1.0)
        
        distance = haversine_nm(a, b)
        assert 59 < distance < 61  # Allow small tolerance
    
    def test_haversine_meridian_distance(self):
        """Test distance calculation along meridian."""
        # 1 degree of latitude ~= 60 NM
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        
        distance = haversine_nm(a, b)
        assert 59 < distance < 61  # Allow small tolerance


class TestBearing:
    """Test bearing calculations."""
    
    def test_bearing_north(self):
        """Test bearing calculation for northward direction."""
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        
        bearing = bearing_rad(a, b)
        assert abs(bearing - 0.0) < 1e-6  # 0 radians = North
    
    def test_bearing_east(self):
        """Test bearing calculation for eastward direction."""
        a = (0.0, 0.0)
        b = (0.0, 1.0)
        
        bearing = bearing_rad(a, b)
        assert abs(bearing - math.pi/2) < 1e-6  # pi/2 radians = East
    
    def test_bearing_south(self):
        """Test bearing calculation for southward direction."""
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        
        bearing = bearing_rad(a, b)
        assert abs(abs(bearing) - math.pi) < 1e-6  # +/-pi radians = South
    
    def test_bearing_west(self):
        """Test bearing calculation for westward direction."""
        a = (0.0, 1.0)
        b = (0.0, 0.0)
        
        bearing = bearing_rad(a, b)
        assert abs(bearing - (-math.pi/2)) < 1e-6  # -pi/2 radians = West


class TestCPA:
    """Test Closest Point of Approach calculations."""
    
    def test_cpa_basic_converging(self):
        """Test CPA for basic converging scenario."""
        own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East
        intr = {"lat": 0.5, "lon": 1.0, "spd_kt": 460, "hdg_deg": 270}  # West
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Should converge (time > 0, distance finite)
        assert tmin >= 0
        assert dmin >= 0
        assert dmin < 100  # Should get reasonably close
    
    def test_cpa_parallel_same_direction(self):
        """Test CPA for parallel aircraft, same direction."""
        own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East
        intr = {"lat": 0.1, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East, parallel
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Parallel flight paths - distance should remain constant
        assert tmin == 0.0  # CPA is now
        assert dmin > 0  # Non-zero separation
    
    def test_cpa_parallel_opposite_direction(self):
        """Test CPA for parallel aircraft, opposite directions."""
        own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East
        intr = {"lat": 0.1, "lon": 1.0, "spd_kt": 480, "hdg_deg": 270}  # West
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Should pass each other
        assert tmin > 0
        assert dmin >= 0
    
    def test_cpa_head_on_collision(self):
        """Test CPA for head-on scenario."""
        own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East
        intr = {"lat": 0, "lon": 1.0, "spd_kt": 480, "hdg_deg": 270}  # West, head-on
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Should meet in the middle
        assert tmin > 0
        assert dmin < 1.0  # Very close approach
    
    def test_cpa_diverging(self):
        """Test CPA for diverging aircraft."""
        own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # East
        intr = {"lat": 0, "lon": -1.0, "spd_kt": 480, "hdg_deg": 180}  # South
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Already diverging - CPA is now
        assert tmin == 0.0
        assert dmin > 0
    
    def test_cpa_zero_speed(self):
        """Test CPA with stationary aircraft."""
        own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}  # Moving east
        intr = {"lat": 0.1, "lon": 1.0, "spd_kt": 0, "hdg_deg": 0}  # Stationary
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Should approach the stationary aircraft
        assert tmin > 0
        assert dmin >= 0


class TestCrossTrackDistance:
    """Test cross-track distance calculations."""
    
    def test_cross_track_on_track(self):
        """Test point exactly on track line."""
        point = (0.5, 0.5)
        track_start = (0, 0)
        track_end = (1, 1)
        
        distance = cross_track_distance_nm(point, track_start, track_end)
        
        # Point on track should have zero cross-track distance
        assert abs(distance) < 1.0  # Small tolerance for numerical precision
    
    def test_cross_track_right_of_track(self):
        """Test point to the right of track."""
        point = (0, 1)
        track_start = (0, 0)
        track_end = (1, 0)  # Track going east
        
        distance = cross_track_distance_nm(point, track_start, track_end)
        
        # Point should be to the right (positive distance)
        assert distance > 0
        assert abs(distance - 60) < 5  # ~60 NM for 1deg latitude difference
    
    def test_cross_track_left_of_track(self):
        """Test point to the left of track."""
        point = (0, -1)
        track_start = (0, 0)
        track_end = (1, 0)  # Track going east
        
        distance = cross_track_distance_nm(point, track_start, track_end)
        
        # Point should be to the left (negative distance)
        assert distance < 0
        assert abs(distance + 60) < 5  # ~-60 NM for 1deg latitude difference


class TestGeodecyEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_haversine_antipodal_points(self):
        """Test distance between antipodal points."""
        north_pole = (90, 0)
        south_pole = (-90, 0)
        
        distance = haversine_nm(north_pole, south_pole)
        
        # Half circumference of Earth in nautical miles
        expected = math.pi * 3440.065  # pi * R_NM
        assert abs(distance - expected) < 1.0
    
    def test_cpa_identical_aircraft(self):
        """Test CPA with identical aircraft states."""
        aircraft = {"lat": 59.3, "lon": 18.1, "spd_kt": 480, "hdg_deg": 90}
        
        dmin, tmin = cpa_nm(aircraft, aircraft)
        
        # Identical aircraft should have zero separation and CPA now
        assert tmin == 0.0
        assert dmin == 0.0
    
    def test_cpa_very_high_speed(self):
        """Test CPA with very high speeds."""
        own = {"lat": 0, "lon": 0, "spd_kt": 2000, "hdg_deg": 90}  # Supersonic
        intr = {"lat": 0.1, "lon": 1.0, "spd_kt": 2000, "hdg_deg": 270}
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Should still produce valid results
        assert tmin >= 0
        assert dmin >= 0
        assert not math.isnan(dmin)
        assert not math.isnan(tmin)


# Integration tests
def test_geodesy_integration():
    """Integration test combining multiple geodesy functions."""
    # Create a scenario with known geometry
    own_pos = (59.3, 18.1)  # Stockholm area
    intr_pos = (59.4, 18.3)  # Nearby position
    
    # Calculate distance between positions
    separation = haversine_nm(own_pos, intr_pos)
    
    # Calculate bearing
    bearing = bearing_rad(own_pos, intr_pos)
    
    # Create aircraft states for CPA
    own_aircraft = {
        "lat": own_pos[0], "lon": own_pos[1],
        "spd_kt": 450, "hdg_deg": math.degrees(bearing)  # Heading toward intruder
    }
    intr_aircraft = {
        "lat": intr_pos[0], "lon": intr_pos[1],
        "spd_kt": 400, "hdg_deg": math.degrees(bearing) + 180  # Heading toward ownship
    }
    
    # Calculate CPA
    dmin, tmin = cpa_nm(own_aircraft, intr_aircraft)
    
    # Verify reasonable results
    assert separation > 0
    assert -math.pi <= bearing <= math.pi
    assert tmin > 0  # Should converge
    assert dmin >= 0  # Non-negative separation
