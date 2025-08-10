"""Test suite for geodesy calculations with comprehensive edge case coverage."""

import pytest
import math
from src.cdr.geodesy import (
    haversine_nm, bearing_rad, bearing_deg, normalize_heading_deg, 
    destination_point_nm, cpa_nm, cross_track_distance_nm
)


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


class TestGeodacticEdgeCases:
    """Test edge cases and boundary conditions for comprehensive coverage."""
    
    def test_normalize_heading_edge_cases(self):
        """Test heading normalization edge cases - covers lines 64-65."""
        # Test positive wrap-around
        assert normalize_heading_deg(370.0) == 10.0
        assert normalize_heading_deg(720.0) == 0.0
        
        # Test negative values - covers line 65
        assert normalize_heading_deg(-10.0) == 350.0
        assert normalize_heading_deg(-370.0) == 350.0
        
        # Test boundary values
        assert normalize_heading_deg(0.0) == 0.0
        assert normalize_heading_deg(359.999) == pytest.approx(359.999, abs=1e-6)
        assert normalize_heading_deg(-0.001) == pytest.approx(359.999, abs=1e-6)
    
    def test_bearing_deg_antipodal_points(self):
        """Test bearing calculation for antipodal points - covers lines 69-76."""
        # Test antipodal points (opposite sides of Earth)
        bearing1 = bearing_deg(0.0, 0.0, 0.0, 180.0)  # Equator, opposite longitude
        assert 89.0 < bearing1 < 91.0  # Should be ~90 degrees (East)
        
        bearing2 = bearing_deg(45.0, 0.0, -45.0, 180.0)  # Crossing equator
        assert 0.0 <= bearing2 <= 360.0
        
        # Test identical points (zero distance)
        bearing3 = bearing_deg(45.0, 30.0, 45.0, 30.0)
        assert 0.0 <= bearing3 <= 360.0  # Should be defined even for zero distance
    
    def test_bearing_deg_wrapping_cases(self):
        """Test bearing calculations that wrap around 0/360 degrees."""
        # Test cases that produce negative bearings before normalization
        bearing1 = bearing_deg(60.0, 10.0, 50.0, 5.0)  # Southwestward
        assert 180.0 < bearing1 < 270.0
        
        # Test extreme longitude differences - the actual result depends on great circle
        bearing2 = bearing_deg(0.0, -179.0, 0.0, 179.0)  # Cross antimeridian
        assert 0.0 <= bearing2 < 360.0  # Just ensure it's normalized
    
    def test_destination_point_boundary_cases(self):
        """Test destination point calculation edge cases - covers lines 83-95."""
        # Test zero distance
        lat, lon = destination_point_nm(45.0, 30.0, 90.0, 0.0)
        assert lat == pytest.approx(45.0, abs=1e-10)
        assert lon == pytest.approx(30.0, abs=1e-10)
        
        # Test very small distance
        lat, lon = destination_point_nm(0.0, 0.0, 45.0, 0.001)
        assert abs(lat) < 0.01
        assert abs(lon) < 0.01
        
        # Test crossing antimeridian (longitude normalization)
        lat, lon = destination_point_nm(0.0, 179.0, 90.0, 120.0)  # East from near antimeridian
        assert -180.0 <= lon <= 180.0  # Should normalize to [-180, 180]
        
        # Test pole proximity
        lat, lon = destination_point_nm(89.0, 0.0, 0.0, 60.0)  # North from near pole
        assert lat <= 90.0  # Should not exceed North pole
    
    def test_cpa_parallel_flight_paths(self):
        """Test CPA with exactly parallel flight paths - covers line 148."""
        # Exactly parallel, same speed
        own = {"lat": 0.0, "lon": 0.0, "spd_kt": 480.0, "hdg_deg": 90.0}  # East
        intr = {"lat": 0.1, "lon": 0.0, "spd_kt": 480.0, "hdg_deg": 90.0}  # East, parallel
        
        dmin, tmin = cpa_nm(own, intr)
        
        # With zero relative velocity, CPA time should be 0 and distance constant
        assert tmin == 0.0
        assert dmin > 0  # Should maintain constant separation
    
    def test_cpa_zero_relative_velocity(self):
        """Test CPA with zero relative velocity (dv_squared == 0)."""
        # Same velocity vectors but different positions
        own = {"lat": 0.0, "lon": 0.0, "spd_kt": 500.0, "hdg_deg": 45.0}
        intr = {"lat": 0.1, "lon": 0.1, "spd_kt": 500.0, "hdg_deg": 45.0}
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Zero relative velocity means constant separation
        assert tmin == 0.0
        assert dmin > 0
    
    def test_cpa_head_on_collision_course(self):
        """Test CPA for aircraft on exact collision course."""
        # Head-on at same altitude and position
        own = {"lat": 0.0, "lon": 0.0, "spd_kt": 480.0, "hdg_deg": 90.0}  # East
        intr = {"lat": 0.0, "lon": 1.0, "spd_kt": 480.0, "hdg_deg": 270.0}  # West
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Should converge to very small separation
        assert tmin > 0  # Future convergence
        assert dmin < 1.0  # Very close approach
    
    def test_cpa_past_convergence(self):
        """Test CPA when aircraft are already diverging."""
        # Aircraft that have already passed each other
        own = {"lat": 0.0, "lon": 0.0, "spd_kt": 480.0, "hdg_deg": 90.0}  # East
        intr = {"lat": 0.0, "lon": -0.5, "spd_kt": 480.0, "hdg_deg": 270.0}  # West
        
        dmin, tmin = cpa_nm(own, intr)
        
        # Time to CPA should be 0 (already at closest point or past it)
        assert tmin == 0.0
    
    def test_cross_track_distance_edge_cases(self):
        """Test cross track distance edge cases."""
        # Point exactly on track
        point = (0.5, 0.5)
        track_start = (0.0, 0.0)
        track_end = (1.0, 1.0)
        
        xtd = cross_track_distance_nm(point, track_start, track_end)
        assert abs(xtd) < 1.0  # Should be very small
        
        # Point far from track
        point = (2.0, 0.0)
        track_start = (0.0, 0.0)
        track_end = (0.0, 1.0)  # North-South track
        
        xtd = cross_track_distance_nm(point, track_start, track_end)
        assert abs(xtd) > 100.0  # Should be significant distance
    
    @pytest.mark.parametrize("heading", [0, 90, 180, 270, 359.9, 360.1])
    def test_heading_normalization_parametrized(self, heading):
        """Parametrized test for heading normalization."""
        normalized = normalize_heading_deg(heading)
        assert 0.0 <= normalized < 360.0
    
    @pytest.mark.parametrize("lat,lon", [
        (0, 0), (45, 45), (-45, -45), (89.9, 179.9), (-89.9, -179.9)
    ])
    def test_bearing_calculation_parametrized(self, lat, lon):
        """Parametrized test for bearing calculations."""
        target_lat, target_lon = lat + 0.1, lon + 0.1
        bearing = bearing_deg(lat, lon, target_lat, target_lon)
        assert 0.0 <= bearing < 360.0
