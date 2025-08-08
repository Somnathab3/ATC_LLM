"""Geodesy calculations for aviation conflict detection.

This module provides core mathematical functions for:
- Distance calculations using haversine formula
- Bearing calculations between waypoints  
- Closest Point of Approach (CPA) prediction
- Cross-track distance calculations

All calculations use nautical miles and aviation-standard coordinate systems.
"""

import math
from typing import Dict, Tuple, Union

# Earth radius in nautical miles
R_NM = 3440.065

# Type aliases for clarity
Coordinate = Tuple[float, float]  # (latitude, longitude) in degrees
Aircraft = Dict[str, Union[float, int]]  # Aircraft state dict


def haversine_nm(a: Coordinate, b: Coordinate) -> float:
    """Calculate great circle distance between two points using haversine formula.
    
    Args:
        a: First coordinate (lat, lon) in degrees
        b: Second coordinate (lat, lon) in degrees
        
    Returns:
        Distance in nautical miles
        
    Example:
        >>> haversine_nm((59.3, 18.1), (59.4, 18.3))
        8.794...
    """
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R_NM * math.asin(math.sqrt(h))


def bearing_rad(a: Coordinate, b: Coordinate) -> float:
    """Calculate initial bearing from point a to point b.
    
    Args:
        a: Starting coordinate (lat, lon) in degrees
        b: Destination coordinate (lat, lon) in degrees
        
    Returns:
        Bearing in radians (0 = North, π/2 = East)
        
    Example:
        >>> bearing_rad((0, 0), (0, 1))  # East
        1.570...
    """
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.atan2(y, x)

def normalize_heading_deg(hdg: float) -> float:
    """Normalize a heading to [0, 360)."""
    hdg %= 360.0
    return hdg if hdg >= 0.0 else hdg + 360.0

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from (lat1,lon1) to (lat2,lon2) in degrees."""
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)

    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    θ = math.degrees(math.atan2(y, x))
    return normalize_heading_deg(θ)

def destination_point_nm(lat: float, lon: float, bearing_deg_in: float, distance_nm: float):
    """
    Great-circle forward problem: from (lat,lon) go 'distance_nm' at 'bearing_deg_in'.
    Returns (lat2, lon2) in degrees.
    """
    θ = math.radians(bearing_deg_in)
    δ = distance_nm / R_NM  # angular distance
    φ1 = math.radians(lat)
    λ1 = math.radians(lon)

    sinφ2 = math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ)
    φ2 = math.asin(sinφ2)

    y = math.sin(θ) * math.sin(δ) * math.cos(φ1)
    x = math.cos(δ) - math.sin(φ1) * sinφ2
    λ2 = λ1 + math.atan2(y, x)

    return (math.degrees(φ2),
            (math.degrees(λ2) + 540.0) % 360.0 - 180.0)  # normalize lon to [-180,180]

def cpa_nm(own: Aircraft, intr: Aircraft) -> Tuple[float, float]:
    """Calculate Closest Point of Approach assuming constant velocity.
    
    Uses flat Earth approximation in local ENU coordinates around ownship.
    Suitable for 10-minute prediction horizons in most airspace.
    
    Args:
        own: Ownship state with keys: lat, lon, spd_kt, hdg_deg
        intr: Intruder state with keys: lat, lon, spd_kt, hdg_deg
        
    Returns:
        Tuple of (minimum_distance_nm, time_to_cpa_minutes)
        
    Example:
        >>> own = {"lat": 0, "lon": 0, "spd_kt": 480, "hdg_deg": 90}
        >>> intr = {"lat": 0.5, "lon": 1.0, "spd_kt": 460, "hdg_deg": 270}
        >>> dmin, tmin = cpa_nm(own, intr)
        >>> dmin > 0 and tmin >= 0
        True
    """
    # Convert to flat local ENU approximation around ownship
    # x = East, y = North in nautical miles from ownship
    def to_xy(p: Aircraft) -> Tuple[float, float]:
        lat_ref, lon_ref = math.radians(own["lat"]), math.radians(own["lon"])
        lat_p, lon_p = math.radians(p["lat"]), math.radians(p["lon"])
        
        x = R_NM * math.cos(lat_ref) * (lon_p - lon_ref)
        y = R_NM * (lat_p - lat_ref)
        return x, y
    
    # Initial positions
    xo, yo = to_xy(own)
    xi, yi = to_xy(intr)
    
    # Velocity vectors (East, North components in nm/hour)
    # Convert heading from aviation (0°=North, clockwise) to math (0°=East, counter-clockwise)
    def velocity_components(aircraft: Aircraft) -> Tuple[float, float]:
        hdg_rad = math.radians(aircraft["hdg_deg"])
        speed = aircraft["spd_kt"]
        # Aviation heading: 0° = North, 90° = East
        vx = speed * math.sin(hdg_rad)  # East component
        vy = speed * math.cos(hdg_rad)  # North component
        return vx, vy
    
    vox, voy = velocity_components(own)
    vix, viy = velocity_components(intr)
    
    # Relative velocity
    dvx, dvy = vox - vix, voy - viy
    
    # Relative position
    dx, dy = xo - xi, yo - yi
    
    # Time to CPA calculation
    dv_squared = dvx * dvx + dvy * dvy
    
    if dv_squared == 0:
        # Parallel flight paths - distance remains constant
        tmin_hr = 0.0
        dmin = math.sqrt(dx * dx + dy * dy)
    else:
        # Calculate time when relative velocity is perpendicular to relative position
        tmin_hr = max(0.0, -(dx * dvx + dy * dvy) / dv_squared)
        
        # Distance at CPA
        dx_cpa = dx + dvx * tmin_hr
        dy_cpa = dy + dvy * tmin_hr
        dmin = math.sqrt(dx_cpa * dx_cpa + dy_cpa * dy_cpa)
    
    # Convert time from hours to minutes
    tmin_min = tmin_hr * 60.0
    
    return dmin, tmin_min


def cross_track_distance_nm(point: Coordinate, track_start: Coordinate, track_end: Coordinate) -> float:
    """Calculate perpendicular distance from point to great circle track.
    
    Args:
        point: Point coordinate (lat, lon) in degrees
        track_start: Track starting point (lat, lon) in degrees  
        track_end: Track ending point (lat, lon) in degrees
        
    Returns:
        Cross-track distance in nautical miles (positive = right of track)
        
    Example:
        >>> cross_track_distance_nm((1, 0), (0, 0), (0, 1))
        60.0...
    """
    # Convert to radians
    lat1, lon1 = map(math.radians, point)
    lat2, lon2 = map(math.radians, track_start)
    lat3, lon3 = map(math.radians, track_end)
    
    # Distance from track start to point
    d13 = math.acos(math.sin(lat2) * math.sin(lat1) + 
                    math.cos(lat2) * math.cos(lat1) * math.cos(lon1 - lon2))
    
    # Bearing from track start to point
    brng23 = bearing_rad(track_start, track_end)
    brng13 = bearing_rad(track_start, point)
    
    # Cross-track distance (signed)
    dxt = math.asin(math.sin(d13) * math.sin(brng13 - brng23))
    
    return dxt * R_NM
