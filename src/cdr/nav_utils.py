"""Navigation utilities for waypoint resolution and fix lookup.

This module provides:
- Named fix resolution using BlueSky navigation database
- Nearest fix suggestions to reduce LLM hallucination
- Distance validation for waypoint diversion limits
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from .geodesy import haversine_nm

logger = logging.getLogger(__name__)

# Try to import BlueSky navigation database
try:
    from bluesky.navdatabase import Navdatabase
    navdb = Navdatabase()
    NAV_OK = True
    logger.info("BlueSky navigation database loaded successfully")
except Exception as e:
    NAV_OK = False
    navdb = None
    logger.warning(f"BlueSky navigation database not available: {e}")


def resolve_fix(name: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for named fix if found; else None.
    
    Args:
        name: Waypoint/fix name to resolve
        
    Returns:
        (latitude, longitude) tuple if found, None otherwise
    """
    if not NAV_OK or not navdb:
        logger.warning("Navigation database not available for fix resolution")
        return None
        
    try:
        name = name.strip().upper()
        
        # Try waypoints first (most common for enroute navigation)
        if hasattr(navdb, 'wpid') and hasattr(navdb, 'wplat') and hasattr(navdb, 'wplon'):
            wp_ids = getattr(navdb, 'wpid', [])
            wp_lats = getattr(navdb, 'wplat', [])
            wp_lons = getattr(navdb, 'wplon', [])
            
            for i, wp_id in enumerate(wp_ids):
                if str(wp_id).strip().upper() == name:
                    lat = float(wp_lats[i])
                    lon = float(wp_lons[i])
                    logger.info(f"Resolved waypoint {name} to ({lat:.6f}, {lon:.6f})")
                    return (lat, lon)
        
        # Try airports/runways as backup
        if hasattr(navdb, 'aptid') and hasattr(navdb, 'aptlat') and hasattr(navdb, 'aptlon'):
            apt_ids = getattr(navdb, 'aptid', [])
            apt_lats = getattr(navdb, 'aptlat', [])
            apt_lons = getattr(navdb, 'aptlon', [])
            
            for i, apt_id in enumerate(apt_ids):
                if str(apt_id).strip().upper() == name:
                    lat = float(apt_lats[i])
                    lon = float(apt_lons[i])
                    logger.info(f"Resolved airport {name} to ({lat:.6f}, {lon:.6f})")
                    return (lat, lon)
        
        # Try navaids as last resort
        if hasattr(navdb, 'navid') and hasattr(navdb, 'navlat') and hasattr(navdb, 'navlon'):
            nav_ids = getattr(navdb, 'navid', [])
            nav_lats = getattr(navdb, 'navlat', [])
            nav_lons = getattr(navdb, 'navlon', [])
            
            for i, nav_id in enumerate(nav_ids):
                if str(nav_id).strip().upper() == name:
                    lat = float(nav_lats[i])
                    lon = float(nav_lons[i])
                    logger.info(f"Resolved navaid {name} to ({lat:.6f}, {lon:.6f})")
                    return (lat, lon)
        
        logger.warning(f"Fix '{name}' not found in navigation database")
        return None
        
    except Exception as e:
        logger.error(f"Error resolving fix '{name}': {e}")
        return None


def nearest_fixes(lat: float, lon: float, k: int = 3, max_dist_nm: float = 80.0) -> List[Dict[str, Any]]:
    """Return up to k nearby named fixes within max_dist_nm, sorted by distance.
    
    Args:
        lat: Aircraft latitude
        lon: Aircraft longitude  
        k: Maximum number of fixes to return
        max_dist_nm: Maximum distance to search
        
    Returns:
        List of dictionaries with name, lat, lon, dist_nm keys
    """
    if not NAV_OK or not navdb:
        logger.warning("Navigation database not available for nearest fixes")
        return []
    
    try:
        fixes = []
        aircraft_pos = (lat, lon)
        
        # Search waypoints first (most relevant for enroute)
        if hasattr(navdb, 'wpid') and hasattr(navdb, 'wplat') and hasattr(navdb, 'wplon'):
            wp_ids = getattr(navdb, 'wpid', [])
            wp_lats = getattr(navdb, 'wplat', [])
            wp_lons = getattr(navdb, 'wplon', [])
            
            for i, wp_id in enumerate(wp_ids):
                try:
                    wp_lat = float(wp_lats[i])
                    wp_lon = float(wp_lons[i])
                    wp_pos = (wp_lat, wp_lon)
                    distance = haversine_nm(aircraft_pos, wp_pos)
                    
                    if distance <= max_dist_nm:
                        fixes.append({
                            "name": str(wp_id).strip().upper(),
                            "lat": wp_lat,
                            "lon": wp_lon,
                            "dist_nm": distance,
                            "type": "waypoint"
                        })
                except (ValueError, IndexError):
                    continue
        
        # Add airports if we don't have enough waypoints
        if len(fixes) < k and hasattr(navdb, 'aptid') and hasattr(navdb, 'aptlat') and hasattr(navdb, 'aptlon'):
            apt_ids = getattr(navdb, 'aptid', [])
            apt_lats = getattr(navdb, 'aptlat', [])
            apt_lons = getattr(navdb, 'aptlon', [])
            
            for i, apt_id in enumerate(apt_ids):
                try:
                    apt_lat = float(apt_lats[i])
                    apt_lon = float(apt_lons[i])
                    apt_pos = (apt_lat, apt_lon)
                    distance = haversine_nm(aircraft_pos, apt_pos)
                    
                    if distance <= max_dist_nm:
                        # Avoid duplicates
                        apt_name = str(apt_id).strip().upper()
                        if not any(f["name"] == apt_name for f in fixes):
                            fixes.append({
                                "name": apt_name,
                                "lat": apt_lat,
                                "lon": apt_lon,
                                "dist_nm": distance,
                                "type": "airport"
                            })
                except (ValueError, IndexError):
                    continue
        
        # Sort by distance and return top k
        fixes.sort(key=lambda x: x["dist_nm"])
        result = fixes[:k]
        
        logger.info(f"Found {len(result)} nearby fixes within {max_dist_nm} NM of ({lat:.4f}, {lon:.4f})")
        for fix in result:
            logger.debug(f"  {fix['name']}: {fix['dist_nm']:.1f} NM ({fix['type']})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error finding nearest fixes: {e}")
        return []


def validate_waypoint_diversion(aircraft_lat: float, aircraft_lon: float, 
                              waypoint_name: str, max_diversion_nm: float) -> Optional[Tuple[float, float, float]]:
    """Validate waypoint exists and is within diversion limits.
    
    Args:
        aircraft_lat: Current aircraft latitude
        aircraft_lon: Current aircraft longitude
        waypoint_name: Name of target waypoint
        max_diversion_nm: Maximum allowed diversion distance
        
    Returns:
        (lat, lon, distance_nm) tuple if valid, None if invalid
    """
    # Resolve waypoint coordinates
    coords = resolve_fix(waypoint_name)
    if not coords:
        logger.warning(f"Waypoint '{waypoint_name}' not found in navigation database")
        return None
    
    wp_lat, wp_lon = coords
    
    # Check distance constraint
    aircraft_pos = (aircraft_lat, aircraft_lon)
    waypoint_pos = (wp_lat, wp_lon)
    distance = haversine_nm(aircraft_pos, waypoint_pos)
    
    if distance > max_diversion_nm:
        logger.warning(f"Waypoint '{waypoint_name}' at {distance:.1f} NM exceeds diversion limit {max_diversion_nm:.1f} NM")
        return None
    
    logger.info(f"Waypoint '{waypoint_name}' validated: ({wp_lat:.6f}, {wp_lon:.6f}) at {distance:.1f} NM")
    return (wp_lat, wp_lon, distance)
