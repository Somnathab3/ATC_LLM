"""Enhanced CPA and minimum separation verification functions.

This module provides enhanced conflict detection capabilities:
- 2D and 3D CPA calculations with adaptive cadence
- Minimum separation verification (5 NM / 1000 ft)
- Cross-validation with BlueSky conflict detection
- Adaptive polling intervals based on proximity and time-to-CPA
"""

import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .geodesy import cpa_nm, haversine_nm
from .schemas import AircraftState, ConflictPrediction

logger = logging.getLogger(__name__)

# Aviation separation standards
MIN_HORIZONTAL_SEP_NM = 5.0
MIN_VERTICAL_SEP_FT = 1000.0

# Proximity thresholds for adaptive cadence
CRITICAL_PROXIMITY_NM = 25.0  # Close proximity threshold
URGENT_TIME_TO_CPA_MIN = 6.0  # Urgent time threshold
IMMINENT_TIME_TO_CPA_MIN = 2.0  # Imminent threat threshold

@dataclass
class CPAResult:
    """Result of CPA calculation with enhanced metadata."""
    distance_at_cpa_nm: float
    time_to_cpa_min: float
    cpa_position_own: Tuple[float, float]  # (lat, lon) at CPA
    cpa_position_intruder: Tuple[float, float]  # (lat, lon) at CPA
    relative_speed_kt: float
    convergence_rate_nm_min: float
    is_converging: bool
    confidence: float = 1.0

@dataclass
class MinSepCheck:
    """Minimum separation check result."""
    horizontal_sep_nm: float
    vertical_sep_ft: float
    horizontal_violation: bool
    vertical_violation: bool
    is_conflict: bool
    margin_horizontal_nm: float
    margin_vertical_ft: float

def calculate_enhanced_cpa(own: AircraftState, intruder: AircraftState) -> CPAResult:
    """Calculate enhanced CPA with additional metadata.
    
    Args:
        own: Ownship aircraft state
        intruder: Intruder aircraft state
        
    Returns:
        Enhanced CPA result with metadata
    """
    # Convert to format expected by geodesy functions
    own_dict = {
        "lat": own.latitude,
        "lon": own.longitude,
        "spd_kt": own.ground_speed_kt,
        "hdg_deg": own.heading_deg,
        "alt_ft": own.altitude_ft
    }
    
    intruder_dict = {
        "lat": intruder.latitude,
        "lon": intruder.longitude,
        "spd_kt": intruder.ground_speed_kt,
        "hdg_deg": intruder.heading_deg,
        "alt_ft": intruder.altitude_ft
    }
    
    # Basic CPA calculation
    distance_nm, time_min = cpa_nm(own_dict, intruder_dict)
    
    # Calculate CPA positions
    cpa_own_lat, cpa_own_lon = _project_position(own, time_min)
    cpa_intruder_lat, cpa_intruder_lon = _project_position(intruder, time_min)
    
    # Calculate relative speed and convergence
    relative_speed_kt = _calculate_relative_speed(own, intruder)
    convergence_rate = _calculate_convergence_rate(own, intruder)
    is_converging = convergence_rate < 0  # Negative means decreasing distance
    
    # Confidence based on data quality and scenario complexity
    confidence = _calculate_cpa_confidence(own, intruder, time_min)
    
    return CPAResult(
        distance_at_cpa_nm=distance_nm,
        time_to_cpa_min=time_min,
        cpa_position_own=(cpa_own_lat, cpa_own_lon),
        cpa_position_intruder=(cpa_intruder_lat, cpa_intruder_lon),
        relative_speed_kt=relative_speed_kt,
        convergence_rate_nm_min=convergence_rate,
        is_converging=is_converging,
        confidence=confidence
    )

def check_minimum_separation(own: AircraftState, intruder: AircraftState) -> MinSepCheck:
    """Check current separation against minimum standards.
    
    Args:
        own: Ownship aircraft state
        intruder: Intruder aircraft state
        
    Returns:
        Minimum separation check result
    """
    # Current horizontal separation
    own_pos = (own.latitude, own.longitude)
    intruder_pos = (intruder.latitude, intruder.longitude)
    horizontal_sep_nm = haversine_nm(own_pos, intruder_pos)
    
    # Current vertical separation
    vertical_sep_ft = abs(own.altitude_ft - intruder.altitude_ft)
    
    # Check violations
    horizontal_violation = horizontal_sep_nm < MIN_HORIZONTAL_SEP_NM
    vertical_violation = vertical_sep_ft < MIN_VERTICAL_SEP_FT
    
    # Conflict occurs when both standards are violated
    is_conflict = horizontal_violation and vertical_violation
    
    # Calculate margins
    margin_horizontal_nm = horizontal_sep_nm - MIN_HORIZONTAL_SEP_NM
    margin_vertical_ft = vertical_sep_ft - MIN_VERTICAL_SEP_FT
    
    return MinSepCheck(
        horizontal_sep_nm=horizontal_sep_nm,
        vertical_sep_ft=vertical_sep_ft,
        horizontal_violation=horizontal_violation,
        vertical_violation=vertical_violation,
        is_conflict=is_conflict,
        margin_horizontal_nm=margin_horizontal_nm,
        margin_vertical_ft=margin_vertical_ft
    )

def calculate_adaptive_cadence(
    ownship: AircraftState,
    traffic: List[AircraftState],
    conflicts: List[ConflictPrediction]
) -> float:
    """Calculate adaptive polling cadence based on proximity and time-to-CPA.
    
    Addresses the issue of fixed cadence missing imminent conflicts or causing
    unnecessary LLM calls when traffic is sparse.
    
    Args:
        ownship: Current ownship state
        traffic: List of traffic aircraft
        conflicts: Current conflict predictions
        
    Returns:
        Polling interval in minutes (0.5 to 5.0)
    """
    # Base intervals
    IMMINENT_INTERVAL = 0.5  # 30 seconds for imminent threats
    URGENT_INTERVAL = 1.0    # 1 minute for urgent situations
    NORMAL_INTERVAL = 2.0    # 2 minutes for normal conditions
    SPARSE_INTERVAL = 5.0    # 5 minutes when traffic is sparse
    
    # Check for imminent conflicts (< 2 min CPA)
    if conflicts:
        active_conflicts = [c for c in conflicts if c.is_conflict]
        if active_conflicts:
            min_time_to_cpa = min(c.time_to_cpa_min for c in active_conflicts)
            
            if min_time_to_cpa <= IMMINENT_TIME_TO_CPA_MIN:
                logger.info(f"IMMINENT conflict detected (CPA in {min_time_to_cpa:.1f} min), using {IMMINENT_INTERVAL} min interval")
                return IMMINENT_INTERVAL
            elif min_time_to_cpa <= URGENT_TIME_TO_CPA_MIN:
                logger.info(f"URGENT conflict detected (CPA in {min_time_to_cpa:.1f} min), using {URGENT_INTERVAL} min interval")
                return URGENT_INTERVAL
    
    # Check proximity to any traffic
    if not traffic:
        logger.debug("No traffic detected, using sparse interval")
        return SPARSE_INTERVAL
    
    # Find closest aircraft
    closest_distance = float('inf')
    closest_converging_time = float('inf')
    
    for aircraft in traffic:
        if aircraft.aircraft_id == ownship.aircraft_id:
            continue
            
        # Current distance
        own_pos = (ownship.latitude, ownship.longitude)
        aircraft_pos = (aircraft.latitude, aircraft.longitude)
        distance_nm = haversine_nm(own_pos, aircraft_pos)
        closest_distance = min(closest_distance, distance_nm)
        
        # CPA time for converging aircraft
        cpa_result = calculate_enhanced_cpa(ownship, aircraft)
        if cpa_result.is_converging and cpa_result.time_to_cpa_min > 0:
            closest_converging_time = min(closest_converging_time, cpa_result.time_to_cpa_min)
    
    # Adaptive logic based on proximity and convergence
    if closest_distance < CRITICAL_PROXIMITY_NM:
        if closest_converging_time < URGENT_TIME_TO_CPA_MIN:
            logger.debug(f"Close proximity ({closest_distance:.1f} NM) with convergence in {closest_converging_time:.1f} min, using urgent interval")
            return URGENT_INTERVAL
        else:
            logger.debug(f"Close proximity ({closest_distance:.1f} NM), using normal interval")
            return NORMAL_INTERVAL
    elif closest_distance < 50.0:  # Moderate proximity
        logger.debug(f"Moderate proximity ({closest_distance:.1f} NM), using normal interval")
        return NORMAL_INTERVAL
    else:
        logger.debug(f"Distant traffic ({closest_distance:.1f} NM), using sparse interval")
        return SPARSE_INTERVAL

def cross_validate_with_bluesky(
    conflicts: List[ConflictPrediction],
    bluesky_client: Any  # BlueSkyClient
) -> Dict[str, Any]:
    """Cross-validate conflict detection with BlueSky's built-in CD.
    
    Args:
        conflicts: Our conflict predictions
        bluesky_client: BlueSky client instance
        
    Returns:
        Validation results and discrepancies
    """
    validation_result = {
        "our_conflicts": len(conflicts),
        "bluesky_conflicts": 0,
        "matched_conflicts": 0,
        "false_positives": [],
        "false_negatives": [],
        "discrepancies": []
    }
    
    # This is a placeholder for BlueSky CD integration
    # In practice, you would query BlueSky's conflict detection results
    # and compare with our predictions
    
    logger.debug("Cross-validation with BlueSky CD not yet implemented")
    
    return validation_result

def _project_position(aircraft: AircraftState, time_min: float) -> Tuple[float, float]:
    """Project aircraft position forward in time.
    
    Args:
        aircraft: Aircraft state
        time_min: Time in minutes to project forward
        
    Returns:
        Projected (latitude, longitude) position
    """
    # Convert time to hours
    time_hr = time_min / 60.0
    
    # Calculate distance traveled
    distance_nm = aircraft.ground_speed_kt * time_hr
    
    # Convert heading to radians (aviation convention: 0=North, clockwise)
    heading_rad = math.radians(aircraft.heading_deg)
    
    # Calculate displacement components
    delta_lat_nm = distance_nm * math.cos(heading_rad)
    delta_lon_nm = distance_nm * math.sin(heading_rad)
    
    # Convert to degrees (approximate)
    delta_lat_deg = delta_lat_nm / 60.0  # 1 degree ≈ 60 NM
    delta_lon_deg = delta_lon_nm / (60.0 * math.cos(math.radians(aircraft.latitude)))
    
    projected_lat = aircraft.latitude + delta_lat_deg
    projected_lon = aircraft.longitude + delta_lon_deg
    
    return projected_lat, projected_lon

def _calculate_relative_speed(own: AircraftState, intruder: AircraftState) -> float:
    """Calculate relative speed between two aircraft.
    
    Returns:
        Relative speed in knots
    """
    # Convert headings to velocity components
    own_vx = own.ground_speed_kt * math.sin(math.radians(own.heading_deg))
    own_vy = own.ground_speed_kt * math.cos(math.radians(own.heading_deg))
    
    intr_vx = intruder.ground_speed_kt * math.sin(math.radians(intruder.heading_deg))
    intr_vy = intruder.ground_speed_kt * math.cos(math.radians(intruder.heading_deg))
    
    # Relative velocity components
    rel_vx = own_vx - intr_vx
    rel_vy = own_vy - intr_vy
    
    # Relative speed magnitude
    relative_speed = math.sqrt(rel_vx**2 + rel_vy**2)
    
    return relative_speed

def _calculate_convergence_rate(own: AircraftState, intruder: AircraftState) -> float:
    """Calculate rate of distance change (convergence/divergence).
    
    Returns:
        Rate in NM/min (negative = converging, positive = diverging)
    """
    # Current distance
    own_pos = (own.latitude, own.longitude)
    intr_pos = (intruder.latitude, intruder.longitude)
    current_distance = haversine_nm(own_pos, intr_pos)
    
    # Project positions 1 minute ahead
    own_future_lat, own_future_lon = _project_position(own, 1.0)
    intr_future_lat, intr_future_lon = _project_position(intruder, 1.0)
    
    future_distance = haversine_nm(
        (own_future_lat, own_future_lon),
        (intr_future_lat, intr_future_lon)
    )
    
    # Rate of change (NM/min)
    convergence_rate = future_distance - current_distance
    
    return convergence_rate

def _calculate_cpa_confidence(own: AircraftState, intruder: AircraftState, time_to_cpa: float) -> float:
    """Calculate confidence in CPA prediction.
    
    Factors affecting confidence:
    - Time horizon (shorter = more reliable)
    - Aircraft speeds (higher speeds = less predictable)
    - Data freshness
    
    Returns:
        Confidence score from 0.0 to 1.0
    """
    confidence = 1.0
    
    # Reduce confidence for long time horizons
    if time_to_cpa > 10.0:
        confidence *= 0.8
    elif time_to_cpa > 5.0:
        confidence *= 0.9
    
    # Reduce confidence for high-speed aircraft (more unpredictable)
    max_speed = max(own.ground_speed_kt, intruder.ground_speed_kt)
    if max_speed > 500.0:
        confidence *= 0.9
    elif max_speed > 600.0:
        confidence *= 0.8
    
    # Could add factors for:
    # - Data age/freshness
    # - Weather conditions
    # - Aircraft type predictability
    # - ATC control status
    
    return max(0.1, confidence)  # Minimum confidence of 0.1
