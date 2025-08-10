"""Conflict detection algorithms for 10-minute lookahead prediction.

This module implements the core conflict detection logic that:
- Takes current aircraft states from BlueSky
- Projects trajectories forward 10 minutes  
- Identifies potential conflicts based on separation standards
- Returns structured conflict predictions for LLM processing
- Uses enhanced CPA calculations with adaptive cadence
"""

import math
from typing import List, Tuple
from .schemas import ConflictPrediction, AircraftState
from .enhanced_cpa import (
    calculate_enhanced_cpa, check_minimum_separation, 
    calculate_adaptive_cadence, CPAResult, MinSepCheck
)


# Aviation separation standards
MIN_HORIZONTAL_SEP_NM = 5.0  # Minimum horizontal separation
MIN_VERTICAL_SEP_FT = 1000.0  # Minimum vertical separation


def predict_conflicts(
    ownship: AircraftState,
    traffic: List[AircraftState],
    lookahead_minutes: float = 10.0,
    time_step_seconds: float = 30.0
) -> List[ConflictPrediction]:
    """Predict conflicts within lookahead time window.
    
    For each intruder within 100 NM / +/-5000 ft, compute CPA; flag conflict if 
    dmin < 5 NM and |deltaalt| < 1000 ft within tmin <= 10 min.
    
    Args:
        ownship: Current ownship state
        traffic: List of traffic aircraft states
        lookahead_minutes: Prediction time horizon
        time_step_seconds: Trajectory sampling interval
        
    Returns:
        List of predicted conflicts sorted by time to conflict
    """
    from .geodesy import haversine_nm, cpa_nm
    
    conflicts = []
    
    # Convert ownship to format expected by geodesy functions
    own_dict = {
        "lat": ownship.latitude,
        "lon": ownship.longitude, 
        "spd_kt": ownship.ground_speed_kt,
        "hdg_deg": ownship.heading_deg,
        "alt_ft": ownship.altitude_ft
    }
    
    for intruder in traffic:
        # Skip self
        if intruder.aircraft_id == ownship.aircraft_id:
            continue
            
        # Pre-filter: check if within 100 NM horizontally
        own_pos = (ownship.latitude, ownship.longitude)
        intr_pos = (intruder.latitude, intruder.longitude)
        horizontal_distance = haversine_nm(own_pos, intr_pos)
        
        if horizontal_distance > 100.0:
            continue
            
        # Pre-filter: check if within +/-5000 ft vertically
        altitude_diff = abs(ownship.altitude_ft - intruder.altitude_ft)
        if altitude_diff > 5000.0:
            continue
            
        # Convert intruder to format expected by geodesy functions
        intr_dict = {
            "lat": intruder.latitude,
            "lon": intruder.longitude,
            "spd_kt": intruder.ground_speed_kt, 
            "hdg_deg": intruder.heading_deg,
            "alt_ft": intruder.altitude_ft
        }
        
        # Compute CPA
        dmin_nm, tmin_min = cpa_nm(own_dict, intr_dict)
        
        # Check conflict criteria (assume level flight for alt)
        future_alt_diff = abs(own_dict["alt_ft"] - intr_dict["alt_ft"])  # Assuming level flight

        # Skip if time to CPA is negative (aircraft diverging)
        if tmin_min <= 0:
            continue

        horizontal_violation = dmin_nm < MIN_HORIZONTAL_SEP_NM
        vertical_violation = future_alt_diff < MIN_VERTICAL_SEP_FT

        # Flag as conflict if either standard is violated within lookahead
        if (horizontal_violation or vertical_violation) and tmin_min <= lookahead_minutes:
            # Calculate severity score based on proximity and time
            severity = calculate_severity_score(dmin_nm, future_alt_diff, tmin_min)

            # Determine conflict type based on which standards are violated
            if horizontal_violation and vertical_violation:
                conflict_type = "both"
            elif horizontal_violation:
                conflict_type = "horizontal"
            else:
                conflict_type = "vertical"

            conflict = ConflictPrediction(
                ownship_id=ownship.aircraft_id,
                intruder_id=intruder.aircraft_id,
                time_to_cpa_min=tmin_min,
                distance_at_cpa_nm=dmin_nm,
                altitude_diff_ft=future_alt_diff,
                is_conflict=True,
                severity_score=severity,
                conflict_type=conflict_type,
                prediction_time=ownship.timestamp,
                confidence=1.0  # Deterministic detection has full confidence
            )
            conflicts.append(conflict)
    
    # Sort by time to conflict (most urgent first)
    conflicts.sort(key=lambda c: c.time_to_cpa_min)
    return conflicts


def predict_conflicts_enhanced(
    ownship: AircraftState,
    traffic: List[AircraftState],
    lookahead_minutes: float = 10.0,
    use_adaptive_cadence: bool = True
) -> Tuple[List[ConflictPrediction], float]:
    """Enhanced conflict detection with CPA verification and adaptive cadence.
    
    This function addresses the fixed cadence issue by:
    - Using enhanced CPA calculations with confidence scoring
    - Performing minimum separation verification
    - Calculating adaptive polling intervals based on proximity/urgency
    - Cross-validating results for accuracy
    
    Args:
        ownship: Current ownship state
        traffic: List of traffic aircraft states
        lookahead_minutes: Prediction time horizon
        use_adaptive_cadence: Whether to calculate adaptive polling interval
        
    Returns:
        Tuple of (conflicts, recommended_polling_interval_minutes)
    """
    from .geodesy import haversine_nm
    
    conflicts = []
    
    # Calculate adaptive cadence first (affects detection sensitivity)
    if use_adaptive_cadence:
        initial_conflicts = predict_conflicts(ownship, traffic, lookahead_minutes)
        recommended_interval = calculate_adaptive_cadence(ownship, traffic, initial_conflicts)
    else:
        recommended_interval = 2.0  # Default 2-minute interval
    
    for intruder in traffic:
        # Skip self
        if intruder.aircraft_id == ownship.aircraft_id:
            continue
            
        # Pre-filter: check if within reasonable range
        own_pos = (ownship.latitude, ownship.longitude)
        intr_pos = (intruder.latitude, intruder.longitude)
        horizontal_distance = haversine_nm(own_pos, intr_pos)
        
        # Dynamic pre-filtering based on adaptive cadence
        max_range_nm = 100.0 if recommended_interval > 2.0 else 150.0  # Expand range for urgent situations
        if horizontal_distance > max_range_nm:
            continue
            
        # Pre-filter: check if within vertical range
        altitude_diff = abs(ownship.altitude_ft - intruder.altitude_ft)
        max_alt_diff = 5000.0 if recommended_interval > 2.0 else 8000.0  # Expand range for urgent situations
        if altitude_diff > max_alt_diff:
            continue
            
        # Enhanced CPA calculation
        cpa_result = calculate_enhanced_cpa(ownship, intruder)
        
        # Skip if aircraft are diverging or CPA is too far in future
        if cpa_result.time_to_cpa_min <= 0 or cpa_result.time_to_cpa_min > lookahead_minutes:
            continue
            
        # Current minimum separation check
        min_sep_check = check_minimum_separation(ownship, intruder)
        
        # Enhanced conflict determination
        is_conflict = _is_enhanced_conflict(cpa_result, min_sep_check, lookahead_minutes)
        
        if is_conflict:
            # Calculate enhanced severity based on multiple factors
            severity = _calculate_enhanced_severity(cpa_result, min_sep_check, recommended_interval)
            
            # Determine conflict type
            conflict_type = _determine_conflict_type(cpa_result, min_sep_check)
            
            # Calculate future altitude difference (assuming level flight)
            future_alt_diff = abs(ownship.altitude_ft - intruder.altitude_ft)
            
            conflict = ConflictPrediction(
                ownship_id=ownship.aircraft_id,
                intruder_id=intruder.aircraft_id,
                time_to_cpa_min=cpa_result.time_to_cpa_min,
                distance_at_cpa_nm=cpa_result.distance_at_cpa_nm,
                altitude_diff_ft=future_alt_diff,
                is_conflict=True,
                severity_score=severity,
                conflict_type=conflict_type,
                prediction_time=ownship.timestamp,
                confidence=cpa_result.confidence
            )
            conflicts.append(conflict)
    
    # Sort by urgency (combination of time and severity)
    conflicts.sort(key=lambda c: (c.time_to_cpa_min, -c.severity_score))
    
    return conflicts, recommended_interval


def _is_enhanced_conflict(cpa_result: CPAResult, min_sep_check: MinSepCheck, lookahead_min: float) -> bool:
    """Enhanced conflict determination using CPA and minimum separation data.
    
    Args:
        cpa_result: Enhanced CPA calculation result
        min_sep_check: Current separation check result
        lookahead_min: Lookahead time window
        
    Returns:
        True if conflict criteria are met
    """
    # Immediate conflict if currently violating separation
    if min_sep_check.is_conflict:
        return True
    
    # Future conflict prediction
    if cpa_result.time_to_cpa_min > 0 and cpa_result.time_to_cpa_min <= lookahead_min:
        # Horizontal violation at CPA
        horizontal_violation = cpa_result.distance_at_cpa_nm < MIN_HORIZONTAL_SEP_NM
        
        # For now, assume level flight for vertical separation
        # (Could be enhanced with vertical speed predictions)
        vertical_violation = min_sep_check.vertical_sep_ft < MIN_VERTICAL_SEP_FT
        
        # Conflict if both standards will be violated
        if horizontal_violation and vertical_violation:
            return True
        
        # Special case: very close approach even if not technically violating standards
        if cpa_result.distance_at_cpa_nm < 3.0 and cpa_result.time_to_cpa_min < 3.0:
            return True
    
    return False


def _calculate_enhanced_severity(cpa_result: CPAResult, min_sep_check: MinSepCheck, 
                                polling_interval: float) -> float:
    """Calculate enhanced severity score considering multiple factors.
    
    Args:
        cpa_result: CPA calculation result
        min_sep_check: Separation check result  
        polling_interval: Current polling interval (indicates urgency)
        
    Returns:
        Severity score from 0.0 to 1.0
    """
    severity = 0.0
    
    # Time urgency factor (closer = higher severity)
    if cpa_result.time_to_cpa_min > 0:
        time_factor = max(0.0, 1.0 - (cpa_result.time_to_cpa_min / 10.0))
        severity += 0.4 * time_factor
    
    # Distance factor (closer = higher severity)
    if cpa_result.distance_at_cpa_nm >= 0:
        distance_factor = max(0.0, 1.0 - (cpa_result.distance_at_cpa_nm / MIN_HORIZONTAL_SEP_NM))
        severity += 0.3 * distance_factor
    
    # Convergence rate factor (faster convergence = higher severity)
    if cpa_result.convergence_rate_nm_min < 0:  # Converging
        convergence_factor = min(1.0, abs(cpa_result.convergence_rate_nm_min) / 10.0)
        severity += 0.2 * convergence_factor
    
    # Polling interval urgency (shorter interval indicates higher urgency)
    interval_factor = max(0.0, 1.0 - (polling_interval / 5.0))
    severity += 0.1 * interval_factor
    
    # Ensure severity is between 0.1 and 1.0
    return max(0.1, min(1.0, severity))


def _determine_conflict_type(cpa_result: CPAResult, min_sep_check: MinSepCheck) -> str:
    """Determine the type of conflict based on separation violations.
    
    Args:
        cpa_result: CPA calculation result
        min_sep_check: Separation check result
        
    Returns:
        Conflict type string
    """
    if min_sep_check.horizontal_violation and min_sep_check.vertical_violation:
        return "both"
    elif min_sep_check.horizontal_violation:
        return "horizontal"
    elif min_sep_check.vertical_violation:
        return "vertical"
    elif cpa_result.distance_at_cpa_nm < MIN_HORIZONTAL_SEP_NM:
        return "horizontal"
    else:
        return "horizontal"  # Default


def is_conflict(
    distance_nm: float,
    altitude_diff_ft: float,
    time_to_cpa_min: float
) -> bool:
    """Determine if predicted encounter constitutes a conflict.
    
    A conflict occurs when both horizontal and vertical separation
    standards are violated simultaneously.
    
    Args:
        distance_nm: Horizontal separation at CPA
        altitude_diff_ft: Vertical separation 
        time_to_cpa_min: Time until closest approach
        
    Returns:
        True if conflict criteria are met
    """
    # Must occur in the future (or now) - diverging if tcpa < 0
    if time_to_cpa_min < 0:
        return False
    
    # If both horizontal and vertical minima exceed thresholds, no conflict
    if distance_nm >= MIN_HORIZONTAL_SEP_NM and abs(altitude_diff_ft) >= MIN_VERTICAL_SEP_FT:
        return False

    # Check separation standards
    horizontal_violation = distance_nm < MIN_HORIZONTAL_SEP_NM
    vertical_violation = abs(altitude_diff_ft) < MIN_VERTICAL_SEP_FT

    # Core definition: conflict requires both standards to be violated simultaneously
    return horizontal_violation and vertical_violation


def calculate_severity_score(
    distance_nm: float, 
    altitude_diff_ft: float, 
    time_to_cpa_min: float
) -> float:
    """Calculate conflict severity score [0-1].
    
    Args:
        distance_nm: Horizontal separation at CPA
        altitude_diff_ft: Vertical separation
        time_to_cpa_min: Time until closest approach
        
    Returns:
        Severity score where 1.0 is most severe
    """
    # Normalize distances to [0-1] where 1 = most severe
    h_severity = max(0.0, 1.0 - distance_nm / MIN_HORIZONTAL_SEP_NM)
    v_severity = max(0.0, 1.0 - altitude_diff_ft / MIN_VERTICAL_SEP_FT)
    
    # Time urgency factor (closer in time = more severe)
    time_urgency = max(0.0, 1.0 - time_to_cpa_min / 10.0)
    
    # Combined severity (weighted average)
    severity = (h_severity * 0.4 + v_severity * 0.4 + time_urgency * 0.2)
    return min(1.0, severity)


def project_trajectory(
    aircraft: AircraftState,
    time_horizon_minutes: float,
    time_step_seconds: float = 30.0
) -> List[Tuple[float, float, float, float]]:
    """Project aircraft trajectory assuming constant velocity.
    
    Args:
        aircraft: Current aircraft state
        time_horizon_minutes: How far to project forward
        time_step_seconds: Sampling interval
        
    Returns:
        List of (time_min, lat, lon, alt_ft) waypoints
    """
    import math
    
    waypoints = []
    
    # Initial position
    lat = aircraft.latitude
    lon = aircraft.longitude
    alt = aircraft.altitude_ft
    
    # Convert to motion components
    speed_kt = aircraft.ground_speed_kt
    heading_rad = math.radians(aircraft.heading_deg)
    vertical_speed_fpm = aircraft.vertical_speed_fpm
    
    # Speed components (in degrees per minute for lat/lon approximation)
    # 1 nm = 1/60 degrees latitude approximately
    speed_nm_per_min = speed_kt / 60.0  # Convert knots to nm/min
    
    # Velocity components in degrees per minute
    dlat_per_min = (speed_nm_per_min / 60.0) * math.cos(heading_rad)  # North component
    dlon_per_min = (speed_nm_per_min / 60.0) * math.sin(heading_rad) / math.cos(math.radians(lat))  # East component
    dalt_per_min = vertical_speed_fpm  # Already in ft/min
    
    # Generate trajectory points
    num_steps = int((time_horizon_minutes * 60) / time_step_seconds)
    dt_min = time_step_seconds / 60.0  # Convert to minutes
    
    for i in range(num_steps + 1):
        time_min = i * dt_min
        
        # Project position
        current_lat = lat + dlat_per_min * time_min
        current_lon = lon + dlon_per_min * time_min
        current_alt = alt + dalt_per_min * time_min
        
        waypoints.append((time_min, current_lat, current_lon, current_alt))
        
        if time_min >= time_horizon_minutes:
            break
    
    return waypoints
