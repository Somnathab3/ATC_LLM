"""Conflict resolution algorithms with safety validation.

This module implements conflict resolution logic that:
- Processes LLM-generated resolution suggestions
- Applies safety validation before execution
- Implements fallback strategies for unsafe suggestions
- Supports horizontal (heading) and vertical (altitude) maneuvers
"""

import logging
import math
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from .schemas import (
    ResolveOut, ResolutionCommand, ResolutionType, 
    AircraftState, ConflictPrediction
)
from .geodesy import cpa_nm

logger = logging.getLogger(__name__)

# Safety constraints
MAX_HEADING_CHANGE_DEG = 30.0
MIN_ALTITUDE_CHANGE_FT = 1000.0  
MAX_ALTITUDE_CHANGE_FT = 2000.0
MIN_SAFE_SEPARATION_NM = 5.0
MIN_SAFE_SEPARATION_FT = 1000.0


def execute_resolution(
    llm_resolution: ResolveOut,
    ownship: AircraftState,
    intruder: AircraftState,
    conflict: ConflictPrediction
) -> Optional[ResolutionCommand]:
    """Execute LLM resolution with safety validation.
    
    Args:
        llm_resolution: LLM-generated resolution
        ownship: Current ownship state
        intruder: Intruder aircraft state
        conflict: Predicted conflict details
        
    Returns:
        Validated resolution command or None if unsafe
    """
    try:
        # Create initial resolution command
        resolution_cmd = _create_resolution_command(llm_resolution, ownship)
        
        if resolution_cmd is None:
            logger.error("Failed to create resolution command")
            return None
            
        # Validate safety
        if _validate_resolution_safety(resolution_cmd, ownship, intruder):
            resolution_cmd.is_validated = True
            logger.info(f"Resolution validated: {llm_resolution.action}")
            return resolution_cmd
        else:
            logger.warning(f"LLM resolution failed safety validation: {llm_resolution.action}")
            
            # Try fallback resolution
            fallback_cmd = _generate_fallback_resolution(ownship, intruder, conflict)
            if fallback_cmd and _validate_resolution_safety(fallback_cmd, ownship, intruder):
                fallback_cmd.is_validated = True
                logger.info("Fallback resolution validated")
                return fallback_cmd
            else:
                logger.error("No safe resolution found")
                return None
                
    except Exception as e:
        logger.error(f"Resolution execution failed: {e}")
        return None


def generate_horizontal_resolution(
    conflict: ConflictPrediction,
    ownship: AircraftState,
    preferred_turn: str = "right"
) -> Optional[ResolutionCommand]:
    """Generate horizontal conflict resolution.
    
    Args:
        conflict: Predicted conflict details
        ownship: Current ownship state
        preferred_turn: Preferred turn direction ("left" or "right")
        
    Returns:
        Resolution command or None if no solution found
    """
    try:
        # Calculate turn direction and magnitude
        turn_magnitude = 20.0  # Default 20-degree turn
        
        if preferred_turn == "right":
            new_heading = (ownship.heading_deg + turn_magnitude) % 360
        else:
            new_heading = (ownship.heading_deg - turn_magnitude) % 360
            
        return ResolutionCommand(
            resolution_id=f"h_res_{int(datetime.now().timestamp())}",
            target_aircraft=ownship.aircraft_id,
            resolution_type=ResolutionType.HEADING_CHANGE,
            new_heading_deg=new_heading,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=False
        )
        
    except Exception as e:
        logger.error(f"Horizontal resolution generation failed: {e}")
        return None


def generate_vertical_resolution(
    conflict: ConflictPrediction,
    ownship: AircraftState,
    preferred_direction: str = "climb"
) -> Optional[ResolutionCommand]:
    """Generate vertical conflict resolution.
    
    Args:
        conflict: Predicted conflict details
        ownship: Current ownship state
        preferred_direction: Preferred direction ("climb" or "descend")
        
    Returns:
        Resolution command or None if no solution found
    """
    try:
        # Standard altitude change
        altitude_change = 1000.0
        
        if preferred_direction == "climb":
            new_altitude = ownship.altitude_ft + altitude_change
        else:
            new_altitude = ownship.altitude_ft - altitude_change
            
        # Ensure altitude is within reasonable bounds
        new_altitude = max(1000, min(45000, new_altitude))
        
        return ResolutionCommand(
            resolution_id=f"v_res_{int(datetime.now().timestamp())}",
            target_aircraft=ownship.aircraft_id,
            resolution_type=ResolutionType.ALTITUDE_CHANGE,
            new_altitude_ft=new_altitude,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=False
        )
        
        return cmd
        
    except Exception as e:
        logger.error(f"Vertical resolution generation failed: {e}")
        return None


def _create_resolution_command(
    llm_resolution: ResolveOut, 
    ownship: AircraftState
) -> Optional[ResolutionCommand]:
    """Create resolution command from LLM output.
    
    Args:
        llm_resolution: LLM resolution output
        ownship: Current ownship state
        
    Returns:
        Resolution command or None if invalid
    """
    try:
        # Validate action type
        if llm_resolution.action not in ["turn", "climb", "descend"]:
            logger.error(f"Invalid action type: {llm_resolution.action}")
            return None
            
        # Create base command
        cmd = ResolutionCommand(
            resolution_id=f"res_{int(datetime.now().timestamp())}",
            target_aircraft=ownship.aircraft_id,
            resolution_type=_map_action_to_resolution_type(llm_resolution.action),
            issue_time=datetime.now(),
            safety_margin_nm=0.0,  # Will be calculated later
            is_validated=False
        )
        
        # Apply action-specific parameters
        if llm_resolution.action == "turn":
            new_heading = llm_resolution.params.get("heading_deg")
            if new_heading is None:
                logger.error("Turn action missing heading_deg parameter")
                return None
                
            # Validate heading change magnitude
            heading_change = abs(new_heading - ownship.heading_deg)
            if heading_change > 180:
                heading_change = 360 - heading_change
                
            if heading_change > MAX_HEADING_CHANGE_DEG:
                logger.warning(f"Heading change {heading_change:.1f}° exceeds limit {MAX_HEADING_CHANGE_DEG}°")
                # Clamp to maximum allowed change
                if new_heading > ownship.heading_deg:
                    cmd.new_heading_deg = (ownship.heading_deg + MAX_HEADING_CHANGE_DEG) % 360
                else:
                    cmd.new_heading_deg = (ownship.heading_deg - MAX_HEADING_CHANGE_DEG) % 360
            else:
                cmd.new_heading_deg = new_heading % 360
                
        elif llm_resolution.action in ["climb", "descend"]:
            delta_ft = llm_resolution.params.get("delta_ft")
            if delta_ft is None:
                logger.error(f"{llm_resolution.action} action missing delta_ft parameter")
                return None
                
            # Validate altitude change magnitude
            if abs(delta_ft) < MIN_ALTITUDE_CHANGE_FT:
                logger.warning(f"Altitude change {delta_ft} ft below minimum {MIN_ALTITUDE_CHANGE_FT} ft")
                delta_ft = MIN_ALTITUDE_CHANGE_FT if delta_ft > 0 else -MIN_ALTITUDE_CHANGE_FT
                
            if abs(delta_ft) > MAX_ALTITUDE_CHANGE_FT:
                logger.warning(f"Altitude change {delta_ft} ft exceeds limit {MAX_ALTITUDE_CHANGE_FT} ft")
                delta_ft = MAX_ALTITUDE_CHANGE_FT if delta_ft > 0 else -MAX_ALTITUDE_CHANGE_FT
                
            # Apply altitude change
            if llm_resolution.action == "climb":
                cmd.new_altitude_ft = ownship.altitude_ft + abs(delta_ft)
            else:  # descend
                cmd.new_altitude_ft = ownship.altitude_ft - abs(delta_ft)
                
            # Ensure altitude stays within reasonable bounds
            cmd.new_altitude_ft = max(1000, min(45000, cmd.new_altitude_ft))
            
        return cmd
        
    except Exception as e:
        logger.error(f"Failed to create resolution command: {e}")
        return None


def _validate_resolution_safety(
    resolution_cmd: ResolutionCommand,
    ownship: AircraftState, 
    intruder: AircraftState
) -> bool:
    """Validate that resolution maintains safe separation.
    
    Args:
        resolution_cmd: Proposed resolution command
        ownship: Current ownship state
        intruder: Intruder aircraft state
        
    Returns:
        True if resolution is safe
    """
    try:
        # Create modified ownship state with resolution applied
        modified_ownship = AircraftState(
            aircraft_id=ownship.aircraft_id,
            timestamp=ownship.timestamp,
            latitude=ownship.latitude,
            longitude=ownship.longitude,
            altitude_ft=resolution_cmd.new_altitude_ft or ownship.altitude_ft,
            ground_speed_kt=ownship.ground_speed_kt,
            heading_deg=resolution_cmd.new_heading_deg or ownship.heading_deg,
            vertical_speed_fpm=ownship.vertical_speed_fpm
        )
        
        # Convert to format for CPA calculation
        own_dict = {
            "lat": modified_ownship.latitude,
            "lon": modified_ownship.longitude,
            "spd_kt": modified_ownship.ground_speed_kt,
            "hdg_deg": modified_ownship.heading_deg,
            "alt_ft": modified_ownship.altitude_ft
        }
        
        intr_dict = {
            "lat": intruder.latitude,
            "lon": intruder.longitude,
            "spd_kt": intruder.ground_speed_kt,
            "hdg_deg": intruder.heading_deg,
            "alt_ft": intruder.altitude_ft
        }
        
        # Calculate CPA with resolution applied
        dmin_nm, tmin_min = cpa_nm(own_dict, intr_dict)
        
        # Calculate altitude separation
        alt_diff = abs(modified_ownship.altitude_ft - intruder.altitude_ft)
        
        # Check if resolution maintains safe separation
        horizontal_safe = dmin_nm >= MIN_SAFE_SEPARATION_NM
        vertical_safe = alt_diff >= MIN_SAFE_SEPARATION_FT
        
        is_safe = horizontal_safe or vertical_safe  # Only need one dimension to be safe
        
        if is_safe:
            # Update safety margin in command
            resolution_cmd.safety_margin_nm = dmin_nm
            logger.debug(f"Resolution safety check passed: dmin={dmin_nm:.2f} NM, alt_diff={alt_diff:.0f} ft")
        else:
            logger.warning(f"Resolution safety check failed: dmin={dmin_nm:.2f} NM, alt_diff={alt_diff:.0f} ft")
            
        return is_safe
        
    except Exception as e:
        logger.error(f"Safety validation failed: {e}")
        return False


def _generate_fallback_resolution(
    ownship: AircraftState,
    intruder: AircraftState, 
    conflict: ConflictPrediction
) -> Optional[ResolutionCommand]:
    """Generate fallback resolution when LLM suggestion is unsafe.
    
    Args:
        ownship: Current ownship state
        intruder: Intruder aircraft state
        conflict: Predicted conflict
        
    Returns:
        Fallback resolution command or None
    """
    try:
        logger.info("Generating fallback resolution: vertical climb +1000 ft")
        
        # Default fallback: climb 1000 ft
        fallback_cmd = ResolutionCommand(
            resolution_id=f"fallback_{int(datetime.now().timestamp())}",
            target_aircraft=ownship.aircraft_id,
            resolution_type=ResolutionType.ALTITUDE_CHANGE,
            new_altitude_ft=ownship.altitude_ft + 1000,
            issue_time=datetime.now(),
            safety_margin_nm=0.0,
            is_validated=False
        )
        
        return fallback_cmd
        
    except Exception as e:
        logger.error(f"Fallback resolution generation failed: {e}")
        return None


def _map_action_to_resolution_type(action: str) -> ResolutionType:
    """Map action string to ResolutionType enum.
    
    Args:
        action: Action string
        
    Returns:
        Corresponding ResolutionType
    """
    action_map = {
        "turn": ResolutionType.HEADING_CHANGE,
        "climb": ResolutionType.ALTITUDE_CHANGE,
        "descend": ResolutionType.ALTITUDE_CHANGE
    }
    
    return action_map.get(action, ResolutionType.HEADING_CHANGE)


def format_resolution_command(cmd: ResolutionCommand) -> str:
    """Format resolution command for BlueSky execution.
    
    Args:
        cmd: Resolution command
        
    Returns:
        BlueSky command string
    """
    if cmd.resolution_type == ResolutionType.HEADING_CHANGE and cmd.new_heading_deg:
        return f"HDG {cmd.target_aircraft} {cmd.new_heading_deg:.0f}"
    elif cmd.resolution_type == ResolutionType.ALTITUDE_CHANGE and cmd.new_altitude_ft:
        return f"ALT {cmd.target_aircraft} {cmd.new_altitude_ft:.0f}"
    else:
        return f"# Invalid resolution command for {cmd.target_aircraft}"
def validate_resolution(
    resolution: ResolutionCommand,
    ownship: AircraftState,
    traffic: list
) -> bool:
    """Validate resolution against safety constraints.
    
    Args:
        resolution: Proposed resolution command
        ownship: Current ownship state
        traffic: Current traffic states
        
    Returns:
        True if resolution is safe to execute
    """
    # TODO: Implement in Sprint 2
    pass


def to_bluesky_command(
    resolution: ResolutionCommand,
    aircraft_id: str
) -> str:
    """Convert resolution to BlueSky command format.
    
    Args:
        resolution: Resolution command to execute
        aircraft_id: Target aircraft identifier
        
    Returns:
        BlueSky command string (e.g., "HDG KLM123 090")
    """
    # TODO: Implement in Sprint 2
    pass
