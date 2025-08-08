"""Conflict resolution algorithms with safety validation.

This module implements conflict resolution logic that:
- Processes LLM-generated resolution suggestions
- Applies safety validation before execution
- Implements fallback strategies for unsafe suggestions
- Supports horizontal (heading) and vertical (altitude) maneuvers
"""

import logging
import math
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

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

# Oscillation guard constraints
OSCILLATION_WINDOW_MIN = 10.0  # 10 minutes window for oscillation detection
MIN_NET_BENEFIT_THRESHOLD = 0.5  # Minimum improvement in separation (nm) to allow opposite command


@dataclass
class CommandHistory:
    """Track command history for oscillation detection."""
    aircraft_id: str
    command_type: str  # "turn_left", "turn_right", "climb", "descend"
    timestamp: datetime
    heading_change: Optional[float] = None
    altitude_change: Optional[float] = None
    separation_benefit: Optional[float] = None


# Global command history storage
_command_history: Dict[str, List[CommandHistory]] = defaultdict(list)


def _add_command_to_history(
    aircraft_id: str,
    command_type: str,
    heading_change: Optional[float] = None,
    altitude_change: Optional[float] = None,
    separation_benefit: Optional[float] = None
) -> None:
    """Add command to history for oscillation tracking."""
    history_entry = CommandHistory(
        aircraft_id=aircraft_id,
        command_type=command_type,
        timestamp=datetime.now(),
        heading_change=heading_change,
        altitude_change=altitude_change,
        separation_benefit=separation_benefit
    )
    
    _command_history[aircraft_id].append(history_entry)
    
    # Clean old history (keep only last 20 minutes)
    cutoff_time = datetime.now() - timedelta(minutes=20)
    _command_history[aircraft_id] = [
        cmd for cmd in _command_history[aircraft_id] 
        if cmd.timestamp > cutoff_time
    ]


def _check_oscillation_guard(
    aircraft_id: str,
    proposed_command_type: str,
    proposed_separation_benefit: float
) -> bool:
    """Check if proposed command would cause oscillation.
    
    Args:
        aircraft_id: Aircraft identifier
        proposed_command_type: Type of proposed command (turn_left, turn_right, climb, descend)
        proposed_separation_benefit: Expected separation improvement (nm)
        
    Returns:
        True if command is allowed, False if would cause oscillation
    """
    history = _command_history.get(aircraft_id, [])
    if not history:
        return True  # No history, allow command
    
    # Check for opposite commands within oscillation window
    cutoff_time = datetime.now() - timedelta(minutes=OSCILLATION_WINDOW_MIN)
    recent_commands = [cmd for cmd in history if cmd.timestamp > cutoff_time]
    
    # Define opposite command pairs
    opposite_commands = {
        "turn_left": "turn_right",
        "turn_right": "turn_left", 
        "climb": "descend",
        "descend": "climb"
    }
    
    opposite_command = opposite_commands.get(proposed_command_type)
    if not opposite_command:
        return True  # Unknown command type, allow
    
    # Look for recent opposite commands
    for cmd in recent_commands:
        if cmd.command_type == opposite_command:
            # Found opposite command within window
            # Only allow if there's significant net benefit
            if proposed_separation_benefit < MIN_NET_BENEFIT_THRESHOLD:
                logger.warning(
                    f"Oscillation guard blocked {proposed_command_type} for {aircraft_id}: "
                    f"recent {opposite_command} at {cmd.timestamp}, "
                    f"insufficient benefit {proposed_separation_benefit:.2f}nm"
                )
                return False
            else:
                logger.info(
                    f"Oscillation guard allowing {proposed_command_type} for {aircraft_id}: "
                    f"sufficient benefit {proposed_separation_benefit:.2f}nm"
                )
                break
    
    return True


def _classify_command_type(resolution_cmd: ResolutionCommand, ownship: AircraftState) -> str:
    """Classify command type for oscillation tracking."""
    if resolution_cmd.resolution_type == ResolutionType.HEADING_CHANGE:
        if resolution_cmd.new_heading_deg is None:
            return "unknown"
        
        heading_diff = (resolution_cmd.new_heading_deg - ownship.heading_deg + 180) % 360 - 180
        if heading_diff > 0:
            return "turn_right"
        else:
            return "turn_left"
    
    elif resolution_cmd.resolution_type == ResolutionType.ALTITUDE_CHANGE:
        if resolution_cmd.new_altitude_ft is None:
            return "unknown"
            
        if resolution_cmd.new_altitude_ft > ownship.altitude_ft:
            return "climb"
        else:
            return "descend"
    
    return "unknown"


def _estimate_separation_benefit(
    resolution_cmd: ResolutionCommand,
    ownship: AircraftState,
    intruder: AircraftState
) -> float:
    """Estimate separation benefit from resolution command."""
    try:
        # Simplified estimation - can be enhanced with full trajectory prediction
        if resolution_cmd.resolution_type == ResolutionType.HEADING_CHANGE:
            # Estimate based on heading change magnitude
            if resolution_cmd.new_heading_deg is None:
                return 0.0
            heading_change = abs(resolution_cmd.new_heading_deg - ownship.heading_deg)
            return min(heading_change / 10.0, 3.0)  # Up to 3nm benefit for 30deg turn
        
        elif resolution_cmd.resolution_type == ResolutionType.ALTITUDE_CHANGE:
            # Estimate based on altitude change
            if resolution_cmd.new_altitude_ft is None:
                return 0.0
            altitude_change = abs(resolution_cmd.new_altitude_ft - ownship.altitude_ft)
            return min(altitude_change / 1000.0 * 2.0, 5.0)  # Up to 5nm benefit for 2500ft change
        
        return 1.0  # Default modest benefit
        
    except Exception as e:
        logger.error(f"Error estimating separation benefit: {e}")
        return 0.0


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
        
        # Check oscillation guard before proceeding
        command_type = _classify_command_type(resolution_cmd, ownship)
        separation_benefit = _estimate_separation_benefit(resolution_cmd, ownship, intruder)
        
        if not _check_oscillation_guard(ownship.aircraft_id, command_type, separation_benefit):
            logger.warning(f"Oscillation guard blocked resolution for {ownship.aircraft_id}")
            return None
            
        # Validate safety
        if _validate_resolution_safety(resolution_cmd, ownship, intruder):
            resolution_cmd.is_validated = True
            
            # Add to command history for oscillation tracking
            heading_change = None
            altitude_change = None
            
            if resolution_cmd.resolution_type == ResolutionType.HEADING_CHANGE:
                heading_change = resolution_cmd.new_heading_deg - ownship.heading_deg if resolution_cmd.new_heading_deg else 0
            elif resolution_cmd.resolution_type == ResolutionType.ALTITUDE_CHANGE:
                altitude_change = resolution_cmd.new_altitude_ft - ownship.altitude_ft if resolution_cmd.new_altitude_ft else 0
            
            _add_command_to_history(
                ownship.aircraft_id,
                command_type,
                heading_change=heading_change,
                altitude_change=altitude_change,
                separation_benefit=separation_benefit
            )
            
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
            new_speed_kt=None,
            new_altitude_ft=None,
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
            new_heading_deg=None,
            new_speed_kt=None,
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
            new_heading_deg=None,
            new_speed_kt=None,
            new_altitude_ft=None,
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
        dmin_nm, _ = cpa_nm(own_dict, intr_dict)
        
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
            new_heading_deg=None,
            new_speed_kt=None,
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


def apply_resolution(
    bs: Any, 
    cs: str, 
    advise: ResolveOut, 
    next_wpt: Optional[str] = None
) -> bool:
    """Apply validated LLM resolution advice to BlueSky.
    
    Args:
        bs: BlueSky client instance
        cs: Aircraft callsign
        advise: LLM resolution advice
        next_wpt: Next waypoint for return after heading change
        
    Returns:
        True if resolution applied successfully
    """
    import threading
    import time
    
    try:
        if advise.action == "turn":
            heading_deg = advise.params.get("heading_deg")
            if heading_deg is None:
                logger.error("Turn action missing heading_deg parameter")
                return False
                
            # Execute heading change
            success = bs.set_heading(cs, heading_deg)
            if not success:
                return False
                
            # Schedule return to next waypoint after hold time
            hold_min = advise.params.get("hold_min", 4)  # Default 4 minutes
            if next_wpt and hold_min > 0:
                def delayed_direct():
                    time.sleep(hold_min * 60)  # Convert to seconds
                    bs.direct_to_waypoint(cs, next_wpt)
                    logger.info(f"Returned {cs} direct to {next_wpt} after {hold_min} min hold")
                
                # Execute in background thread
                threading.Thread(target=delayed_direct, daemon=True).start()
                
            logger.info(f"Applied turn resolution: {cs} heading {heading_deg}°")
            return True
            
        elif advise.action in ("climb", "descend"):
            target_ft = advise.params.get("target_ft")
            if target_ft is None:
                # Try delta_ft parameter
                delta_ft = advise.params.get("delta_ft")
                if delta_ft is None:
                    logger.error(f"{advise.action} action missing target_ft or delta_ft parameter")
                    return False
                # Calculate target from current altitude (would need ownship state)
                logger.warning("Using delta_ft without current altitude reference")
                return False
                
            success = bs.set_altitude(cs, target_ft)
            logger.info(f"Applied altitude resolution: {cs} to {target_ft} ft")
            return success
            
        else:
            logger.error(f"Unknown resolution action: {advise.action}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to apply resolution for {cs}: {e}")
        return False


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
    traffic: List[AircraftState]
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
    return True  # Placeholder return


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
    return f"# TODO: {aircraft_id}"  # Placeholder return
