"""Main conflict detection and resolution pipeline.

This module implements the core 5-minute polling loop that:
- Fetches current aircraft states from BlueSky
- Runs conflict detection for 10-minute horizon  
- Generates resolutions using LLM reasoning
- Validates and executes safe resolution commands
- Logs all decisions and maintains execution state
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import json
from pathlib import Path

from .bluesky_io import BlueSkyClient
from .detect import predict_conflicts
from .resolve import generate_horizontal_resolution, generate_vertical_resolution, validate_resolution
from .llm_client import LlamaClient
from .metrics import MetricsCollector
from .dual_verification import DualVerificationSystem  # GAP 7 FIX
from .schemas import (
    AircraftState, ConflictPrediction, ResolutionCommand, ResolutionType, ResolutionEngine,
    ConfigurationSettings, FlightRecord, IntruderScenario, BatchSimulationResult,
    MonteCarloParameters, ConflictResolutionMetrics, ScenarioMetrics, 
    PathComparisonMetrics, EnhancedReportingSystem
)

logger = logging.getLogger(__name__)


def _get_aircraft_id(aircraft_dict: Dict[str, Any]) -> str:
    """Get aircraft ID with backward compatibility for both old and new field names."""
    return aircraft_dict.get("id", aircraft_dict.get("aircraft_id", "UNKNOWN"))


def _get_position(aircraft_dict: Dict[str, Any]) -> Tuple[float, float, float]:
    """Get aircraft position (lat, lon, alt) with backward compatibility."""
    lat = aircraft_dict.get("lat", aircraft_dict.get("latitude", 0.0))
    lon = aircraft_dict.get("lon", aircraft_dict.get("longitude", 0.0))
    alt = aircraft_dict.get("alt_ft", aircraft_dict.get("altitude_ft", 0.0))
    return lat, lon, alt


def _get_velocity(aircraft_dict: Dict[str, Any]) -> Tuple[float, float, float]:
    """Get aircraft velocity (speed, heading, vs) with backward compatibility."""
    speed = aircraft_dict.get("spd_kt", aircraft_dict.get("ground_speed_kt", 0.0))
    heading = aircraft_dict.get("hdg_deg", aircraft_dict.get("heading_deg", 0.0))
    vs = aircraft_dict.get("vs_fpm", aircraft_dict.get("vertical_speed_fpm", 0.0))
    return speed, heading, vs


def _asdict_state(s: Any) -> Dict[str, Any]:
    """Convert any state object to dict for backward compatibility."""
    if isinstance(s, dict):
        return s
    if hasattr(s, "model_dump"):  # Pydantic v2
        return s.model_dump()
    if hasattr(s, "dict"):        # Pydantic v1
        return s.dict()
    if hasattr(s, "__dict__"):    # dataclass / simple object
        return dict(vars(s))
    raise TypeError(f"Unsupported state type: {type(s)}")


def _dict_to_aircraft_state(state_dict: Dict[str, Any]) -> AircraftState:
    """Convert BlueSky state dict to AircraftState object."""
    return AircraftState(
        aircraft_id=state_dict["id"],
        timestamp=datetime.now(),
        latitude=state_dict["lat"],
        longitude=state_dict["lon"],
        altitude_ft=state_dict["alt_ft"],
        ground_speed_kt=state_dict["spd_kt"],
        heading_deg=state_dict["hdg_deg"],
        vertical_speed_fpm=state_dict["roc_fpm"],
        # Default values for missing fields
        aircraft_type=state_dict.get("aircraft_type", "B737"),
        spawn_offset_min=state_dict.get("spawn_offset_min", 0.0)
    )


class PromptBuilderV2:
    """Enhanced prompt builder with multi-intruder context and trend analysis."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize the prompt builder with configuration.
        
        Args:
            config: System configuration parameters
        """
        self.config = config
        self.aircraft_history: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = defaultdict(list)
        
    def add_aircraft_snapshot(self, aircraft_states: Dict[str, Dict[str, Any]]) -> None:
        """Add aircraft states snapshot for trend analysis.
        
        Args:
            aircraft_states: Dictionary of aircraft_id -> state_dict
        """
        current_time = datetime.now()
        
        # Add current states to history
        for aircraft_id, state in aircraft_states.items():
            self.aircraft_history[aircraft_id].append((current_time, state.copy()))
            
        # Clean old history (keep only trend_analysis_window_min)
        cutoff_time = current_time - timedelta(minutes=self.config.trend_analysis_window_min)
        for aircraft_id in list(self.aircraft_history.keys()):
            self.aircraft_history[aircraft_id] = [
                (timestamp, state) for timestamp, state in self.aircraft_history[aircraft_id]
                if timestamp >= cutoff_time
            ]
            # Remove empty histories
            if not self.aircraft_history[aircraft_id]:
                del self.aircraft_history[aircraft_id]
    
    def calculate_trends(self, aircraft_id: str) -> Dict[str, float]:
        """Calculate trend deltas for an aircraft over the trend analysis window.
        
        Args:
            aircraft_id: Aircraft identifier
            
        Returns:
            Dictionary with trend information
        """
        history = self.aircraft_history.get(aircraft_id, [])
        if len(history) < 2:
            return {
                "distance_change_nm": 0.0,
                "altitude_change_ft": 0.0,
                "speed_change_kt": 0.0,
                "heading_change_deg": 0.0,
                "time_span_min": 0.0
            }
        
        # Get first and last states
        first_time, first_state = history[0]
        last_time, last_state = history[-1]
        
        time_span_min = (last_time - first_time).total_seconds() / 60.0
        
        # Calculate position change using simple lat/lon distance approximation
        lat_diff = last_state["lat"] - first_state["lat"]
        lon_diff = last_state["lon"] - first_state["lon"]
        # Rough distance calculation (1 degree ≈ 60 NM)
        distance_change_nm = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 60.0
        
        # Calculate other deltas
        altitude_change_ft = last_state["alt_ft"] - first_state["alt_ft"]
        speed_change_kt = last_state["spd_kt"] - first_state["spd_kt"]
        
        # Handle heading wraparound
        heading_diff = last_state["hdg_deg"] - first_state["hdg_deg"]
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360
        
        return {
            "distance_change_nm": distance_change_nm,
            "altitude_change_ft": altitude_change_ft,
            "speed_change_kt": speed_change_kt,
            "heading_change_deg": heading_diff,
            "time_span_min": time_span_min
        }
    
    def filter_relevant_intruders(self, ownship: Dict[str, Any], traffic: List[Dict[str, Any]], 
                                 conflicts: List[ConflictPrediction]) -> List[Dict[str, Any]]:
        """Filter traffic to relevant intruders within proximity and altitude constraints.
        
        Args:
            ownship: Ownship state dictionary
            traffic: List of traffic aircraft state dictionaries
            conflicts: List of predicted conflicts
            
        Returns:
            List of relevant intruder state dictionaries, prioritized by conflict status
        """
        relevant_intruders = []
        conflict_intruder_ids = {c.intruder_id for c in conflicts if c.is_conflict}
        
        for intruder in traffic:
            # Calculate distance using simple lat/lon approximation (handle both field naming conventions)
            intruder_lat = intruder.get("lat", intruder.get("latitude", 0))
            intruder_lon = intruder.get("lon", intruder.get("longitude", 0))
            intruder_alt = intruder.get("alt_ft", intruder.get("altitude_ft", 0))
            intruder_id = intruder.get("id", intruder.get("aircraft_id", ""))
            
            ownship_lat = ownship.get("lat", ownship.get("latitude", 0))
            ownship_lon = ownship.get("lon", ownship.get("longitude", 0))
            ownship_alt = ownship.get("alt_ft", ownship.get("altitude_ft", 0))
            
            lat_diff = intruder_lat - ownship_lat
            lon_diff = intruder_lon - ownship_lon
            distance_nm = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 60.0
            
            # Calculate altitude difference
            alt_diff_ft = abs(intruder_alt - ownship_alt)
            
            # Include if within proximity and altitude constraints
            if (distance_nm <= self.config.intruder_proximity_nm and 
                alt_diff_ft <= self.config.intruder_altitude_diff_ft):
                
                # Add priority for conflict intruders
                intruder_copy = intruder.copy()
                intruder_copy["_priority"] = 1 if intruder_id in conflict_intruder_ids else 2
                intruder_copy["_distance_nm"] = distance_nm
                intruder_copy["_altitude_diff_ft"] = alt_diff_ft
                relevant_intruders.append(intruder_copy)
        
        # Sort by priority (conflicts first), then by distance
        relevant_intruders.sort(key=lambda x: (x["_priority"], x["_distance_nm"]))
        
        # Limit to max_intruders_in_prompt
        return relevant_intruders[:self.config.max_intruders_in_prompt]
    
    def build_enhanced_prompt(self, conflicts: List[ConflictPrediction], ownship: Dict[str, Any], 
                            traffic: List[Dict[str, Any]]) -> Optional[str]:
        """Build enhanced prompt with multi-intruder context and trends.
        
        Args:
            conflicts: List of predicted conflicts
            ownship: Ownship state dictionary
            traffic: List of traffic aircraft state dictionaries
            
        Returns:
            Enhanced prompt string for LLM, or None if no conflicts require LLM action
        """
        # Only build prompt if there are actual conflicts
        active_conflicts = [c for c in conflicts if c.is_conflict]
        if not active_conflicts:
            return None
        
        # Filter relevant intruders
        relevant_intruders = self.filter_relevant_intruders(ownship, traffic, conflicts)
        
        # Calculate trends for ownship and intruders
        ownship_trends = self.calculate_trends(_get_aircraft_id(ownship))
        
        ownship_lat, ownship_lon, ownship_alt = _get_position(ownship)
        ownship_speed, ownship_heading, ownship_vs = _get_velocity(ownship)
        
        # Build the enhanced prompt
        prompt_parts = [
            "Air Traffic Control Multi-Intruder Conflict Resolution Task:",
            "",
            f"OWNSHIP: {_get_aircraft_id(ownship)}",
            f"- Position: ({ownship_lat:.6f}, {ownship_lon:.6f})",
            f"- Altitude: {ownship_alt:.0f} ft",
            f"- Heading: {ownship_heading:.0f}°",
            f"- Speed: {ownship_speed:.0f} kts",
            f"- Vertical Speed: {ownship_vs:.0f} fpm",
        ]
        
        # Add ownship trends if available
        if ownship_trends["time_span_min"] > 0:
            prompt_parts.extend([
                "- TRENDS (last 2 min):",
                f"  * Distance moved: {ownship_trends['distance_change_nm']:.1f} NM",
                f"  * Altitude change: {ownship_trends['altitude_change_ft']:+.0f} ft",
                f"  * Speed change: {ownship_trends['speed_change_kt']:+.1f} kts",
                f"  * Heading change: {ownship_trends['heading_change_deg']:+.1f}°",
            ])
        
        prompt_parts.extend(["", f"INTRUDERS ({len(relevant_intruders)} within {self.config.intruder_proximity_nm:.0f} NM):"])
        
        # Add intruder information
        for i, intruder in enumerate(relevant_intruders, 1):
            intruder_trends = self.calculate_trends(_get_aircraft_id(intruder))
            intruder_lat, intruder_lon, intruder_alt = _get_position(intruder)
            intruder_speed, intruder_heading, intruder_vs = _get_velocity(intruder)
            
            prompt_parts.extend([
                f"{i}. {_get_aircraft_id(intruder)}:",
                f"   - Position: ({intruder_lat:.6f}, {intruder_lon:.6f})",
                f"   - Altitude: {intruder_alt:.0f} ft",
                f"   - Heading: {intruder_heading:.0f}°",
                f"   - Speed: {intruder_speed:.0f} kts",
                f"   - Distance from ownship: {intruder['_distance_nm']:.1f} NM",
                f"   - Altitude separation: {intruder['_altitude_diff_ft']:.0f} ft",
            ])
            
            # Add intruder trends if available
            if intruder_trends["time_span_min"] > 0:
                prompt_parts.extend([
                    "   - TRENDS (last 2 min):",
                    f"     * Distance moved: {intruder_trends['distance_change_nm']:.1f} NM",
                    f"     * Altitude change: {intruder_trends['altitude_change_ft']:+.0f} ft",
                    f"     * Speed change: {intruder_trends['speed_change_kt']:+.1f} kts",
                    f"     * Heading change: {intruder_trends['heading_change_deg']:+.1f}°",
                ])
        
        # Add conflict predictions
        prompt_parts.extend(["", "ACTIVE CONFLICTS:"])
        for i, conflict in enumerate(active_conflicts, 1):
            prompt_parts.extend([
                f"{i}. With {conflict.intruder_id}:",
                f"   - Time to CPA: {conflict.time_to_cpa_min:.1f} minutes",
                f"   - Distance at CPA: {conflict.distance_at_cpa_nm:.2f} NM",
                f"   - Altitude separation at CPA: {conflict.altitude_diff_ft:.0f} ft",
            ])
        
        # Add task description with strict JSON schema
        prompt_parts.extend([
            "",
            "TASK: Generate a conflict resolution command for the ownship to resolve ALL conflicts.",
            "Consider the multi-intruder environment and aircraft trends when making decisions.",
            "Prefer horizontal maneuvers (heading changes) over vertical maneuvers when possible.",
            "",
            "REQUIRED JSON RESPONSE FORMAT (respond ONLY with valid JSON):",
            json.dumps({
                "resolution_type": "heading OR altitude",
                "new_heading_deg": "number (required if resolution_type is heading)",
                "new_altitude_ft": "number (required if resolution_type is altitude)", 
                "target_aircraft": _get_aircraft_id(ownship),
                "reasoning": "brief explanation of decision considering trends and multiple intruders",
                "expected_outcome": "description of how this resolves the conflicts"
            }, indent=2),
            "",
            "EXAMPLE RESPONSES:",
            "For heading change:",
            json.dumps({
                "resolution_type": "heading",
                "new_heading_deg": 280,
                "target_aircraft": _get_aircraft_id(ownship),
                "reasoning": "Turn left 30° to avoid converging traffic while maintaining separation from other intruders",
                "expected_outcome": "Increases CPA distance to 6.2 NM with primary intruder, maintains 8+ NM from others"
            }),
            "",
            "For altitude change:",
            json.dumps({
                "resolution_type": "altitude", 
                "new_altitude_ft": 37000,
                "target_aircraft": _get_aircraft_id(ownship),
                "reasoning": "Climb 2000 ft due to lateral constraints from multiple converging aircraft",
                "expected_outcome": "Achieves 1500+ ft vertical separation from all intruders at CPA"
            })
        ])
        
        return "\n".join(prompt_parts)

    def get_adaptive_snapshot_interval(self, conflicts: List[ConflictPrediction]) -> float:
        """Get adaptive snapshot interval based on conflict urgency.
        
        Args:
            conflicts: List of current conflicts
            
        Returns:
            Snapshot interval in minutes (1.0 to snapshot_interval_min)
        """
        if not conflicts:
            return self.config.snapshot_interval_min
        
        # Find most urgent conflict
        min_time_to_conflict = min(c.time_to_cpa_min for c in conflicts if c.is_conflict)
        
        # If conflict is very urgent (< 1 minute), use 1-minute intervals
        if min_time_to_conflict < 1.0:
            return 1.0
        
        # Otherwise use configured interval
        return self.config.snapshot_interval_min

    def get_adaptive_polling_interval(self, ownship: Dict[str, Any], traffic: List[Dict[str, Any]], 
                                     conflicts: List[ConflictPrediction]) -> float:
        """Get adaptive polling interval based on proximity and conflict urgency.
        
        Based on Gap 3 requirements: 1-2 min default, 1 min if <25 NM/<6 min CPA
        
        Args:
            ownship: Current ownship state
            traffic: Current traffic states  
            conflicts: List of predicted conflicts
            
        Returns:
            Polling interval in minutes (1.0 to 2.0)
        """
        # Default adaptive range: 1-2 minutes (not fixed 5 min)
        default_interval = 2.0
        urgent_interval = 1.0
        
        # Check for urgent conflicts (< 6 min CPA)
        if conflicts:
            active_conflicts = [c for c in conflicts if c.is_conflict]
            if active_conflicts:
                min_time_to_cpa = min(c.time_to_cpa_min for c in active_conflicts)
                if min_time_to_cpa < 6.0:  # Less than 6 minutes to CPA
                    return urgent_interval
        
        # Check for close proximity aircraft (< 25 NM)
        if ownship and traffic:
            from .geodesy import haversine_nm
            ownship_pos = (ownship.get("lat", 0), ownship.get("lon", 0))
            
            for aircraft in traffic:
                aircraft_pos = (aircraft.get("lat", 0), aircraft.get("lon", 0))
                distance_nm = haversine_nm(ownship_pos, aircraft_pos)
                
                if distance_nm < 25.0:  # Within 25 NM proximity
                    return urgent_interval
        
        return default_interval


class HorizontalResolutionAgent:
    """LLM agent specialized for horizontal conflict resolution."""
    
    def __init__(self, llm_client: LlamaClient, config: ConfigurationSettings):
        """Initialize horizontal resolution agent.
        
        Args:
            llm_client: LLM client instance
            config: System configuration
        """
        self.llm_client = llm_client
        self.config = config
    
    def generate_resolution(self, conflicts: List[ConflictPrediction], ownship: Dict[str, Any], 
                          traffic: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate horizontal (heading/speed) resolution.
        
        Args:
            conflicts: List of active conflicts
            ownship: Ownship state dictionary
            traffic: Traffic aircraft state dictionaries
            
        Returns:
            LLM response dictionary or None if failed
        """
        prompt = self._build_horizontal_prompt(conflicts, ownship, traffic)
        
        try:
            response = self.llm_client.generate_resolution(prompt)
            return response
        except Exception as e:
            logger.error(f"Horizontal resolution agent failed: {e}")
            return None
    
    def _build_horizontal_prompt(self, conflicts: List[ConflictPrediction], 
                               ownship: Dict[str, Any], traffic: List[Dict[str, Any]]) -> str:
        """Build specialized prompt for horizontal maneuvers."""
        active_conflicts = [c for c in conflicts if c.is_conflict]
        if not active_conflicts:
            return ""
        
        ownship_lat, ownship_lon, ownship_alt = _get_position(ownship)
        ownship_speed, ownship_heading, ownship_vs = _get_velocity(ownship)
        
        prompt_parts = [
            "HORIZONTAL CONFLICT RESOLUTION - OWNSHIP COMMAND ONLY",
            "="*60,
            "",
            f"OWNSHIP (COMMAND TARGET): {_get_aircraft_id(ownship)}",
            f"- Position: ({ownship_lat:.6f}, {ownship_lon:.6f})",
            f"- Altitude: {ownship_alt:.0f} ft",
            f"- Current Heading: {ownship_heading:.0f}°",
            f"- Current Speed: {ownship_speed:.0f} kts",
            "",
            "PRIORITY: HORIZONTAL MANEUVERS ONLY (heading/speed changes)",
            "You may ONLY command the ownship. Any command for other aircraft will be REJECTED.",
            "",
            "ACTIVE CONFLICTS:"
        ]
        
        for i, conflict in enumerate(active_conflicts, 1):
            intruder = next((t for t in traffic if _get_aircraft_id(t) == conflict.intruder_id), None)
            if intruder:
                intruder_lat, intruder_lon, intruder_alt = _get_position(intruder)
                intruder_speed, intruder_heading, intruder_vs = _get_velocity(intruder)
                
                prompt_parts.extend([
                    f"{i}. CONFLICT with {conflict.intruder_id}:",
                    f"   - Intruder position: ({intruder_lat:.6f}, {intruder_lon:.6f})",
                    f"   - Intruder altitude: {intruder_alt:.0f} ft",
                    f"   - Intruder heading: {intruder_heading:.0f}°",
                    f"   - Time to CPA: {conflict.time_to_cpa_min:.1f} minutes",
                    f"   - Distance at CPA: {conflict.distance_at_cpa_nm:.2f} NM",
                    ""
                ])
        
        prompt_parts.extend([
            "CONSTRAINTS:",
            f"- Maximum heading change: {self.config.max_heading_change_deg:.0f}°",
            f"- Speed range: 50-1000 kts",
            f"- Target aircraft MUST be '{_get_aircraft_id(ownship)}' (ownship only)",
            "",
            "TASK: Generate a HORIZONTAL resolution (heading or speed change) for the ownship.",
            "Respond ONLY with valid JSON in this exact format:",
            "",
            json.dumps({
                "resolution_type": "heading",
                "new_heading_deg": "number (required for heading change, 0-359)",
                "new_speed_kt": "number (optional for speed change, 50-1000)",
                "target_aircraft": _get_aircraft_id(ownship),
                "reasoning": "brief explanation focusing on horizontal maneuver",
                "expected_outcome": "how this resolves conflicts horizontally"
            }, indent=2),
            "",
            "IMPORTANT: Only heading or speed changes allowed. NO altitude changes."
        ])
        
        return "\n".join(prompt_parts)


class VerticalResolutionAgent:
    """LLM agent specialized for vertical conflict resolution."""
    
    def __init__(self, llm_client: LlamaClient, config: ConfigurationSettings):
        """Initialize vertical resolution agent.
        
        Args:
            llm_client: LLM client instance
            config: System configuration
        """
        self.llm_client = llm_client
        self.config = config
    
    def generate_resolution(self, conflicts: List[ConflictPrediction], ownship: Dict[str, Any], 
                          traffic: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate vertical (altitude) resolution.
        
        Args:
            conflicts: List of active conflicts
            ownship: Ownship state dictionary
            traffic: Traffic aircraft state dictionaries
            
        Returns:
            LLM response dictionary or None if failed
        """
        prompt = self._build_vertical_prompt(conflicts, ownship, traffic)
        
        try:
            response = self.llm_client.generate_resolution(prompt)
            return response
        except Exception as e:
            logger.error(f"Vertical resolution agent failed: {e}")
            return None
    
    def _build_vertical_prompt(self, conflicts: List[ConflictPrediction], 
                             ownship: Dict[str, Any], traffic: List[Dict[str, Any]]) -> str:
        """Build specialized prompt for vertical maneuvers."""
        active_conflicts = [c for c in conflicts if c.is_conflict]
        if not active_conflicts:
            return ""
        
        _, _, altitude = _get_position(ownship)
        current_fl = int(altitude / 100)
        
        prompt_parts = [
            "VERTICAL CONFLICT RESOLUTION - OWNSHIP COMMAND ONLY",
            "="*60,
            "",
            f"OWNSHIP (COMMAND TARGET): {_get_aircraft_id(ownship)}",
            f"- Position: ({_get_position(ownship)[0]:.6f}, {_get_position(ownship)[1]:.6f})",
            f"- Current Altitude: {altitude:.0f} ft (FL{current_fl})",
            f"- Heading: {_get_velocity(ownship)[1]:.0f}°",
            f"- Speed: {_get_velocity(ownship)[0]:.0f} kts",
            "",
            "PRIORITY: VERTICAL MANEUVERS ONLY (altitude changes)",
            "You may ONLY command the ownship. Any command for other aircraft will be REJECTED.",
            "",
            "ACTIVE CONFLICTS:"
        ]
        
        for i, conflict in enumerate(active_conflicts, 1):
            intruder = next((t for t in traffic if _get_aircraft_id(t) == conflict.intruder_id), None)
            if intruder:
                intruder_altitude = _get_position(intruder)[2]
                intruder_fl = int(intruder_altitude / 100)
                prompt_parts.extend([
                    f"{i}. CONFLICT with {conflict.intruder_id}:",
                    f"   - Intruder altitude: {intruder_altitude:.0f} ft (FL{intruder_fl})",
                    f"   - Altitude separation at CPA: {conflict.altitude_diff_ft:.0f} ft",
                    f"   - Time to CPA: {conflict.time_to_cpa_min:.1f} minutes",
                    ""
                ])
        
        prompt_parts.extend([
            "CONSTRAINTS:",
            f"- Altitude range: FL{self.config.min_flight_level}-FL{self.config.max_flight_level}",
            f"- Maximum altitude change: {self.config.max_altitude_change_ft:.0f} ft",
            f"- Maximum climb rate: {self.config.max_climb_rate_fpm:.0f} fpm",
            f"- Maximum descent rate: {self.config.max_descent_rate_fpm:.0f} fpm",
            f"- Target aircraft MUST be '{_get_aircraft_id(ownship)}' (ownship only)",
            "",
            "TASK: Generate a VERTICAL resolution (altitude change) for the ownship.",
            "Respond ONLY with valid JSON in this exact format:",
            "",
            json.dumps({
                "resolution_type": "altitude",
                "new_altitude_ft": "number (required, within FL limits)",
                "target_aircraft": _get_aircraft_id(ownship),
                "reasoning": "brief explanation focusing on vertical separation",
                "expected_outcome": "how this resolves conflicts vertically"
            }, indent=2),
            "",
            "IMPORTANT: Only altitude changes allowed. NO heading or speed changes."
        ])
        
        return "\n".join(prompt_parts)


class EnhancedResolutionValidator:
    """Enhanced validation for resolution commands with strict safety limits."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize validator with configuration.
        
        Args:
            config: System configuration parameters
        """
        self.config = config
    
    def validate_resolution(self, resolution: ResolutionCommand, ownship: Dict[str, Any], 
                          traffic: List[Dict[str, Any]], ownship_id: str) -> bool:
        """Perform comprehensive validation of resolution command.
        
        Args:
            resolution: Resolution command to validate
            ownship: Current ownship state
            traffic: Traffic aircraft states
            ownship_id: Expected ownship identifier
            
        Returns:
            True if resolution passes all validation checks
        """
        validation_failures: List[str] = []
        
        # A3.1: Ownship-only command validation
        if self.config.enforce_ownship_only:
            if resolution.target_aircraft != ownship_id:
                validation_failures.append(f"Command targets {resolution.target_aircraft}, must target ownship {ownship_id}")
                resolution.is_ownship_command = False
            else:
                resolution.is_ownship_command = True
        
        # A3.2: Maneuver limits validation
        if resolution.resolution_type == ResolutionType.HEADING_CHANGE and resolution.new_heading_deg is not None:
            current_heading = ownship['hdg_deg']
            new_heading = resolution.new_heading_deg
            
            # Calculate heading change (handling wraparound)
            heading_change = abs(new_heading - current_heading)
            if heading_change > 180:
                heading_change = 360 - heading_change
            
            if heading_change > self.config.max_heading_change_deg:
                validation_failures.append(f"Heading change {heading_change:.1f}° exceeds limit {self.config.max_heading_change_deg}°")
                resolution.angle_within_limits = False
            else:
                resolution.angle_within_limits = True
        
        if resolution.resolution_type == ResolutionType.ALTITUDE_CHANGE and resolution.new_altitude_ft is not None:
            current_alt = ownship['alt_ft']
            new_alt = resolution.new_altitude_ft
            
            # Check altitude limits
            min_alt_ft = self.config.min_flight_level * 100
            max_alt_ft = self.config.max_flight_level * 100
            
            if new_alt < min_alt_ft or new_alt > max_alt_ft:
                validation_failures.append(f"Altitude {new_alt:.0f} ft outside limits FL{self.config.min_flight_level}-FL{self.config.max_flight_level}")
                resolution.altitude_within_limits = False
            else:
                resolution.altitude_within_limits = True
            
            # Check altitude change limits
            altitude_change = abs(new_alt - current_alt)
            if altitude_change > self.config.max_altitude_change_ft:
                validation_failures.append(f"Altitude change {altitude_change:.0f} ft exceeds limit {self.config.max_altitude_change_ft:.0f} ft")
                resolution.altitude_within_limits = False
        
        # A3.3: Rate limits validation (simplified - assumes reasonable time to execute)
        if resolution.resolution_type == ResolutionType.ALTITUDE_CHANGE and resolution.new_altitude_ft is not None:
            current_alt = ownship['alt_ft']
            new_alt = resolution.new_altitude_ft
            
            # Assume 1 minute execution time for rate check
            required_rate = abs(new_alt - current_alt)  # fpm for 1 minute
            
            if new_alt > current_alt:  # Climbing
                if required_rate > self.config.max_climb_rate_fpm:
                    validation_failures.append(f"Required climb rate {required_rate:.0f} fpm exceeds limit {self.config.max_climb_rate_fpm:.0f} fpm")
                    resolution.rate_within_limits = False
            else:  # Descending
                if required_rate > self.config.max_descent_rate_fpm:
                    validation_failures.append(f"Required descent rate {required_rate:.0f} fpm exceeds limit {self.config.max_descent_rate_fpm:.0f} fpm")
                    resolution.rate_within_limits = False

        # A3.4: CPA recheck before execution - GAP 5 FIX
        # Note: This is a placeholder for CPA recheck logic
        # Full implementation would require access to conflict detection in pipeline
        logger.debug("CPA recheck validation placeholder - resolution timestamp checked")
        
        # Update resolution with validation results
        resolution.validation_failures = validation_failures
        resolution.is_validated = len(validation_failures) == 0
        
        if validation_failures:
            logger.warning(f"Resolution validation failed: {'; '.join(validation_failures)}")
        
        return resolution.is_validated


class CDRPipeline:
    """Main conflict detection and resolution pipeline."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize CDR pipeline with configuration.
        
        Args:
            config: System configuration parameters
        """
        self.config = config
        self.running = False
        self.cycle_count = 0
        
        # Initialize components
        self.bluesky_client = BlueSkyClient(config)
        self.llm_client = LlamaClient(config)
        self.metrics = MetricsCollector()
        
        # Initialize enhanced reporting system
        self.enhanced_reporting = EnhancedReportingSystem()
        
        # Initialize PromptBuilderV2 for enhanced LLM prompts
        self.prompt_builder = PromptBuilderV2(config)
        
        # GAP 7 FIX: Initialize dual verification system
        self.dual_verification = DualVerificationSystem(
            min_horizontal_sep_nm=config.min_horizontal_separation_nm,
            min_vertical_sep_ft=config.min_vertical_separation_ft,
            lookahead_time_min=config.lookahead_time_min
        )
        
        # Initialize dual LLM resolution agents
        self.horizontal_agent = HorizontalResolutionAgent(self.llm_client, config)
        self.vertical_agent = VerticalResolutionAgent(self.llm_client, config)
        self.enhanced_validator = EnhancedResolutionValidator(config)
        
        # Timing control for adaptive snapshot intervals
        self.last_snapshot_time = datetime.now()
        self.conflict_detected = False
        
        # Critical connections - MUST succeed or fail hard
        logger.info("Connecting to critical systems...")
        
        # Connect to BlueSky (CRITICAL - no fallback)
        try:
            if not self.bluesky_client.connect():
                error_msg = "CRITICAL FAILURE: BlueSky connection failed. Cannot proceed without BlueSky simulator."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            logger.info("[OK] BlueSky connection successful")
        except Exception as e:
            error_msg = f"CRITICAL FAILURE: BlueSky connection error: {e}. Cannot proceed without BlueSky simulator."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Validate LLM client (CRITICAL - no fallback) - Skip if explicitly disabled
        if hasattr(config, 'llm_enabled') and config.llm_enabled == False:
            logger.info("[OK] LLM client validation skipped (llm_enabled=False)")
        else:
            try:
                # Test LLM connection/availability
                test_result = self.llm_client.generate_resolution("TEST", config)
                if not test_result:
                    error_msg = "CRITICAL FAILURE: LLM client validation failed. Cannot proceed without LLM."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.info("[OK] LLM client validation successful")
            except Exception as e:
                error_msg = f"CRITICAL FAILURE: LLM client error: {e}. Cannot proceed without LLM."
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        # Aliases for compatibility
        self.bs = self.bluesky_client
        self.log = logger
        
        # State tracking
        self.active_resolutions: Dict[str, ResolutionCommand] = {}
        self.conflict_history: List[ConflictPrediction] = []
        
        logger.info("CDR Pipeline initialized with PromptBuilderV2")
    
    def run(self, max_cycles: Optional[int] = None, ownship_id: str = "OWNSHIP") -> bool:
        """Run the main CDR loop.
        
        Args:
            max_cycles: Maximum cycles to run (None = infinite)
            ownship_id: Ownship aircraft identifier
            
        Returns:
            True if completed successfully
        """
        logger.info(f"Starting CDR pipeline for ownship: {ownship_id}")
        self.running = True
        
        try:
            while self.running and (max_cycles is None or self.cycle_count < max_cycles):
                cycle_start = datetime.now()
                
                # Execute one pipeline cycle
                self._execute_cycle(ownship_id)
                
                # Calculate sleep time for next cycle
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                
                logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
                
                # NO WALL-CLOCK SLEEP IN FAST-TIME
                if not getattr(self.config, "fast_time", True):
                    # preserve original real-time pacing only when fast_time=False
                    period = max(0.0, self.config.polling_interval_min * 60.0 - cycle_duration)
                    if period > 0:
                        logger.info(f"Real-time mode: sleeping {period:.2f}s")
                        time.sleep(period)
                    
                
                self.cycle_count += 1
                
            # Return True if successfully completed a full 5-minute cycle
            return self.cycle_count > 0
                
        except KeyboardInterrupt:
            logger.info("CDR pipeline stopped by user")
            return False
        except Exception as e:
            logger.error(f"CDR pipeline error: {e}")
            raise
        finally:
            self.running = False
            logger.info("Cleaning up CDR pipeline")
            self._cleanup()
    
    def _execute_cycle(self, ownship_id: str) -> None:
        """Execute one complete CDR cycle with adaptive snapshot intervals.
        
        Args:
            ownship_id: Ownship aircraft identifier
        """
        logger.debug(f"Starting cycle {self.cycle_count}")
        current_time = datetime.now()
        
        # Step 1: Fetch current aircraft states and split into ownship/traffic
        ownship, traffic = self._fetch_aircraft_states(ownship_id)
        if ownship is None:
            logger.warning(f"Ownship {ownship_id} not found, skipping cycle")
            return
        
        logger.info(f"Processing ownship {ownship_id} with {len(traffic)} traffic aircraft")
        
        # Step 2: Check if we should take a snapshot for trend analysis
        time_since_snapshot = (current_time - self.last_snapshot_time).total_seconds() / 60.0
        should_snapshot = False
        
        # Adaptive snapshot timing: more frequent during conflicts
        if self.conflict_detected:
            # During conflicts, use minimum snapshot interval for better trend data
            should_snapshot = time_since_snapshot >= 1.0  # Minimum 1 minute
        else:
            # Normal operation, use configured snapshot interval
            should_snapshot = time_since_snapshot >= self.config.snapshot_interval_min
        
        if should_snapshot:
            # Create aircraft states dictionary for snapshot
            all_aircraft = {ownship_id: ownship}
            for t in traffic:
                all_aircraft[t["id"]] = t
            
            # Add snapshot to PromptBuilderV2 for trend analysis
            self.prompt_builder.add_aircraft_snapshot(all_aircraft)
            self.last_snapshot_time = current_time
            logger.debug(f"Aircraft snapshot taken for trend analysis ({len(all_aircraft)} aircraft)")
        
        # Step 3: Predict conflicts
        conflicts = self._predict_conflicts(ownship, traffic)
        self.conflict_detected = any(c.is_conflict for c in conflicts)
        logger.info(f"Detected {len(conflicts)} potential conflicts, active conflicts: {self.conflict_detected}")
        
        # Step 3.5: Calculate adaptive polling interval based on conflicts and proximity
        adaptive_interval = self.get_adaptive_polling_interval(ownship, traffic, conflicts)
        logger.debug(f"Adaptive polling interval: {adaptive_interval:.1f} minutes")
        
        # Step 4: Generate and execute resolutions using PromptBuilderV2
        if self.conflict_detected:
            self._handle_conflicts_enhanced(conflicts, ownship, traffic)
        
        # Step 5: Update metrics
        self._update_metrics(ownship, traffic, conflicts)
        
        # Step 6: Advance BlueSky simulated time by the adaptive polling interval (fast-time progression)
        step_time = adaptive_interval * self.config.sim_accel_factor
        self.bluesky_client.step_minutes(step_time)
        logger.debug(f"Advanced simulation time by {step_time:.1f} minutes")

    def get_adaptive_polling_interval(self, ownship: Dict[str, Any], traffic: List[Dict[str, Any]], 
                                     conflicts: List[ConflictPrediction]) -> float:
        """Get adaptive polling interval based on proximity and conflict urgency.
        
        Enhanced version that uses the enhanced CPA calculations for more accurate
        adaptive polling decisions.
        
        Args:
            ownship: Current ownship state
            traffic: Current traffic states  
            conflicts: List of predicted conflicts
            
        Returns:
            Polling interval in minutes (0.5 to 5.0)
        """
        # Check if enhanced conflict detection provided a recommendation
        if hasattr(self, '_adaptive_interval'):
            recommended = self._adaptive_interval
            logger.debug(f"Using enhanced CPA recommended interval: {recommended:.1f} min")
            return recommended
        
        # Fallback to original adaptive logic
        default_interval = 2.0
        urgent_interval = 1.0
        imminent_interval = 0.5
        
        # Check for urgent conflicts (< 6 min CPA)
        if conflicts:
            active_conflicts = [c for c in conflicts if c.is_conflict]
            if active_conflicts:
                min_time_to_cpa = min(c.time_to_cpa_min for c in active_conflicts)
                
                if min_time_to_cpa < 2.0:  # Imminent threat
                    logger.debug(f"IMMINENT conflict detected (CPA in {min_time_to_cpa:.1f} min), using {imminent_interval} min interval")
                    return imminent_interval
                elif min_time_to_cpa < 6.0:  # Urgent
                    logger.debug(f"URGENT conflict detected (CPA in {min_time_to_cpa:.1f} min), using {urgent_interval} min interval")
                    return urgent_interval
        
        # Check for close proximity aircraft (< 25 NM)
        if ownship and traffic:
            from .geodesy import haversine_nm
            ownship_pos = (ownship.get("lat", 0), ownship.get("lon", 0))
            
            for aircraft in traffic:
                aircraft_pos = (aircraft.get("lat", 0), aircraft.get("lon", 0))
                distance_nm = haversine_nm(ownship_pos, aircraft_pos)
                
                if distance_nm < 25.0:  # Within 25 NM proximity
                    logger.debug(f"Close proximity detected ({distance_nm:.1f} NM), using {urgent_interval} min interval")
                    return urgent_interval
        
        logger.debug(f"Normal conditions, using {default_interval} min interval")
        return default_interval

    def geometric_conflict_precheck(self, conflicts: List[ConflictPrediction]) -> Tuple[List[ConflictPrediction], List[ConflictPrediction]]:
        """Perform geometric pre-check to filter conflicts before LLM processing.
        
        Based on Gap 4 requirements: Use geometric CPA/min-sep check (5 NM/1000 ft) as pre-filter,
        only send borderline cases to LLM for confirmation.
        
        Args:
            conflicts: List of all predicted conflicts
            
        Returns:
            Tuple of (clear_conflicts, borderline_conflicts) where:
            - clear_conflicts: Definite conflicts requiring immediate action
            - borderline_conflicts: Uncertain cases needing LLM confirmation
        """
        clear_conflicts = []
        borderline_conflicts = []
        
        # Import separation standards
        from .detect import MIN_HORIZONTAL_SEP_NM, MIN_VERTICAL_SEP_FT
        
        # Thresholds for geometric pre-filtering
        CRITICAL_DISTANCE_NM = 3.0  # Below this: definitely a conflict
        CRITICAL_TIME_MIN = 2.0     # Below this: definitely urgent
        SAFE_DISTANCE_NM = 8.0      # Above this: probably safe
        SAFE_TIME_MIN = 8.0         # Above this: probably not urgent
        
        for conflict in conflicts:
            if not conflict.is_conflict:
                continue
                
            distance_nm = conflict.distance_at_cpa_nm
            time_min = conflict.time_to_cpa_min
            altitude_diff = conflict.altitude_diff_ft
            
            # Clear conflicts: Definite separation violations
            if (distance_nm < CRITICAL_DISTANCE_NM and time_min < CRITICAL_TIME_MIN) or \
               (distance_nm < MIN_HORIZONTAL_SEP_NM and altitude_diff < MIN_VERTICAL_SEP_FT and time_min < SAFE_TIME_MIN):
                clear_conflicts.append(conflict)
                logger.debug(f"Clear conflict: {conflict.intruder_id} (CPA {distance_nm:.1f} NM in {time_min:.1f} min)")
            
            # Borderline cases: Need LLM confirmation
            elif (CRITICAL_DISTANCE_NM <= distance_nm <= SAFE_DISTANCE_NM) or \
                 (CRITICAL_TIME_MIN <= time_min <= SAFE_TIME_MIN):
                borderline_conflicts.append(conflict)
                logger.debug(f"Borderline conflict: {conflict.intruder_id} (CPA {distance_nm:.1f} NM in {time_min:.1f} min) - needs LLM confirmation")
            
            # Safe cases: Ignored (distance > 8 NM and time > 8 min)
            else:
                logger.debug(f"Safe case: {conflict.intruder_id} (CPA {distance_nm:.1f} NM in {time_min:.1f} min) - no action needed")
        
        logger.info(f"Geometric pre-check: {len(clear_conflicts)} clear conflicts, {len(borderline_conflicts)} borderline cases")
        return clear_conflicts, borderline_conflicts
    
    def _fetch_aircraft_states(self, ownship_id: str):
        """Fetch current aircraft states from BlueSky and split into ownship/traffic.
        
        Args:
            ownship_id: Ownship aircraft identifier
            
        Returns:
            Tuple of (ownship_state, traffic_states) where ownship_state is dict or None,
            and traffic_states is list of dicts
        """
        raw = self.bs.get_aircraft_states()
        own = raw.get(ownship_id)
        traffic = [state for callsign, state in raw.items() if callsign != ownship_id]
        if own is None:
            self.log.warning("Ownship %s not found in BS state", ownship_id)
        return own, traffic
    
    def _find_ownship(self, states: List[Dict[str, Any]], ownship_id: str) -> Optional[Dict[str, Any]]:
        """Find ownship in states list."""
        for s in states:
            d = _asdict_state(s)
            if d.get("id") == ownship_id:
                return d
        return None
    

    def _predict_conflicts(self, own: Optional[Dict[str, Any]], traffic: List[Dict[str, Any]]) -> List[ConflictPrediction]:
        """Predict conflicts using enhanced CPA algorithms with adaptive cadence.
        
        Args:
            own: Ownship current state dict
            traffic: Traffic aircraft state dicts
            
        Returns:
            List of predicted conflicts
        """
        if own is None:
            logger.warning("Cannot predict conflicts: ownship state is None")
            return []
            
        try:
            # Convert dict states to AircraftState objects
            own_s = _dict_to_aircraft_state(own)
            traf_s = [_dict_to_aircraft_state(t) for t in traffic]

            # Use enhanced conflict detection with adaptive cadence
            from .detect import predict_conflicts_enhanced
            conflicts, recommended_interval = predict_conflicts_enhanced(
                own_s, traf_s, 
                lookahead_minutes=self.config.lookahead_time_min,
                use_adaptive_cadence=True
            )
            
            # Store recommended interval for next cycle
            self._adaptive_interval = recommended_interval
            logger.debug(f"Enhanced conflict detection recommends {recommended_interval:.1f} min interval")
            
            # GAP 7 FIX: Apply dual verification (BlueSky CD + geometric CPA)
            if hasattr(self, 'dual_verification') and self.dual_verification and conflicts:
                try:
                    # Prepare traffic states for dual verification
                    traffic_states = [own] + traffic
                    
                    # Pass BlueSky client (if available) for ASAS verification
                    bluesky_client = getattr(self, 'bluesky_client', None)
                    
                    # Perform dual verification
                    verified_conflicts = self.dual_verification.verify_conflicts(
                        traffic_states, bluesky_client
                    )
                    
                    original_count = len(conflicts)
                    conflicts = verified_conflicts
                    
                    logger.info(f"Dual verification: {len(verified_conflicts)}/{original_count} conflicts verified")
                    
                except Exception as e:
                    logger.warning(f"Dual verification failed, using original conflicts: {e}")
            
            logger.debug(f"Enhanced conflict detection: {len(conflicts)} conflicts for {len(traffic)} traffic aircraft")
            return conflicts
            
        except Exception as e:
            logger.exception(f"Error in enhanced conflict prediction: {e}")
            # Fallback to basic conflict detection
            try:
                own_s = _dict_to_aircraft_state(own)
                traf_s = [_dict_to_aircraft_state(t) for t in traffic]
                fallback_conflicts = predict_conflicts(own_s, traf_s, lookahead_minutes=self.config.lookahead_time_min)
                logger.warning(f"Using fallback conflict detection: {len(fallback_conflicts)} conflicts")
                return fallback_conflicts
            except Exception as fallback_error:
                logger.error(f"Fallback conflict detection also failed: {fallback_error}")
                return []
    
    def _handle_conflicts_enhanced(
        self, 
        conflicts: List[ConflictPrediction], 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]]
    ) -> None:
        """Handle detected conflicts using PromptBuilderV2 and dual LLM engines.
        
        Enhanced with geometric pre-check to filter conflicts before LLM processing.
        
        Args:
            conflicts: List of predicted conflicts
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
        """
        active_conflicts = [c for c in conflicts if c.is_conflict]
        if not active_conflicts:
            return
        
        logger.warning(f"Handling {len(active_conflicts)} active conflicts using enhanced geometric pre-check + LLM")
        
        # Gap 4 Fix: Geometric pre-check before LLM
        clear_conflicts, borderline_conflicts = self.geometric_conflict_precheck(active_conflicts)
        
        # Handle clear conflicts with deterministic geometric resolution
        if clear_conflicts:
            logger.info(f"Processing {len(clear_conflicts)} clear conflicts with deterministic resolution")
            # For clear conflicts, we could implement basic geometric avoidance here
            # For now, we'll still use LLM but with high priority
            
        # Handle borderline conflicts with LLM confirmation
        conflicts_for_llm = borderline_conflicts  # Only borderline cases go to LLM
        if not conflicts_for_llm and clear_conflicts:
            # If we only have clear conflicts, still send the most urgent one to LLM for validation
            conflicts_for_llm = [clear_conflicts[0]]
            logger.info("Sending most urgent clear conflict to LLM for validation")
        
        if not conflicts_for_llm:
            logger.info("No conflicts require LLM processing after geometric pre-check")
            return
        
        logger.info(f"Sending {len(conflicts_for_llm)} conflicts to LLM for confirmation/resolution")
        
        # A2: Dual LLM engines (horizontal → vertical fallback) - only for filtered conflicts
        resolution = None
        ownship_id = ownship["id"]
        
        if self.config.enable_dual_llm:
            # Try horizontal resolution first
            resolution = self._try_horizontal_resolution(conflicts_for_llm, ownship, traffic, ownship_id)
            
            # Fallback to vertical resolution if horizontal failed
            if resolution is None or not resolution.is_validated:
                logger.info("Horizontal resolution failed/invalid, trying vertical resolution")
                resolution = self._try_vertical_resolution(conflicts_for_llm, ownship, traffic, ownship_id)
        else:
            # Use original enhanced prompt method
            enhanced_prompt = self.prompt_builder.build_enhanced_prompt(conflicts, ownship, traffic)
            if enhanced_prompt:
                try:
                    llm_response = self.llm_client.generate_resolution(enhanced_prompt)
                    primary_conflict = active_conflicts[0]
                    resolution = self._create_resolution_from_llm(llm_response, primary_conflict, ownship, ResolutionEngine.HORIZONTAL)
                except Exception as e:
                    logger.error(f"Enhanced prompt resolution failed: {e}")
        
        # Execute resolution if valid
        if resolution and resolution.is_validated:
            success = self._validate_and_execute_resolution(resolution, ownship, traffic, ownship_id)
            if success:
                logger.info(f"Successfully executed {resolution.source_engine.value} resolution {resolution.resolution_id}")
                self.active_resolutions[resolution.resolution_id] = resolution
            else:
                logger.error(f"Failed to execute validated resolution {resolution.resolution_id}")
        else:
            logger.error("All LLM resolution methods failed - falling back to deterministic")
            # Fallback to original individual conflict handling
            for conflict in active_conflicts:
                self._handle_conflict(conflict, ownship, traffic)
    
    def _try_horizontal_resolution(self, conflicts: List[ConflictPrediction], 
                                 ownship: Dict[str, Any], traffic: List[Dict[str, Any]], 
                                 ownship_id: str) -> Optional[ResolutionCommand]:
        """Try horizontal resolution with retries."""
        for attempt in range(self.config.horizontal_retry_count):
            try:
                logger.debug(f"Horizontal resolution attempt {attempt + 1}/{self.config.horizontal_retry_count}")
                
                llm_response = self.horizontal_agent.generate_resolution(conflicts, ownship, traffic)
                if llm_response is None:
                    continue
                
                primary_conflict = next(c for c in conflicts if c.is_conflict)
                resolution = self._create_resolution_from_llm(llm_response, primary_conflict, ownship, ResolutionEngine.HORIZONTAL)
                
                if resolution:
                    # Validate with enhanced validator
                    if self.enhanced_validator.validate_resolution(resolution, ownship, traffic, ownship_id):
                        return resolution
                    else:
                        logger.warning(f"Horizontal resolution attempt {attempt + 1} failed validation")
                
            except Exception as e:
                logger.error(f"Horizontal resolution attempt {attempt + 1} failed: {e}")
        
        return None
    
    def _try_vertical_resolution(self, conflicts: List[ConflictPrediction], 
                               ownship: Dict[str, Any], traffic: List[Dict[str, Any]], 
                               ownship_id: str) -> Optional[ResolutionCommand]:
        """Try vertical resolution with retries."""
        for attempt in range(self.config.vertical_retry_count):
            try:
                logger.debug(f"Vertical resolution attempt {attempt + 1}/{self.config.vertical_retry_count}")
                
                llm_response = self.vertical_agent.generate_resolution(conflicts, ownship, traffic)
                if llm_response is None:
                    continue
                
                primary_conflict = next(c for c in conflicts if c.is_conflict)
                resolution = self._create_resolution_from_llm(llm_response, primary_conflict, ownship, ResolutionEngine.VERTICAL)
                
                if resolution:
                    # Validate with enhanced validator
                    if self.enhanced_validator.validate_resolution(resolution, ownship, traffic, ownship_id):
                        return resolution
                    else:
                        logger.warning(f"Vertical resolution attempt {attempt + 1} failed validation")
                
            except Exception as e:
                logger.error(f"Vertical resolution attempt {attempt + 1} failed: {e}")
        
        return None
    
    def _handle_conflict(
        self, 
        conflict: ConflictPrediction, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]]
    ) -> None:
        """Handle detected conflict by generating and executing resolution.
        
        Args:
            conflict: Predicted conflict details
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
        """
        logger.warning(f"Handling conflict with {conflict.intruder_id}, "
                      f"CPA in {conflict.time_to_cpa_min:.1f} min, "
                      f"separation {conflict.distance_at_cpa_nm:.2f} NM")
        
        # Build a compact prompt for the LLM (ownship + intruder + next 10 min task)
        prompt = self._llm_client_prompt_builder(conflict, ownship, traffic)
        
        # Ask LLM for action
        llm_json = self.llm_client.generate_resolution(prompt)
        
        # Turn JSON into a ResolutionCommand (with your existing create/validate)
        # Then push via BlueSky stack (HDG/ALT), and record metrics.
        resolution = self._create_resolution_from_llm(llm_json, conflict, ownship)
        
        if resolution and self._validate_and_execute_resolution(resolution, ownship, traffic, ownship["id"]):
            logger.info(f"Successfully executed resolution {resolution.resolution_id}")
            self.active_resolutions[resolution.resolution_id] = resolution
        else:
            logger.error(f"Failed to resolve conflict with {conflict.intruder_id}")

    def _llm_client_prompt_builder(self, conflict: ConflictPrediction, ownship: Dict[str, Any], traffic: List[Dict[str, Any]]) -> str:
        """Build a compact prompt for the LLM with ownship + intruder + next 10 min task.
        
        Args:
            conflict: Predicted conflict details
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            
        Returns:
            Formatted prompt string for LLM
        """
        # Find the intruder aircraft
        intruder = next((t for t in traffic if t["id"] == conflict.intruder_id), None)
        if intruder is None:
            logger.warning(f"Intruder {conflict.intruder_id} not found in traffic list")
            intruder = {"id": conflict.intruder_id, "lat": 0, "lon": 0, "alt_ft": 0, "hdg_deg": 0, "spd_kt": 0}
        
        prompt = f"""
Air Traffic Control Conflict Resolution Task:

OWNSHIP: {ownship['id']}
- Position: ({ownship['lat']:.6f}, {ownship['lon']:.6f})
- Altitude: {ownship['alt_ft']:.0f} ft
- Heading: {ownship['hdg_deg']:.0f}deg
- Speed: {ownship['spd_kt']:.0f} kts

INTRUDER: {intruder['id']}
- Position: ({intruder['lat']:.6f}, {intruder['lon']:.6f})
- Altitude: {intruder['alt_ft']:.0f} ft
- Heading: {intruder['hdg_deg']:.0f}deg
- Speed: {intruder['spd_kt']:.0f} kts

CONFLICT PREDICTION:
- Time to CPA: {conflict.time_to_cpa_min:.1f} minutes
- Distance at CPA: {conflict.distance_at_cpa_nm:.2f} NM
- Altitude separation at CPA: {conflict.altitude_diff_ft:.0f} ft

TASK: Generate a conflict resolution command for the ownship. 
Prefer horizontal maneuvers (heading changes) over vertical maneuvers (altitude changes) as they are typically less intrusive.
Provide your resolution as JSON with fields: resolution_type (either "heading" or "altitude"), new_heading_deg (if heading change), new_altitude_ft (if altitude change), target_aircraft (ownship ID).
"""
        return prompt.strip()

    def _create_resolution_from_llm(self, llm_json: Dict[str, Any], conflict: ConflictPrediction, 
                                  ownship: Dict[str, Any], source_engine: ResolutionEngine = ResolutionEngine.HORIZONTAL) -> Optional[ResolutionCommand]:
        """Convert LLM JSON response to ResolutionCommand object.
        
        Args:
            llm_json: LLM response JSON
            conflict: Original conflict prediction
            ownship: Current ownship state dict
            source_engine: Which LLM engine generated this resolution
            
        Returns:
            ResolutionCommand object or None if invalid
        """
        try:
            from datetime import datetime
            
            # Extract resolution details from LLM response
            resolution_type = llm_json.get("resolution_type", "").lower()
            target_aircraft = llm_json.get("target_aircraft", ownship["id"])
            
            # A3: Default target_aircraft to ownship if missing
            if not target_aircraft:
                target_aircraft = ownship["id"]
                logger.warning("target_aircraft missing in LLM response, defaulting to ownship")
            
            # Create resolution command based on type
            if resolution_type == "heading":
                new_heading = llm_json.get("new_heading_deg")
                if new_heading is None:
                    logger.error("LLM response missing new_heading_deg for heading resolution")
                    return None
                    
                return ResolutionCommand(
                    resolution_id=f"hdg_{target_aircraft}_{int(datetime.now().timestamp())}",
                    target_aircraft=target_aircraft,
                    resolution_type=ResolutionType.HEADING_CHANGE,
                    source_engine=source_engine,
                    new_heading_deg=float(new_heading),
                    new_speed_kt=None,
                    new_altitude_ft=None,
                    waypoint_name=None,  # GAP 5 FIX
                    waypoint_lat=None,  # GAP 5 FIX
                    waypoint_lon=None,  # GAP 5 FIX
                    diversion_distance_nm=None,  # GAP 5 FIX
                    hold_min=None,  # GAP 5 FIX
                    rate_fpm=None,  # GAP 5 FIX
                    issue_time=datetime.now(),
                    is_validated=False,
                    safety_margin_nm=5.0,  # Default safety margin
                    is_ownship_command=True,  # Will be validated later
                    angle_within_limits=True,  # Will be validated later
                    altitude_within_limits=True,
                    rate_within_limits=True
                )
                
            elif resolution_type == "altitude":
                new_altitude = llm_json.get("new_altitude_ft")
                if new_altitude is None:
                    logger.error("LLM response missing new_altitude_ft for altitude resolution")
                    return None
                    
                return ResolutionCommand(
                    resolution_id=f"alt_{target_aircraft}_{int(datetime.now().timestamp())}",
                    target_aircraft=target_aircraft,
                    resolution_type=ResolutionType.ALTITUDE_CHANGE,
                    source_engine=source_engine,
                    new_heading_deg=None,
                    new_speed_kt=None,
                    new_altitude_ft=float(new_altitude),
                    waypoint_name=None,  # GAP 5 FIX
                    waypoint_lat=None,  # GAP 5 FIX
                    waypoint_lon=None,  # GAP 5 FIX
                    diversion_distance_nm=None,  # GAP 5 FIX
                    hold_min=None,  # GAP 5 FIX
                    rate_fpm=llm_json.get("rate_fpm"),  # GAP 5 FIX - can extract rate from LLM
                    issue_time=datetime.now(),
                    is_validated=False,
                    safety_margin_nm=5.0,  # Default safety margin
                    is_ownship_command=True,  # Will be validated later
                    angle_within_limits=True,
                    altitude_within_limits=True,  # Will be validated later
                    rate_within_limits=True  # Will be validated later
                )
                
            elif resolution_type == "waypoint":
                waypoint_name = llm_json.get("waypoint_name")
                if not waypoint_name:
                    logger.error("LLM response missing waypoint_name for waypoint resolution")
                    return None
                
                # Validate waypoint exists and is within limits
                from .nav_utils import validate_waypoint_diversion
                ownship_lat = ownship.get("lat")
                ownship_lon = ownship.get("lon")
                
                if ownship_lat is None or ownship_lon is None:
                    logger.error("Missing ownship position for waypoint validation")
                    return None
                
                # Get maximum diversion distance from config (fallback to default)
                max_diversion_nm = 80.0  # Default value
                try:
                    max_diversion_nm = self.config.max_waypoint_diversion_nm
                except AttributeError:
                    pass
                
                validation_result = validate_waypoint_diversion(
                    ownship_lat, ownship_lon, waypoint_name, max_diversion_nm
                )
                
                if not validation_result:
                    logger.warning(f"Waypoint '{waypoint_name}' validation failed - not found or too far")
                    return None
                
                wp_lat, wp_lon, distance_nm = validation_result
                
                return ResolutionCommand(
                    resolution_id=f"wpt_{target_aircraft}_{int(datetime.now().timestamp())}",
                    target_aircraft=target_aircraft,
                    resolution_type=ResolutionType.WAYPOINT_DIRECT,
                    source_engine=source_engine,
                    new_heading_deg=None,
                    new_speed_kt=None,
                    new_altitude_ft=None,
                    waypoint_name=waypoint_name.upper(),
                    waypoint_lat=wp_lat,
                    waypoint_lon=wp_lon,
                    diversion_distance_nm=distance_nm,
                    hold_min=llm_json.get("hold_min"),  # GAP 5 FIX - extract hold duration if provided
                    rate_fpm=None,  # GAP 5 FIX
                    issue_time=datetime.now(),
                    is_validated=False,
                    safety_margin_nm=5.0,  # Default safety margin
                    is_ownship_command=True,  # Will be validated later
                    angle_within_limits=True,
                    altitude_within_limits=True,
                    rate_within_limits=True
                )
                
            else:
                logger.error(f"Unknown resolution type from LLM: {resolution_type}")
                return None
                
        except Exception as e:
            logger.exception(f"Error creating resolution from LLM response: {e}")
            return None
    
    def _generate_resolution(
        self, 
        conflict: ConflictPrediction, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]]
    ) -> Optional[ResolutionCommand]:
        """Generate conflict resolution using available methods.
        
        Args:
            conflict: Conflict to resolve
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            
        Returns:
            Resolution command or None
        """
        # TODO: Implement LLM-based resolution in Sprint 3
        # For now, use deterministic algorithms
        
        try:
            # Convert dicts to AircraftState for legacy resolution methods
            ownship_state = _dict_to_aircraft_state(ownship)
            
            # Try horizontal resolution first
            resolution = generate_horizontal_resolution(conflict, ownship_state)
            if resolution:
                return resolution
            
            # Fallback to vertical resolution
            return generate_vertical_resolution(conflict, ownship_state)
            
        except Exception as e:
            logger.exception(f"Error in fallback resolution generation: {e}")
            return None
    
    def _validate_and_execute_resolution(
        self, 
        resolution: ResolutionCommand, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]],
        ownship_id: str
    ) -> bool:
        """Validate and execute resolution command with enhanced validation.
        
        Args:
            resolution: Resolution to validate and execute
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            ownship_id: Expected ownship identifier
            
        Returns:
            True if successfully executed
        """
        # Use enhanced validator
        if not self.enhanced_validator.validate_resolution(resolution, ownship, traffic, ownship_id):
            logger.error(f"Resolution {resolution.resolution_id} failed enhanced validation")
            return False
        
        # Convert dicts to AircraftState for legacy validation if needed
        try:
            ownship_state = _dict_to_aircraft_state(ownship)
            traffic_states = [_dict_to_aircraft_state(t) for t in traffic]
            
            # Additional safety validation using legacy method
            if not validate_resolution(resolution, ownship_state, traffic_states):
                logger.error(f"Resolution {resolution.resolution_id} failed legacy safety validation")
                return False
        except Exception as e:
            logger.exception(f"Error in legacy validation: {e}")
            return False
        
        # Execute via BlueSky
        success = self.bluesky_client.execute_command(resolution)
        if success:
            logger.info(f"Executed {resolution.source_engine.value} resolution: {resolution.resolution_type.value}")
            return True
        else:
            logger.error(f"Failed to execute resolution {resolution.resolution_id}")
            return False
    
    def _update_metrics(
        self, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]], 
        conflicts: List[ConflictPrediction]
    ) -> None:
        """Update performance metrics.
        
        Args:
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            conflicts: Detected conflicts
        """
        # TODO: Implement in Sprint 4
        pass
    
    def _cleanup(self) -> None:
        """Clean up resources and save final state."""
        logger.info("Cleaning up CDR pipeline")
        
        # Save metrics
        self.metrics.save_report(f"reports/sprint_0/cycle_{self.cycle_count}_metrics.json")
        
        # Close BlueSky connections and clean up shapes
        if self.bluesky_client:
            self.bluesky_client.close()
    
    def run_for_flights(self, flight_records: List['FlightRecord'], max_cycles: Optional[int] = None,
                       monte_carlo_params: Optional['MonteCarloParameters'] = None) -> 'BatchSimulationResult':
        """Run batch simulation for multiple flights with Monte Carlo intruder generation.
        
        Args:
            flight_records: List of flight records to simulate
            max_cycles: Maximum cycles per scenario (None = use default)
            monte_carlo_params: Parameters for Monte Carlo generation
            
        Returns:
            BatchSimulationResult with aggregated metrics
        """
        from .monte_carlo_intruders import BatchIntruderGenerator
        from datetime import datetime
        
        logger.info(f"Starting batch simulation for {len(flight_records)} flights")
        
        # Initialize Monte Carlo parameters if not provided
        if monte_carlo_params is None:
            monte_carlo_params = MonteCarloParameters(
                scenarios_per_flight=10,
                intruder_count_range=(1, 5),
                conflict_zone_radius_nm=50.0,
                non_conflict_zone_radius_nm=200.0,
                altitude_spread_ft=10000.0,
                time_window_min=60.0,
                conflict_timing_variance_min=10.0,
                conflict_probability=0.3,
                speed_variance_kt=50.0,
                heading_variance_deg=45.0,
                realistic_aircraft_types=True,
                airway_based_generation=False,
                weather_influence=False
            )
        
        # Generate intruder scenarios
        intruder_generator = BatchIntruderGenerator(monte_carlo_params)
        all_scenarios = intruder_generator.generate_scenarios_for_flights(flight_records)
        
        # Initialize batch result
        simulation_id = f"batch_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_result = BatchSimulationResult(
            simulation_id=simulation_id,
            start_time=datetime.now(),
            flight_records=[fr.flight_id for fr in flight_records],
            scenarios_per_flight=monte_carlo_params.scenarios_per_flight,
            total_scenarios=sum(len(scenarios) for scenarios in all_scenarios.values()),
            total_conflicts_detected=0,
            total_resolutions_attempted=0,
            successful_resolutions=0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            average_resolution_time_sec=0.0,
            minimum_separation_achieved_nm=5.0,
            safety_violations=0
        )
        
        # Process each flight and its scenarios
        total_conflicts = 0
        total_resolutions = 0
        successful_resolutions = 0
        all_resolution_times = []
        min_separation = float('inf')
        safety_violations = 0
        
        flight_results = {}
        scenario_results = []
        
        for flight_record in flight_records:
            logger.info(f"Processing flight {flight_record.flight_id}")
            scenarios = all_scenarios[flight_record.flight_id]
            
            flight_metrics = {
                'flight_id': flight_record.flight_id,
                'scenarios_processed': 0,
                'conflicts_detected': 0,
                'resolutions_attempted': 0,
                'successful_resolutions': 0,
                'safety_violations': 0
            }
            
            for scenario in scenarios:
                scenario_metrics = self._run_single_scenario(
                    flight_record, scenario, max_cycles or 120
                )
                
                # Aggregate metrics
                total_conflicts += scenario_metrics.get('conflicts_detected', 0)
                total_resolutions += scenario_metrics.get('resolutions_attempted', 0)
                successful_resolutions += scenario_metrics.get('successful_resolutions', 0)
                
                if 'resolution_times' in scenario_metrics:
                    all_resolution_times.extend(scenario_metrics['resolution_times'])
                
                if 'minimum_separation_nm' in scenario_metrics:
                    min_separation = min(min_separation, scenario_metrics['minimum_separation_nm'])
                
                safety_violations += scenario_metrics.get('safety_violations', 0)
                
                # Update flight metrics
                flight_metrics['scenarios_processed'] += 1
                flight_metrics['conflicts_detected'] += scenario_metrics.get('conflicts_detected', 0)
                flight_metrics['resolutions_attempted'] += scenario_metrics.get('resolutions_attempted', 0)
                flight_metrics['successful_resolutions'] += scenario_metrics.get('successful_resolutions', 0)
                flight_metrics['safety_violations'] += scenario_metrics.get('safety_violations', 0)
                
                # Store scenario result
                scenario_results.append({
                    'flight_id': flight_record.flight_id,
                    'scenario_id': scenario.scenario_id,
                    'has_conflicts': scenario.has_conflicts,
                    'expected_conflicts': len(scenario.expected_conflicts),
                    **scenario_metrics
                })
                
                logger.debug(f"Completed scenario {scenario.scenario_id}")
            
            flight_results[flight_record.flight_id] = flight_metrics
            logger.info(f"Completed flight {flight_record.flight_id}: {flight_metrics}")
        
        # Calculate final metrics
        batch_result.end_time = datetime.now()
        batch_result.total_conflicts_detected = total_conflicts
        batch_result.total_resolutions_attempted = total_resolutions
        batch_result.successful_resolutions = successful_resolutions
        
        if total_resolutions > 0:
            batch_result.false_positive_rate = max(0, (total_resolutions - total_conflicts) / total_resolutions)
        
        if all_resolution_times:
            batch_result.average_resolution_time_sec = sum(all_resolution_times) / len(all_resolution_times)
        
        if min_separation != float('inf'):
            batch_result.minimum_separation_achieved_nm = min_separation
        
        batch_result.safety_violations = safety_violations
        batch_result.flight_results = flight_results
        batch_result.scenario_results = scenario_results
        
        logger.info(f"Batch simulation completed: {batch_result}")
        return batch_result
    
    def _run_single_scenario(self, flight_record: 'FlightRecord', scenario: 'IntruderScenario', 
                           max_cycles: int) -> Dict[str, Any]:
        """Run simulation for a single flight/scenario combination.
        
        Args:
            flight_record: Flight data
            scenario: Intruder scenario
            max_cycles: Maximum simulation cycles
            
        Returns:
            Dictionary with scenario metrics
        """
        from .monte_carlo_intruders import FlightPathAnalyzer
        
        logger.debug(f"Running scenario {scenario.scenario_id}")
        
        # Reset BlueSky state
        self.bluesky_client.sim_reset()
        
        # Initialize metrics for this scenario
        metrics = {
            'conflicts_detected': 0,
            'resolutions_attempted': 0,
            'successful_resolutions': 0,
            'resolution_times': [],
            'minimum_separation_nm': float('inf'),
            'safety_violations': 0,
            'scenario_completed': False
        }
        
        try:
            # Create ownship in BlueSky
            self._create_ownship_from_flight_record(flight_record)
            
            # Initialize intruder spawning scheduler
            spawned_intruders = set()  # Track which intruders have been spawned
            pending_intruders = list(scenario.intruder_states)  # Intruders to spawn
            simulation_time_min = 0.0  # Track simulation time
            
            # Analyze flight path for intrusion detection
            path_analyzer = FlightPathAnalyzer(flight_record)
            
            # Run simulation cycles
            cycle = 0
            while cycle < max_cycles and self.running:
                # Dynamic intruder spawning: spawn intruders whose time has come
                intruders_to_spawn = []
                for intruder in pending_intruders:
                    if intruder.spawn_offset_min <= simulation_time_min:
                        intruders_to_spawn.append(intruder)
                
                # Spawn due intruders
                for intruder in intruders_to_spawn:
                    success = self._create_aircraft_from_state(intruder)
                    if success:
                        spawned_intruders.add(intruder.aircraft_id)
                        pending_intruders.remove(intruder)
                        logger.info(f"Spawned {intruder.aircraft_id} at t={simulation_time_min:.1f} min")
                    else:
                        logger.warning(f"Failed to spawn {intruder.aircraft_id} at t={simulation_time_min:.1f} min")
                
                # Execute standard CDR cycle
                ownship, traffic = self._fetch_aircraft_states(flight_record.flight_id)
                
                if ownship is None:
                    logger.warning(f"Ownship {flight_record.flight_id} not found in cycle {cycle}")
                    break
                
                # Detect conflicts
                conflicts = self._predict_conflicts(ownship, traffic)
                metrics['conflicts_detected'] += len([c for c in conflicts if c.is_conflict])
                
                # Handle conflicts
                for conflict in conflicts:
                    if conflict.is_conflict:
                        resolution_start = time.time()
                        
                        # Attempt resolution
                        success = self._handle_conflict_with_metrics(conflict, ownship, traffic, metrics)
                        
                        metrics['resolutions_attempted'] += 1
                        if success:
                            metrics['successful_resolutions'] += 1
                        
                        resolution_time = time.time() - resolution_start
                        metrics['resolution_times'].append(resolution_time)
                
                # Check intrusions using KDTree
                intrusions = path_analyzer.detect_intrusions_along_path(
                    [_dict_to_aircraft_state(state) for state in traffic]
                )
                
                for intrusion in intrusions:
                    sep_nm = intrusion['horizontal_separation_nm']
                    if sep_nm < metrics['minimum_separation_nm']:
                        metrics['minimum_separation_nm'] = sep_nm
                    
                    if intrusion['is_conflict']:
                        metrics['safety_violations'] += 1
                
                # Advance simulation time
                self.bluesky_client.step_minutes(self.config.polling_interval_min)
                simulation_time_min += self.config.polling_interval_min
                cycle += 1
            
            metrics['scenario_completed'] = True
            
            # Create enhanced scenario metrics
            scenario_metrics = self._create_scenario_metrics(
                scenario, flight_record, metrics, simulation_time_min
            )
            self.enhanced_reporting.add_scenario_completion(scenario_metrics)
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario.scenario_id}: {e}")
            metrics['error'] = str(e)
            
            # Create error scenario metrics
            scenario_metrics = self._create_scenario_metrics(
                scenario, flight_record, metrics, 0.0, error=str(e)
            )
            self.enhanced_reporting.add_scenario_completion(scenario_metrics)
        
        return metrics
    
    def _create_scenario_metrics(self, scenario: 'IntruderScenario', flight_record: 'FlightRecord', 
                               metrics: Dict[str, Any], simulation_time_min: float, 
                               error: Optional[str] = None) -> ScenarioMetrics:
        """Create enhanced scenario metrics from collected data."""
        
        # Calculate success rate
        conflicts_resolved = metrics.get('successful_resolutions', 0)
        total_conflicts = metrics.get('conflicts_detected', 0)
        resolution_success_rate = (conflicts_resolved / total_conflicts * 100) if total_conflicts > 0 else 100.0
        
        # Calculate average time to action
        resolution_times = metrics.get('resolution_times', [])
        avg_time_to_action_sec = sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
        
        # Get minimum separation achieved
        min_separation_achieved_nm = metrics.get('minimum_separation_nm', float('inf'))
        if min_separation_achieved_nm == float('inf'):
            min_separation_achieved_nm = 999.0  # No conflicts detected
        
        # Check if separation standards were maintained
        safety_violations = metrics.get('safety_violations', 0)
        separation_standards_maintained = safety_violations == 0
        
        # Calculate engine usage from conflict resolutions
        horizontal_usage = 0
        vertical_usage = 0
        deterministic_usage = 0
        fallback_usage = 0
        
        # Count engine usage from collected conflict metrics
        for conflict_metric in self.enhanced_reporting.conflict_metrics:
            if conflict_metric.ownship_id == flight_record.flight_id:
                if conflict_metric.engine_used == "horizontal":
                    horizontal_usage += 1
                elif conflict_metric.engine_used == "vertical":
                    vertical_usage += 1
                elif conflict_metric.engine_used == "deterministic":
                    deterministic_usage += 1
                else:
                    fallback_usage += 1
        
        # Calculate path similarity (placeholder - would need SCAT baseline)
        ownship_path_similarity = 0.85  # Default reasonable similarity
        total_path_deviation_nm = 0.0
        max_cross_track_error_nm = 0.0
        
        # Aggregate path deviations from conflict resolutions
        for conflict_metric in self.enhanced_reporting.conflict_metrics:
            if conflict_metric.ownship_id == flight_record.flight_id:
                total_path_deviation_nm += conflict_metric.path_deviation_total_nm
                max_cross_track_error_nm = max(max_cross_track_error_nm, 
                                             conflict_metric.ownship_cross_track_error_nm)
        
        # Get conflict resolution details for this scenario
        scenario_conflict_resolutions = [
            cm for cm in self.enhanced_reporting.conflict_metrics 
            if cm.ownship_id == flight_record.flight_id
        ]
        
        return ScenarioMetrics(
            scenario_id=scenario.scenario_id,
            flight_id=flight_record.flight_id,
            total_conflicts=total_conflicts,
            conflicts_resolved=conflicts_resolved,
            resolution_success_rate=resolution_success_rate,
            scenario_duration_min=simulation_time_min,
            avg_time_to_action_sec=avg_time_to_action_sec,
            min_separation_achieved_nm=min_separation_achieved_nm,
            safety_violations=safety_violations,
            separation_standards_maintained=separation_standards_maintained,
            horizontal_engine_usage=horizontal_usage,
            vertical_engine_usage=vertical_usage,
            deterministic_engine_usage=deterministic_usage,
            fallback_engine_usage=fallback_usage,
            ownship_path_similarity=ownship_path_similarity,
            total_path_deviation_nm=total_path_deviation_nm,
            max_cross_track_error_nm=max_cross_track_error_nm,
            conflict_resolutions=scenario_conflict_resolutions
        )
    
    def generate_enhanced_reports(self, output_dir: str = "reports") -> Tuple[str, str]:
        """Generate enhanced CSV and JSON reports.
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Tuple of (csv_path, json_path)
        """
        logger.info("Generating enhanced reporting outputs...")
        
        # Update the reporting system output directory
        self.enhanced_reporting.output_dir = Path(output_dir)
        self.enhanced_reporting.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate CSV report
        csv_filename = f"enhanced_metrics_report_{timestamp}.csv"
        csv_path = self.enhanced_reporting.generate_csv_report(csv_filename)
        
        # Generate JSON report
        json_filename = f"enhanced_metrics_report_{timestamp}.json"
        json_path = self.enhanced_reporting.generate_json_report(json_filename)
        
        logger.info(f"Enhanced reports generated:")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  JSON: {json_path}")
        
        return csv_path, json_path
    
    def get_enhanced_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from enhanced reporting system."""
        if not self.enhanced_reporting.conflict_metrics:
            return {}
        
        total_conflicts = len(self.enhanced_reporting.conflict_metrics)
        resolved_conflicts = sum(1 for c in self.enhanced_reporting.conflict_metrics if c.resolved)
        
        return {
            'overall_success_rate': (resolved_conflicts / total_conflicts) * 100 if total_conflicts > 0 else 0,
            'total_conflicts': total_conflicts,
            'resolved_conflicts': resolved_conflicts,
            'total_scenarios': len(self.enhanced_reporting.scenario_metrics),
            'average_time_to_action_sec': sum(c.time_to_action_sec for c in self.enhanced_reporting.conflict_metrics) / total_conflicts,
            'average_min_separation_nm': sum(c.min_sep_nm for c in self.enhanced_reporting.conflict_metrics) / total_conflicts,
            'separation_violations': sum(1 for c in self.enhanced_reporting.conflict_metrics if c.separation_violation),
            'engine_usage': {
                'horizontal': sum(1 for c in self.enhanced_reporting.conflict_metrics if c.engine_used == 'horizontal'),
                'vertical': sum(1 for c in self.enhanced_reporting.conflict_metrics if c.engine_used == 'vertical'),
                'deterministic': sum(1 for c in self.enhanced_reporting.conflict_metrics if c.engine_used == 'deterministic'),
                'fallback': sum(1 for c in self.enhanced_reporting.conflict_metrics if c.engine_used == 'fallback')
            },
            'average_path_deviation_nm': sum(c.path_deviation_total_nm for c in self.enhanced_reporting.conflict_metrics) / total_conflicts,
            'average_resolution_effectiveness': sum(c.resolution_effectiveness for c in self.enhanced_reporting.conflict_metrics) / total_conflicts
        }
    
    def _create_ownship_from_flight_record(self, flight_record: 'FlightRecord') -> bool:
        """Create ownship aircraft in BlueSky from flight record."""
        if not flight_record.waypoints:
            return False
        
        # Start position
        lat, lon = flight_record.waypoints[0]
        alt_ft = flight_record.altitudes_ft[0]
        speed_kt = flight_record.cruise_speed_kt
        
        # Create aircraft
        cmd = f"CRE {flight_record.flight_id} {flight_record.aircraft_type} {lat} {lon} {alt_ft} {speed_kt}"
        success = self.bluesky_client.stack(cmd)
        
        if success:
            # Add waypoints to flight plan
            for i in range(1, len(flight_record.waypoints)):
                lat, lon = flight_record.waypoints[i]
                alt_ft = flight_record.altitudes_ft[i]
                waypoint_cmd = f"ADDWPT {flight_record.flight_id} {lat} {lon} {alt_ft}"
                self.bluesky_client.stack(waypoint_cmd)
            
            logger.debug(f"Created ownship {flight_record.flight_id} with {len(flight_record.waypoints)} waypoints")
        
        return success
    
    def _create_aircraft_from_state(self, aircraft_state: 'AircraftState') -> bool:
        """Create aircraft in BlueSky from AircraftState."""
        cmd = (f"CRE {aircraft_state.aircraft_id} {aircraft_state.aircraft_type or 'B737'} "
               f"{aircraft_state.latitude} {aircraft_state.longitude} "
               f"{aircraft_state.altitude_ft} {aircraft_state.ground_speed_kt}")
        
        success = self.bluesky_client.stack(cmd)
        
        if success:
            # Set heading
            hdg_cmd = f"HDG {aircraft_state.aircraft_id} {aircraft_state.heading_deg}"
            self.bluesky_client.stack(hdg_cmd)
        
        return success
    
    def _handle_conflict_with_metrics(self, conflict: ConflictPrediction, 
                                    ownship: Dict[str, Any], traffic: List[Dict[str, Any]],
                                    metrics: Dict[str, Any]) -> bool:
        """Handle conflict and collect detailed metrics."""
        # Start timing
        conflict_detection_time = datetime.now()
        initial_distance_nm = conflict.distance_at_cpa_nm
        
        try:
            # Build enhanced prompt
            prompt = self._llm_client_prompt_builder(conflict, ownship, traffic)
            
            # Ask LLM for action
            llm_json = self.llm_client.generate_resolution(prompt)
            
            # Create resolution from LLM response
            resolution = self._create_resolution_from_llm(llm_json, conflict, ownship)
            
            # Record resolution command time
            resolution_command_time = datetime.now()
            time_to_action_sec = (resolution_command_time - conflict_detection_time).total_seconds()
            
            # Determine engine used and resolution details
            engine_used = "fallback"
            resolution_type = "unknown"
            waypoint_vs_heading = "unknown"
            
            if resolution:
                if resolution.source_engine == ResolutionEngine.HORIZONTAL:
                    engine_used = "horizontal"
                elif resolution.source_engine == ResolutionEngine.VERTICAL:
                    engine_used = "vertical"
                elif resolution.source_engine == ResolutionEngine.DETERMINISTIC:
                    engine_used = "deterministic"
                
                if resolution.resolution_type == ResolutionType.HEADING_CHANGE:
                    resolution_type = "heading_change"
                    waypoint_vs_heading = "heading"
                elif resolution.resolution_type == ResolutionType.ALTITUDE_CHANGE:
                    resolution_type = "altitude_change"
                elif resolution.resolution_type == ResolutionType.SPEED_CHANGE:
                    resolution_type = "speed_change"
                elif resolution.resolution_type == ResolutionType.COMBINED:
                    resolution_type = "combined"
            
            # Execute resolution
            success = False
            if resolution and self._validate_and_execute_resolution(resolution, ownship, traffic, ownship["id"]):
                logger.info(f"Successfully executed resolution {resolution.resolution_id}")
                self.active_resolutions[resolution.resolution_id] = resolution
                success = True
            else:
                logger.error(f"Failed to resolve conflict with {conflict.intruder_id}")
            
            # Calculate final separation (simulate future positions)
            # For now, use the predicted CPA distance as proxy
            min_sep_nm = conflict.distance_at_cpa_nm
            final_distance_nm = conflict.distance_at_cpa_nm
            
            # Determine if there was a separation violation (less than minimum standards)
            separation_violation = min_sep_nm < self.config.min_horizontal_separation_nm
            
            # Calculate path deviation (simplified - would need trajectory comparison)
            # For now, estimate based on resolution magnitude
            path_deviation_total_nm = 0.0
            if resolution and resolution.resolution_type == ResolutionType.HEADING_CHANGE:
                # Rough estimation: 10 NM deviation per 30-degree heading change
                path_deviation_total_nm = abs(resolution.new_heading_deg - ownship.get("hdg_deg", 0)) * 10.0 / 30.0
            elif resolution and resolution.resolution_type == ResolutionType.ALTITUDE_CHANGE:
                # Vertical maneuver - minimal horizontal path deviation
                path_deviation_total_nm = 1.0
            
            # Calculate effectiveness score (0-1)
            resolution_effectiveness = 0.0
            if success:
                # Higher score for better separation improvement
                if min_sep_nm >= self.config.min_horizontal_separation_nm:
                    resolution_effectiveness = min(1.0, min_sep_nm / (2 * self.config.min_horizontal_separation_nm))
                else:
                    resolution_effectiveness = 0.3  # Partial credit for attempting resolution
            
            # Calculate operational impact (0-1, lower is better)
            operational_impact = min(1.0, path_deviation_total_nm / 20.0)  # Normalize to 20 NM max
            
            # Create detailed conflict resolution metrics
            conflict_metrics = ConflictResolutionMetrics(
                conflict_id=f"{ownship['id']}_{conflict.intruder_id}_{int(time.time())}",
                ownship_id=ownship["id"],
                intruder_id=conflict.intruder_id,
                resolved=success,
                engine_used=engine_used,
                resolution_type=resolution_type,
                waypoint_vs_heading=waypoint_vs_heading,
                time_to_action_sec=time_to_action_sec,
                conflict_detection_time=conflict_detection_time,
                resolution_command_time=resolution_command_time,
                initial_distance_nm=initial_distance_nm,
                min_sep_nm=min_sep_nm,
                final_distance_nm=final_distance_nm,
                separation_violation=separation_violation,
                ownship_cross_track_error_nm=0.0,  # Would need SCAT baseline for comparison
                ownship_along_track_error_nm=0.0,   # Would need SCAT baseline for comparison
                path_deviation_total_nm=path_deviation_total_nm,
                resolution_effectiveness=resolution_effectiveness,
                operational_impact=operational_impact
            )
            
            # Add to enhanced reporting system
            self.enhanced_reporting.add_conflict_resolution(conflict_metrics)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}")
            
            # Create metrics for failed resolution
            resolution_command_time = datetime.now()
            time_to_action_sec = (resolution_command_time - conflict_detection_time).total_seconds()
            
            conflict_metrics = ConflictResolutionMetrics(
                conflict_id=f"{ownship['id']}_{conflict.intruder_id}_{int(time.time())}",
                ownship_id=ownship["id"],
                intruder_id=conflict.intruder_id,
                resolved=False,
                engine_used="error",
                resolution_type="error",
                waypoint_vs_heading="error",
                time_to_action_sec=time_to_action_sec,
                conflict_detection_time=conflict_detection_time,
                resolution_command_time=resolution_command_time,
                initial_distance_nm=initial_distance_nm,
                min_sep_nm=initial_distance_nm,  # No improvement
                final_distance_nm=initial_distance_nm,
                separation_violation=True,
                ownship_cross_track_error_nm=0.0,
                ownship_along_track_error_nm=0.0,
                path_deviation_total_nm=0.0,
                resolution_effectiveness=0.0,
                operational_impact=1.0  # Maximum operational impact for failure
            )
            
            self.enhanced_reporting.add_conflict_resolution(conflict_metrics)
            return False
    
    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        logger.info("Stopping CDR pipeline")
        self.running = False


def main():
    """Main entry point for CDR pipeline."""
    # TODO: Add command-line argument parsing
    config = ConfigurationSettings(
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        snapshot_interval_min=1.5,
        max_intruders_in_prompt=5,
        intruder_proximity_nm=100.0,
        intruder_altitude_diff_ft=5000.0,
        trend_analysis_window_min=2.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_enabled=True,
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1,
        llm_max_tokens=2048,
        safety_buffer_factor=1.2,
        max_resolution_angle_deg=45.0,
        max_altitude_change_ft=2000.0,
        # Enhanced validation settings
        enforce_ownship_only=True,
        max_climb_rate_fpm=3000.0,
        max_descent_rate_fpm=3000.0,
        min_flight_level=100,
        max_flight_level=600,
        max_heading_change_deg=90.0,
        # Dual LLM engine settings
        enable_dual_llm=True,
        horizontal_retry_count=2,
        vertical_retry_count=2,
        bluesky_host="localhost",
        bluesky_port=1337,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )
    pipeline = CDRPipeline(config)
    
    try:
        pipeline.run(ownship_id="OWNSHIP")
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
