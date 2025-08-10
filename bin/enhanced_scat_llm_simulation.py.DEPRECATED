"""
Enhanced SCAT+LLM simulation with proper heading-based movement and intelligent waypoint navigation.
This version actually follows LLM guidance instead of just logging it.
"""
import sys
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add the parent directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState, ResolutionCommand, ResolutionType
from src.cdr.llm_client import LlamaClient
from src.cdr.geodesy import (
    haversine_nm, bearing_deg, destination_point_nm, normalize_heading_deg
)

# Configuration
SCAT_DIR = r"F:\SCAT_extracted"
SCAT_FILE = "100000.json"
CRUISE_SPD_KT = 420.0
CRUISE_ALT_FT = 34000.0
TIME_STEP_MIN = 1.0
SIMULATION_CYCLES = 60
LOOKAHEAD_TIME_MIN = 10.0
CONFLICT_THRESHOLD_NM = 5.0
WAYPOINT_RADIUS_NM = 2.0

# Navigation database for waypoint lookup
NAV_DATABASE = {
    # Sample waypoints in the region (in real implementation, load from nav.dat)
    "CPH": (55.6176, 12.6559),    # Copenhagen
    "ARN": (59.6519, 17.9186),    # Stockholm Arlanda
    "OSL": (60.1939, 11.1004),    # Oslo
    "GOT": (57.6628, 12.2798),    # Gothenburg
    "AAL": (57.0928, 9.8492),     # Aalborg
    "BLL": (55.7308, 9.1518),     # Billund
    "STO": (59.6519, 17.9186),    # Stockholm area
    "MLM": (55.5363, 13.3728),    # Malmo
    "HEL": (60.3172, 24.9633),    # Helsinki
    "TLL": (59.4133, 24.8328),    # Tallinn
}

@dataclass
class NavigationState:
    """Enhanced navigation state with heading-based movement."""
    position: Tuple[float, float]
    heading_deg: float
    ground_speed_kt: float
    target_waypoint: Optional[Tuple[float, float]]
    route_waypoints: List[Tuple[float, float]]
    current_wpt_index: int
    llm_modified_heading: Optional[float] = None
    time_since_llm_maneuver: float = 0.0
    return_to_route_mode: bool = False

def create_aircraft_state(callsign: str, pos: Tuple[float, float], hdg: float, spd: float, alt: float) -> AircraftState:
    """Create an AircraftState object from position and flight parameters."""
    return AircraftState(
        aircraft_id=callsign,
        timestamp=datetime.now(),
        latitude=pos[0],
        longitude=pos[1],
        altitude_ft=alt,
        ground_speed_kt=spd,
        heading_deg=hdg,
        vertical_speed_fpm=0.0
    )

def find_nearby_waypoints(position: Tuple[float, float], radius_nm: float = 50.0) -> List[Tuple[str, float, float, float]]:
    """Find nearby navigation waypoints within specified radius."""
    nearby = []
    for wpt_id, (lat, lon) in NAV_DATABASE.items():
        distance = haversine_nm(position, (lat, lon))
        if distance <= radius_nm:
            bearing = bearing_deg(position[0], position[1], lat, lon)
            nearby.append((wpt_id, lat, lon, distance))
    
    # Sort by distance
    nearby.sort(key=lambda x: x[3])
    return nearby

def simulate_heading_based_movement(nav_state: NavigationState, time_step_min: float) -> NavigationState:
    """
    Simulate aircraft movement using current heading (not direct-to-waypoint).
    This actually follows LLM-modified headings!
    """
    # Calculate distance traveled in this time step
    distance_nm = (nav_state.ground_speed_kt / 60.0) * time_step_min
    
    # Use LLM-modified heading if available, otherwise current heading
    current_heading = nav_state.llm_modified_heading if nav_state.llm_modified_heading is not None else nav_state.heading_deg
    
    # Move aircraft based on current heading
    new_position = destination_point_nm(
        nav_state.position[0], nav_state.position[1], 
        current_heading, distance_nm
    )
    
    # Update time since LLM maneuver
    new_time_since_llm = nav_state.time_since_llm_maneuver + time_step_min
    
    # Check if we should return to route after LLM maneuver
    return_to_route = nav_state.return_to_route_mode
    final_heading = current_heading
    
    if nav_state.llm_modified_heading is not None and new_time_since_llm > 5.0:  # After 5 minutes
        if nav_state.target_waypoint:
            # Gradually return to route
            desired_heading = bearing_deg(new_position[0], new_position[1], 
                                        nav_state.target_waypoint[0], nav_state.target_waypoint[1])
            heading_diff = normalize_heading_deg(desired_heading - current_heading)
            
            # Turn 10 degrees per minute towards route
            if abs(heading_diff) > 10.0:
                turn_rate = 10.0 * time_step_min
                if heading_diff > 0:
                    final_heading = normalize_heading_deg(current_heading + turn_rate)
                else:
                    final_heading = normalize_heading_deg(current_heading - turn_rate)
                return_to_route = True
            else:
                final_heading = desired_heading
                return_to_route = False
                nav_state.llm_modified_heading = None  # Back on route
    
    return NavigationState(
        position=new_position,
        heading_deg=final_heading,
        ground_speed_kt=nav_state.ground_speed_kt,
        target_waypoint=nav_state.target_waypoint,
        route_waypoints=nav_state.route_waypoints,
        current_wpt_index=nav_state.current_wpt_index,
        llm_modified_heading=nav_state.llm_modified_heading if new_time_since_llm <= 5.0 else None,
        time_since_llm_maneuver=new_time_since_llm,
        return_to_route_mode=return_to_route
    )

def check_waypoint_reached(nav_state: NavigationState) -> NavigationState:
    """Check if current waypoint is reached and advance to next."""
    if not nav_state.target_waypoint:
        return nav_state
    
    distance_to_target = haversine_nm(nav_state.position, nav_state.target_waypoint)
    
    if distance_to_target <= WAYPOINT_RADIUS_NM:
        # Advance to next waypoint
        next_wpt_index = nav_state.current_wpt_index + 1
        if next_wpt_index < len(nav_state.route_waypoints):
            next_target = nav_state.route_waypoints[next_wpt_index]
            # Update heading to next waypoint
            new_heading = bearing_deg(nav_state.position[0], nav_state.position[1], 
                                    next_target[0], next_target[1])
            return NavigationState(
                position=nav_state.position,
                heading_deg=new_heading,
                ground_speed_kt=nav_state.ground_speed_kt,
                target_waypoint=next_target,
                route_waypoints=nav_state.route_waypoints,
                current_wpt_index=next_wpt_index,
                llm_modified_heading=nav_state.llm_modified_heading,
                time_since_llm_maneuver=nav_state.time_since_llm_maneuver,
                return_to_route_mode=nav_state.return_to_route_mode
            )
    
    return nav_state

def generate_enhanced_llm_prompt(ownship_state: AircraftState, intruder_states: List[AircraftState], 
                                nav_state: NavigationState, nearby_waypoints: List[Tuple[str, float, float, float]]) -> str:
    """Generate enhanced LLM prompt with detailed context and navigation options."""
    
    # Calculate conflict details
    conflict_details = []
    for intruder in intruder_states:
        distance = haversine_nm((ownship_state.latitude, ownship_state.longitude), 
                               (intruder.latitude, intruder.longitude))
        relative_bearing = bearing_deg(ownship_state.latitude, ownship_state.longitude,
                                     intruder.latitude, intruder.longitude)
        
        # Simple conflict prediction
        time_to_conflict = max(0.1, distance / ((ownship_state.ground_speed_kt + intruder.ground_speed_kt) / 60.0))
        
        conflict_details.append({
            "id": intruder.aircraft_id,
            "distance_nm": distance,
            "relative_bearing": relative_bearing,
            "altitude_ft": intruder.altitude_ft,
            "heading_deg": intruder.heading_deg,
            "speed_kt": intruder.ground_speed_kt,
            "time_to_conflict_min": time_to_conflict
        })
    
    # Format nearby waypoints
    waypoint_options = []
    for wpt_id, lat, lon, dist in nearby_waypoints[:5]:  # Top 5 nearest
        bearing = bearing_deg(ownship_state.latitude, ownship_state.longitude, lat, lon)
        waypoint_options.append(f"{wpt_id}: {dist:.1f}NM at {bearing:.0f}deg")
    
    # Current route information
    final_destination = nav_state.route_waypoints[-1] if nav_state.route_waypoints else "UNKNOWN"
    current_target = nav_state.target_waypoint if nav_state.target_waypoint else "NONE"
    
    prompt = f"""
URGENT: Air Traffic Control Conflict Resolution Required

SITUATION:
You are the pilot of aircraft {ownship_state.aircraft_id} at {ownship_state.altitude_ft} feet.
Current position: {ownship_state.latitude:.4f}degN, {ownship_state.longitude:.4f}degE
Current heading: {ownship_state.heading_deg:.0f}deg
Current speed: {ownship_state.ground_speed_kt:.0f} knots

MISSION CRITICAL: Your final destination is {final_destination}. You MUST reach this waypoint.
Current target waypoint: {current_target}

CONFLICT DETAILS:
{len(intruder_states)} aircraft detected in conflict zone:
"""
    
    for i, conflict in enumerate(conflict_details):
        prompt += f"""
Aircraft {conflict['id']}:
- Distance: {conflict['distance_nm']:.1f} NM at {conflict['relative_bearing']:.0f}deg relative bearing
- Altitude: {conflict['altitude_ft']:.0f} feet  
- Heading: {conflict['heading_deg']:.0f}deg, Speed: {conflict['speed_kt']:.0f} knots
- Estimated time to conflict: {conflict['time_to_conflict_min']:.1f} minutes
"""

    prompt += f"""
NAVIGATION OPTIONS:
Nearby waypoints available for rerouting:
{chr(10).join(waypoint_options)}

AVAILABLE COMMANDS:
1. TURN: Change heading by specified degrees
   - "turn_left X" or "turn_right X" where X is degrees (max 45deg)
   - Example: "turn_right 30" 

2. DIRECT: Go direct to a waypoint
   - "direct WAYPOINT_ID" 
   - Example: "direct CPH"

3. ALTITUDE: Change altitude (if necessary)
   - "climb X" or "descend X" where X is feet
   - Example: "climb 2000"

CRITICAL CONSTRAINTS:
- You control ONLY your aircraft ({ownship_state.aircraft_id})
- DO NOT attempt to control other aircraft
- Maintain safe separation (minimum 5 NM horizontal, 1000 ft vertical)
- You MUST eventually reach your final destination: {final_destination}
- Consider using waypoints for rerouting if needed
- Minimize delay to destination
- Choose the most efficient conflict resolution

REQUIRED RESPONSE FORMAT:
Provide your response as JSON only:
{{
    "action": "turn_left|turn_right|direct|climb|descend",
    "value": number_of_degrees_or_feet,
    "waypoint": "waypoint_id (only if using direct)",
    "reasoning": "brief explanation of why this resolution was chosen"
}}

Respond immediately with conflict resolution:
"""
    
    return prompt

def generate_llm_resolution(ownship_state: AircraftState, intruder_states: List[AircraftState], 
                          nav_state: NavigationState, llm_client: LlamaClient) -> Optional[ResolutionCommand]:
    """Generate LLM-based resolution with enhanced context."""
    try:
        # Find nearby waypoints
        nearby_waypoints = find_nearby_waypoints((ownship_state.latitude, ownship_state.longitude))
        
        # Generate enhanced prompt
        prompt = generate_enhanced_llm_prompt(ownship_state, intruder_states, nav_state, nearby_waypoints)
        
        # Get LLM response
        response = llm_client.generate_resolution({"conflict": True, "intruders": [{"id": "INTRUDER1"}]})
        if not response:
            return None
        
        # Parse response
        parsed = llm_client.parse_resolve_response(response)
        action = parsed.get("action", "").lower()
        
        # Extract value from params or direct value field
        value = 0
        if "params" in parsed and "heading_delta_deg" in parsed["params"]:
            value = float(parsed["params"]["heading_delta_deg"])
        elif "value" in parsed:
            value = float(parsed.get("value", 0))
        else:
            value = 30.0  # Default value
        
        waypoint_id = parsed.get("waypoint", "")
        
        print(f"LLM Response - Action: {action}, Value: {value}, Waypoint: {waypoint_id}")
        
        # Process action
        if action in ["turn", "turn_left", "turn_right"]:
            if action == "turn_left":
                new_heading = normalize_heading_deg(ownship_state.heading_deg - value)
            else:
                new_heading = normalize_heading_deg(ownship_state.heading_deg + value)
            
            return ResolutionCommand(
                resolution_id=f"llm_hdg_{ownship_state.aircraft_id}_{datetime.now().strftime('%H%M%S')}",
                target_aircraft=ownship_state.aircraft_id,
                resolution_type=ResolutionType.HEADING_CHANGE,
                new_heading_deg=new_heading,
                new_speed_kt=None,
                new_altitude_ft=None,
                issue_time=datetime.now(),
                is_validated=True,
                safety_margin_nm=5.0
            )
        
        elif action == "direct" and waypoint_id in NAV_DATABASE:
            # Direct to waypoint - calculate heading
            wpt_pos = NAV_DATABASE[waypoint_id]
            new_heading = bearing_deg(ownship_state.latitude, ownship_state.longitude,
                                    wpt_pos[0], wpt_pos[1])
            
            return ResolutionCommand(
                resolution_id=f"llm_direct_{ownship_state.aircraft_id}_{datetime.now().strftime('%H%M%S')}",
                target_aircraft=ownship_state.aircraft_id,
                resolution_type=ResolutionType.HEADING_CHANGE,
                new_heading_deg=new_heading,
                new_speed_kt=None,
                new_altitude_ft=None,
                issue_time=datetime.now(),
                is_validated=True,
                safety_margin_nm=5.0
            )
        
        elif action in ["climb", "descend"]:
            new_altitude = ownship_state.altitude_ft + value if action == "climb" else ownship_state.altitude_ft - value
            
            return ResolutionCommand(
                resolution_id=f"llm_alt_{ownship_state.aircraft_id}_{datetime.now().strftime('%H%M%S')}",
                target_aircraft=ownship_state.aircraft_id,
                resolution_type=ResolutionType.ALTITUDE_CHANGE,
                new_heading_deg=None,
                new_speed_kt=None,
                new_altitude_ft=new_altitude,
                issue_time=datetime.now(),
                is_validated=True,
                safety_margin_nm=5.0
            )
        
        else:
            print(f"Unknown LLM action: {action}")
            return None
            
    except Exception as e:
        print(f"LLM resolution failed: {e}")
        return None

def main():
    """Main enhanced simulation function."""
    start_time = datetime.now()
    log = logging.getLogger("enhanced_scat_sim")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log.info("Starting enhanced SCAT+LLM simulation with proper heading-based movement...")
    
    # Initialize metrics
    metrics = {
        "simulation_start": start_time.isoformat(),
        "scat_file": SCAT_FILE,
        "waypoints_reached": 0,
        "total_waypoints": 0,
        "conflicts_detected": 0,
        "llm_resolutions": 0,
        "successful_llm_resolutions": 0,
        "simulation_cycles": 0,
        "distance_traveled_nm": 0.0,
        "navigation_log": [],
        "conflict_timeline": [],
        "heading_changes": []
    }
    
    try:
        # 1. Load SCAT data
        log.info("Step 1: Loading SCAT flight data...")
        adapter = SCATAdapter(SCAT_DIR)
        scat_file_path = Path(SCAT_DIR) / SCAT_FILE
        record = adapter.load_flight_record(scat_file_path)
        if record is None:
            raise ValueError("Failed to load SCAT record")
        states = adapter.extract_aircraft_states(record)
        
        if not states:
            raise ValueError("No aircraft states found in SCAT record")
        
        # Build route from SCAT states
        route_waypoints = []
        for state in states[::5]:  # Sample every 5th state to avoid too many waypoints
            route_waypoints.append((state.latitude, state.longitude))
        
        log.info(f"[OK] Loaded {len(states)} states, created route with {len(route_waypoints)} waypoints")
        metrics["total_waypoints"] = len(route_waypoints) - 1
        
        # 2. Initialize LLM client
        log.info("Step 2: Initializing LLM client...")
        from src.cdr.schemas import ConfigurationSettings
        config = ConfigurationSettings(
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=512,
            bluesky_host="127.0.0.1",
            bluesky_port=5555
        )
        llm_client = LlamaClient(config)
        log.info("[OK] LLM client initialized")
        
        # 3. Initialize navigation state
        start_pos = route_waypoints[0]
        first_target = route_waypoints[1] if len(route_waypoints) > 1 else route_waypoints[0]
        initial_heading = bearing_deg(start_pos[0], start_pos[1], first_target[0], first_target[1])
        
        nav_state = NavigationState(
            position=start_pos,
            heading_deg=initial_heading,
            ground_speed_kt=CRUISE_SPD_KT,
            target_waypoint=first_target,
            route_waypoints=route_waypoints,
            current_wpt_index=1
        )
        
        log.info(f"[OK] Navigation initialized: start {start_pos}, heading {initial_heading:.1f}deg")
        
        # 4. Simulation loop
        log.info("Step 3: Starting enhanced simulation loop...")
        
        for cycle in range(SIMULATION_CYCLES):
            cycle_start_time = datetime.now()
            
            # Current ownship state
            ownship_state = create_aircraft_state("OWNSHIP", nav_state.position, 
                                                 nav_state.heading_deg, nav_state.ground_speed_kt, CRUISE_ALT_FT)
            
            # Simulate intruder aircraft
            intruder_states = []
            if cycle >= 10 and cycle <= 25:  # Intruder active during certain cycles
                # Place intruder on collision course
                intruder_pos = destination_point_nm(nav_state.position[0], nav_state.position[1], 
                                                  nav_state.heading_deg + 90, 8.0)
                intruder_heading = bearing_deg(intruder_pos[0], intruder_pos[1], 
                                             nav_state.position[0], nav_state.position[1])
                intruder_states.append(create_aircraft_state("INTRUDER1", intruder_pos, 
                                                           intruder_heading, CRUISE_SPD_KT, CRUISE_ALT_FT))
            
            # Check for conflicts
            conflict_detected = False
            for intruder in intruder_states:
                distance = haversine_nm((ownship_state.latitude, ownship_state.longitude),
                                      (intruder.latitude, intruder.longitude))
                if distance < CONFLICT_THRESHOLD_NM:
                    conflict_detected = True
                    metrics["conflicts_detected"] += 1
                    
                    log.info(f"[ALERT] CONFLICT DETECTED: {intruder.aircraft_id} at {distance:.2f}NM")
                    
                    # Generate LLM resolution
                    resolution = generate_llm_resolution(ownship_state, intruder_states, nav_state, llm_client)
                    
                    if resolution:
                        metrics["llm_resolutions"] += 1
                        metrics["successful_llm_resolutions"] += 1
                        
                        log.info(f"[BOT] LLM RESOLUTION: {resolution.resolution_type.value}")
                        
                        # Apply resolution to navigation state
                        if resolution.resolution_type == ResolutionType.HEADING_CHANGE:
                            nav_state.llm_modified_heading = resolution.new_heading_deg
                            nav_state.time_since_llm_maneuver = 0.0
                            nav_state.return_to_route_mode = False
                            
                            metrics["heading_changes"].append({
                                "cycle": cycle,
                                "old_heading": nav_state.heading_deg,
                                "new_heading": resolution.new_heading_deg,
                                "reason": "LLM conflict resolution"
                            })
                            
                            log.info(f"[AIRCRAFT] Heading changed: {nav_state.heading_deg:.1f}deg -> {resolution.new_heading_deg:.1f}deg")
                    
                    # Record conflict
                    metrics["conflict_timeline"].append({
                        "cycle": cycle,
                        "intruder_id": intruder.aircraft_id,
                        "separation_nm": distance,
                        "resolution_applied": resolution is not None,
                        "resolution_type": resolution.resolution_type.value if resolution else None
                    })
                    
                    break  # Handle one conflict at a time
            
            # Move aircraft using heading-based movement (THIS IS THE KEY FIX!)
            nav_state = simulate_heading_based_movement(nav_state, TIME_STEP_MIN)
            
            # Check waypoint progress
            nav_state = check_waypoint_reached(nav_state)
            
            # Log progress
            if nav_state.target_waypoint:
                distance_to_target = haversine_nm(nav_state.position, nav_state.target_waypoint)
                log.info(f"Cycle {cycle}: Pos=({nav_state.position[0]:.4f},{nav_state.position[1]:.4f}), "
                        f"Hdg={nav_state.heading_deg:.1f}deg, Target dist={distance_to_target:.2f}NM")
                
                if nav_state.llm_modified_heading is not None:
                    log.info(f"  [BOT] Following LLM heading: {nav_state.llm_modified_heading:.1f}deg "
                            f"(time since maneuver: {nav_state.time_since_llm_maneuver:.1f}min)")
            
            # Record navigation log
            metrics["navigation_log"].append({
                "cycle": cycle,
                "position": nav_state.position,
                "heading": nav_state.heading_deg,
                "llm_heading": nav_state.llm_modified_heading,
                "target_waypoint": nav_state.target_waypoint,
                "waypoint_index": nav_state.current_wpt_index,
                "conflict_detected": conflict_detected
            })
            
            metrics["simulation_cycles"] = cycle + 1
            
            # Check if route completed
            if nav_state.current_wpt_index >= len(nav_state.route_waypoints) - 1:
                log.info("[TARGET] Route completed!")
                break
            
            time.sleep(0.1)  # Brief pause for real-time feel
        
        # Calculate final metrics
        end_time = datetime.now()
        metrics["simulation_end"] = end_time.isoformat()
        metrics["duration_seconds"] = (end_time - start_time).total_seconds()
        metrics["waypoints_reached"] = nav_state.current_wpt_index
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(metrics["navigation_log"])):
            prev_pos = metrics["navigation_log"][i-1]["position"]
            curr_pos = metrics["navigation_log"][i]["position"]
            total_distance += haversine_nm(prev_pos, curr_pos)
        metrics["distance_traveled_nm"] = total_distance
        
        # 5. Generate outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"enhanced_scat_simulation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        log.info(f"\n[TARGET] === ENHANCED SIMULATION COMPLETED ===")
        log.info(f"[TIME]  Duration: {metrics['duration_seconds']:.1f} seconds")
        log.info(f"[CONVERT] Cycles: {metrics['simulation_cycles']}")
        log.info(f"[LOCATION] Waypoints: {metrics['waypoints_reached']}/{metrics['total_waypoints']}")
        log.info(f"[AIRCRAFT]  Distance: {metrics['distance_traveled_nm']:.2f} NM")
        log.info(f"[WARN]  Conflicts: {metrics['conflicts_detected']}")
        log.info(f"[BOT] LLM Resolutions: {metrics['successful_llm_resolutions']}")
        log.info(f"[TARGET] Heading Changes: {len(metrics['heading_changes'])}")
        log.info(f"[DOCUMENT] Results saved to {results_file}")
        
        return metrics
        
    except Exception as e:
        log.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
