"""
Simplified SCAT simulation script to test basic functionality.
This script focuses on:
1. Loading SCAT flight data
2. Creating aircraft in BlueSky
3. Running a basic simulation loop
4. Collecting metrics and outputting results
"""
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import src modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.geodesy import haversine_nm, bearing_deg, destination_point_nm, normalize_heading_deg

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("simple_scat_sim")

# Configuration
SCAT_DIR = r"F:\SCAT_extracted"
SCAT_FILE = "100000.json"
CRUISE_SPD_KT = 420.0
CRUISE_ALT_FT = 34000.0
SIMULATION_TIME_MIN = 30.0  # Run for 30 minutes
STEP_MIN = 1.0  # 1 minute steps

@dataclass
class SimulationMetrics:
    start_time: datetime
    end_time: datetime
    total_distance_nm: float
    waypoints_reached: int
    conflicts_detected: int
    resolutions_executed: int
    final_position: Tuple[float, float]
    simulation_success: bool

def build_route_from_scat(states, spacing_nm: float = 15.0) -> List[Tuple[float, float]]:
    """Build a simplified route from SCAT states."""
    if not states:
        return []
    
    # Sort states by timestamp
    states = sorted(states, key=lambda s: s.timestamp)
    
    # Start with first position
    route = [(states[0].latitude, states[0].longitude)]
    last_pos = route[0]
    
    # Add points that are at least spacing_nm apart
    for state in states[1:]:
        current_pos = (state.latitude, state.longitude)
        distance = haversine_nm(last_pos, current_pos)
        if distance >= spacing_nm:
            route.append(current_pos)
            last_pos = current_pos
    
    # Always include the final position
    final_pos = (states[-1].latitude, states[-1].longitude)
    if haversine_nm(route[-1], final_pos) > 1.0:
        route.append(final_pos)
    
    return route

def get_aircraft_states_safe(bs_client: BlueSkyClient, max_retries: int = 3) -> Dict[str, Dict[str, Any]]:
    """Safely get aircraft states with retry logic."""
    for attempt in range(max_retries):
        try:
            states = bs_client.get_aircraft_states()
            if states:
                return states
            log.warning(f"No aircraft states returned (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            log.error(f"Error getting aircraft states (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)
    
    return {}

def create_and_verify_aircraft(bs_client: BlueSkyClient, callsign: str, actype: str, 
                              lat: float, lon: float, hdg: float, alt: float, spd: float) -> bool:
    """Create aircraft and verify it appears in the simulation."""
    log.info(f"Creating {callsign} at ({lat:.4f}, {lon:.4f})")
    
    # Create the aircraft
    success = bs_client.create_aircraft(callsign, actype, lat, lon, hdg, alt, spd)
    if not success:
        log.error(f"Failed to create {callsign}")
        return False
    
    # Wait and verify it appears
    for attempt in range(10):
        time.sleep(2)
        states = get_aircraft_states_safe(bs_client)
        if callsign.upper() in states:
            log.info(f"{callsign} successfully created and verified")
            return True
        log.info(f"Waiting for {callsign} to appear (attempt {attempt + 1}/10)")
    
    log.error(f"{callsign} failed to appear in simulation")
    return False

def step_simulation_safe(bs_client: BlueSkyClient, step_min: float, max_retries: int = 3) -> bool:
    """Safely step the simulation with retry logic."""
    for attempt in range(max_retries):
        try:
            success = bs_client.step_minutes(step_min)
            if success:
                return True
            log.warning(f"Simulation step failed (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            log.error(f"Simulation step error (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(0.5)
    
    return False

def main():
    """Main simulation function."""
    start_time = datetime.now()
    metrics = SimulationMetrics(
        start_time=start_time,
        end_time=start_time,
        total_distance_nm=0.0,
        waypoints_reached=0,
        conflicts_detected=0,
        resolutions_executed=0,
        final_position=(0.0, 0.0),
        simulation_success=False
    )
    
    try:
        # 1. Load SCAT flight data
        log.info("Loading SCAT flight data...")
        adapter = SCATAdapter(SCAT_DIR)
        scat_file_path = Path(SCAT_DIR) / SCAT_FILE
        
        if not scat_file_path.exists():
            log.error(f"SCAT file not found: {scat_file_path}")
            return
        
        record = adapter.load_flight_record(scat_file_path)
        states = adapter.extract_aircraft_states(record)
        
        if not states:
            log.error("No aircraft states found in SCAT record")
            return
        
        log.info(f"Loaded {len(states)} states from SCAT record")
        
        # Build route
        route = build_route_from_scat(states)
        log.info(f"Built route with {len(route)} waypoints")
        
        if len(route) < 2:
            log.error("Route must have at least 2 waypoints")
            return
        
        # 2. Initialize BlueSky
        log.info("Initializing BlueSky...")
        cfg = ConfigurationSettings(
            polling_interval_min=STEP_MIN,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            bluesky_host="127.0.0.1",
            bluesky_port=5555,
            fast_time=True,
            sim_accel_factor=60.0
        )
        
        bs_client = BlueSkyClient(cfg)
        if not bs_client.connect():
            log.error("Failed to connect to BlueSky")
            return
        
        # Reset simulation
        bs_client.sim_reset()
        bs_client.sim_realtime(False)
        bs_client.sim_set_dtmult(60)  # 60x speed
        
        # 3. Create aircraft
        log.info("Creating aircraft...")
        
        # Create ownship at first waypoint
        lat0, lon0 = route[0]
        hdg0 = bearing_deg(lat0, lon0, route[1][0], route[1][1])
        
        if not create_and_verify_aircraft(bs_client, "OWNSHIP", "A320", 
                                         lat0, lon0, hdg0, CRUISE_ALT_FT, CRUISE_SPD_KT):
            log.error("Failed to create OWNSHIP")
            return
        
        # Create intruder near middle of route
        mid_idx = len(route) // 2
        mid_lat, mid_lon = route[mid_idx]
        intr_lat, intr_lon = destination_point_nm(mid_lat, mid_lon, 270.0, 30.0)
        
        if not create_and_verify_aircraft(bs_client, "INTRUDER1", "B738",
                                         intr_lat, intr_lon, 90.0, CRUISE_ALT_FT, CRUISE_SPD_KT):
            log.warning("Failed to create INTRUDER1, continuing with just OWNSHIP")
        
        # 4. Run simulation loop
        log.info("Starting simulation loop...")
        simulation_time = 0.0
        current_waypoint = 1
        
        while simulation_time < SIMULATION_TIME_MIN and current_waypoint < len(route):
            # Get current aircraft state
            states = get_aircraft_states_safe(bs_client)
            
            if "OWNSHIP" not in states:
                log.error("OWNSHIP not found in simulation, stopping")
                break
            
            ownship = states["OWNSHIP"]
            current_pos = (float(ownship["lat"]), float(ownship["lon"]))
            current_hdg = float(ownship["hdg_deg"])
            current_spd = float(ownship["spd_kt"])
            
            # Check if we've reached the current waypoint
            target_pos = route[current_waypoint]
            distance_to_waypoint = haversine_nm(current_pos, target_pos)
            
            log.info(f"Time: {simulation_time:.1f}min, Pos: ({current_pos[0]:.4f}, {current_pos[1]:.4f}), "
                    f"Hdg: {current_hdg:.1f}°, Spd: {current_spd:.0f}kt, "
                    f"Dist to WPT{current_waypoint}: {distance_to_waypoint:.2f}NM")
            
            if distance_to_waypoint <= 2.0:  # Within 2 NM of waypoint
                log.info(f"Reached waypoint {current_waypoint}")
                metrics.waypoints_reached += 1
                current_waypoint += 1
                
                if current_waypoint < len(route):
                    # Set heading to next waypoint
                    next_target = route[current_waypoint]
                    new_hdg = bearing_deg(current_pos[0], current_pos[1], 
                                        next_target[0], next_target[1])
                    bs_client.set_heading("OWNSHIP", new_hdg)
                    log.info(f"Set heading to {new_hdg:.1f}° for waypoint {current_waypoint}")
            else:
                # Navigate towards current waypoint
                desired_hdg = bearing_deg(current_pos[0], current_pos[1], 
                                        target_pos[0], target_pos[1])
                if abs(normalize_heading_deg(desired_hdg - current_hdg)) > 5.0:
                    bs_client.set_heading("OWNSHIP", desired_hdg)
            
            # Simple conflict detection with intruder
            if "INTRUDER1" in states:
                intruder = states["INTRUDER1"]
                intruder_pos = (float(intruder["lat"]), float(intruder["lon"]))
                separation = haversine_nm(current_pos, intruder_pos)
                
                if separation < 10.0:  # Within 10 NM
                    log.info(f"Close traffic detected: {separation:.2f} NM separation")
                    metrics.conflicts_detected += 1
                    
                    if separation < 6.0:  # Conflict threshold
                        log.info("Executing conflict resolution: Turn right 30°")
                        new_hdg = normalize_heading_deg(current_hdg + 30.0)
                        bs_client.set_heading("OWNSHIP", new_hdg)
                        metrics.resolutions_executed += 1
            
            # Step simulation
            if not step_simulation_safe(bs_client, STEP_MIN):
                log.error("Failed to step simulation, stopping")
                break
            
            simulation_time += STEP_MIN
        
        # 5. Collect final metrics
        final_states = get_aircraft_states_safe(bs_client)
        if "OWNSHIP" in final_states:
            final_ownship = final_states["OWNSHIP"]
            metrics.final_position = (float(final_ownship["lat"]), float(final_ownship["lon"]))
            metrics.total_distance_nm = haversine_nm(route[0], metrics.final_position)
            metrics.simulation_success = True
        
        metrics.end_time = datetime.now()
        
        # 6. Output results
        log.info("Simulation completed successfully!")
        log.info(f"Duration: {(metrics.end_time - metrics.start_time).total_seconds():.1f} seconds")
        log.info(f"Waypoints reached: {metrics.waypoints_reached}/{len(route)}")
        log.info(f"Total distance: {metrics.total_distance_nm:.2f} NM")
        log.info(f"Conflicts detected: {metrics.conflicts_detected}")
        log.info(f"Resolutions executed: {metrics.resolutions_executed}")
        
        # Save results to JSON
        results = {
            "simulation_metadata": {
                "scat_file": SCAT_FILE,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat(),
                "duration_seconds": (metrics.end_time - metrics.start_time).total_seconds(),
                "simulation_success": metrics.simulation_success
            },
            "flight_metrics": {
                "waypoints_total": len(route),
                "waypoints_reached": metrics.waypoints_reached,
                "completion_percentage": (metrics.waypoints_reached / len(route)) * 100,
                "total_distance_nm": metrics.total_distance_nm,
                "final_position": {
                    "latitude": metrics.final_position[0],
                    "longitude": metrics.final_position[1]
                }
            },
            "conflict_metrics": {
                "conflicts_detected": metrics.conflicts_detected,
                "resolutions_executed": metrics.resolutions_executed,
                "resolution_success_rate": (metrics.resolutions_executed / max(metrics.conflicts_detected, 1)) * 100
            },
            "route_waypoints": [{"lat": lat, "lon": lon} for lat, lon in route]
        }
        
        output_file = f"scat_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        log.info(f"Results saved to {output_file}")
        
        # Create a simple CSV summary
        csv_file = f"scat_simulation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_file, 'w') as f:
            f.write("Metric,Value\\n")
            f.write(f"SCAT_File,{SCAT_FILE}\\n")
            f.write(f"Simulation_Success,{metrics.simulation_success}\\n")
            f.write(f"Duration_Seconds,{(metrics.end_time - metrics.start_time).total_seconds():.1f}\\n")
            f.write(f"Waypoints_Reached,{metrics.waypoints_reached}\\n")
            f.write(f"Waypoints_Total,{len(route)}\\n")
            f.write(f"Completion_Percentage,{(metrics.waypoints_reached / len(route)) * 100:.1f}\\n")
            f.write(f"Total_Distance_NM,{metrics.total_distance_nm:.2f}\\n")
            f.write(f"Conflicts_Detected,{metrics.conflicts_detected}\\n")
            f.write(f"Resolutions_Executed,{metrics.resolutions_executed}\\n")
        
        log.info(f"Summary saved to {csv_file}")
        
    except Exception as e:
        log.error(f"Simulation failed with error: {e}")
        metrics.simulation_success = False
        raise
    
    finally:
        # Cleanup
        try:
            if 'bs_client' in locals():
                log.info("Cleaning up BlueSky connection...")
        except Exception as e:
            log.warning(f"Cleanup error: {e}")

if __name__ == "__main__":
    main()
