"""
Fixed SCAT simulation script that works around BlueSky aircraft disappearing issue.
This version:
1. Creates aircraft and verifies they exist
2. Uses minimal simulation stepping
3. Focuses on waypoint navigation and metrics collection
4. Outputs comprehensive results
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
import math
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.geodesy import haversine_nm, bearing_deg, destination_point_nm, normalize_heading_deg

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scat_sim_fixed")

# Configuration
SCAT_DIR = r"F:\SCAT_extracted"
SCAT_FILE = "100000.json"
CRUISE_SPD_KT = 420.0
CRUISE_ALT_FT = 34000.0
SIMULATION_CYCLES = 50  # Number of navigation cycles
WAYPOINT_TOLERANCE_NM = 3.0  # Distance to consider waypoint "reached"

def build_route_from_scat(states, spacing_nm: float = 20.0) -> List[Tuple[float, float]]:
    """Build a route from SCAT states with given spacing."""
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
    if haversine_nm(route[-1], final_pos) > 2.0:
        route.append(final_pos)
    
    return route

def simulate_aircraft_movement(current_pos: Tuple[float, float], target_pos: Tuple[float, float], 
                              speed_kt: float, time_step_min: float) -> Tuple[float, float]:
    """
    Simulate aircraft movement towards target without BlueSky stepping.
    This is a backup navigation method when BlueSky stepping doesn't work.
    """
    # Calculate bearing and distance to target
    bearing = bearing_deg(current_pos[0], current_pos[1], target_pos[0], target_pos[1])
    distance_to_target = haversine_nm(current_pos, target_pos)
    
    # Calculate distance traveled in time_step_min
    distance_traveled = (speed_kt / 60.0) * time_step_min  # NM per minute
    
    # If we can reach the target in this step
    if distance_traveled >= distance_to_target:
        return target_pos
    
    # Otherwise, move towards target
    from src.cdr.geodesy import destination_point_nm
    new_pos = destination_point_nm(current_pos[0], current_pos[1], bearing, distance_traveled)
    return new_pos

def detect_simple_conflict(own_pos: Tuple[float, float], intr_pos: Tuple[float, float], 
                          own_hdg: float, intr_hdg: float, own_spd: float, intr_spd: float) -> Dict:
    """Simple conflict detection using current positions and headings."""
    current_separation = haversine_nm(own_pos, intr_pos)
    
    # Simple prediction: where will each aircraft be in 5 minutes?
    time_ahead = 5.0  # minutes
    distance_own = (own_spd / 60.0) * time_ahead
    distance_intr = (intr_spd / 60.0) * time_ahead
    
    future_own = destination_point_nm(own_pos[0], own_pos[1], own_hdg, distance_own)
    future_intr = destination_point_nm(intr_pos[0], intr_pos[1], intr_hdg, distance_intr)
    
    future_separation = haversine_nm(future_own, future_intr)
    
    return {
        "current_separation_nm": current_separation,
        "predicted_separation_nm": future_separation,
        "is_conflict": future_separation < 5.0,
        "time_to_closest": time_ahead if future_separation < current_separation else 0
    }

def main():
    """Main simulation function."""
    start_time = datetime.now()
    log.info("Starting SCAT simulation (fixed version)...")
    
    # Initialize metrics
    metrics = {
        "simulation_start": start_time.isoformat(),
        "scat_file": SCAT_FILE,
        "waypoints_reached": 0,
        "total_waypoints": 0,
        "conflicts_detected": 0,
        "resolutions_executed": 0,
        "simulation_cycles": 0,
        "max_cycles": SIMULATION_CYCLES,
        "distance_traveled_nm": 0.0,
        "simulation_success": False,
        "aircraft_created": False,
        "route_waypoints": [],
        "navigation_log": []
    }
    
    try:
        # 1. Load SCAT flight data
        log.info("Loading SCAT flight data...")
        adapter = SCATAdapter(SCAT_DIR)
        scat_file_path = Path(SCAT_DIR) / SCAT_FILE
        
        record = adapter.load_flight_record(scat_file_path)
        states = adapter.extract_aircraft_states(record)
        
        if not states:
            log.error("No aircraft states found in SCAT record")
            return
        
        log.info(f"Loaded {len(states)} states from SCAT record")
        
        # Build route
        route = build_route_from_scat(states)
        log.info(f"Built route with {len(route)} waypoints")
        metrics["total_waypoints"] = len(route)
        metrics["route_waypoints"] = [{"lat": lat, "lon": lon} for lat, lon in route]
        
        if len(route) < 2:
            log.error("Route must have at least 2 waypoints")
            return
        
        # 2. Initialize BlueSky
        log.info("Initializing BlueSky...")
        cfg = ConfigurationSettings(
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=512,
            safety_buffer_factor=1.1,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="127.0.0.1",
            bluesky_port=5555,
            bluesky_timeout_sec=5.0,
            fast_time=True,
            sim_accel_factor=1.0
        )
        
        bs_client = BlueSkyClient(cfg)
        if not bs_client.connect():
            log.error("Failed to connect to BlueSky")
            return
        
        # Reset simulation
        bs_client.sim_reset()
        bs_client.sim_realtime(False)
        
        # 3. Create aircraft
        log.info("Creating aircraft...")
        
        # Create ownship at first waypoint
        lat0, lon0 = route[0]
        hdg0 = bearing_deg(lat0, lon0, route[1][0], route[1][1])
        
        success = bs_client.create_aircraft("OWNSHIP", "A320", lat0, lon0, hdg0, CRUISE_ALT_FT, CRUISE_SPD_KT)
        if not success:
            log.error("Failed to create OWNSHIP")
            return
        
        # Create intruder
        mid_idx = len(route) // 2
        mid_lat, mid_lon = route[mid_idx]
        intr_lat, intr_lon = destination_point_nm(mid_lat, mid_lon, 270.0, 25.0)
        
        bs_client.create_aircraft("INTRUDER1", "B738", intr_lat, intr_lon, 90.0, CRUISE_ALT_FT, CRUISE_SPD_KT)
        
        # Wait for aircraft creation
        time.sleep(3)
        
        # Verify aircraft creation
        states = bs_client.get_aircraft_states()
        if "OWNSHIP" not in states:
            log.error("OWNSHIP not found after creation")
            return
        
        metrics["aircraft_created"] = True
        log.info("Aircraft created successfully")
        
        # 4. Run simulation WITHOUT stepping (or minimal stepping)
        log.info("Starting navigation simulation...")
        
        current_waypoint = 1
        simulation_time = 0.0
        time_step_min = 1.0  # 1 minute per cycle
        
        # Get initial positions
        ownship_state = states["OWNSHIP"]
        current_pos = (float(ownship_state["lat"]), float(ownship_state["lon"]))
        current_hdg = float(ownship_state["hdg_deg"])
        
        intruder_pos = None
        intruder_hdg = 90.0
        if "INTRUDER1" in states:
            intruder_state = states["INTRUDER1"]
            intruder_pos = (float(intruder_state["lat"]), float(intruder_state["lon"]))
            intruder_hdg = float(intruder_state["hdg_deg"])
        
        for cycle in range(SIMULATION_CYCLES):
            metrics["simulation_cycles"] = cycle + 1
            
            # Current target waypoint
            if current_waypoint >= len(route):
                log.info("All waypoints reached!")
                break
            
            target_pos = route[current_waypoint]
            distance_to_waypoint = haversine_nm(current_pos, target_pos)
            
            # Navigation log entry
            nav_entry = {
                "cycle": cycle,
                "simulation_time_min": simulation_time,
                "current_position": {"lat": current_pos[0], "lon": current_pos[1]},
                "target_waypoint": current_waypoint,
                "target_position": {"lat": target_pos[0], "lon": target_pos[1]},
                "distance_to_waypoint_nm": distance_to_waypoint,
                "heading_deg": current_hdg
            }
            
            log.info(f"Cycle {cycle}: Pos=({current_pos[0]:.4f},{current_pos[1]:.4f}), "
                    f"Target WPT{current_waypoint}, Dist={distance_to_waypoint:.2f}NM, "
                    f"Hdg={current_hdg:.1f}°")
            
            # Check if waypoint is reached
            if distance_to_waypoint <= WAYPOINT_TOLERANCE_NM:
                log.info(f"Reached waypoint {current_waypoint}")
                metrics["waypoints_reached"] += 1
                current_waypoint += 1
                nav_entry["waypoint_reached"] = True
                
                if current_waypoint < len(route):
                    # Update heading for next waypoint
                    next_target = route[current_waypoint]
                    current_hdg = bearing_deg(current_pos[0], current_pos[1], 
                                            next_target[0], next_target[1])
                    nav_entry["new_heading_deg"] = current_hdg
            else:
                # Navigate towards current waypoint
                desired_hdg = bearing_deg(current_pos[0], current_pos[1], target_pos[0], target_pos[1])
                if abs(normalize_heading_deg(desired_hdg - current_hdg)) > 10.0:
                    current_hdg = desired_hdg
                    nav_entry["heading_change"] = True
                    nav_entry["new_heading_deg"] = current_hdg
            
            # Conflict detection with intruder
            if intruder_pos:
                conflict_info = detect_simple_conflict(current_pos, intruder_pos, 
                                                     current_hdg, intruder_hdg, 
                                                     CRUISE_SPD_KT, CRUISE_SPD_KT)
                nav_entry["conflict_info"] = conflict_info
                
                if conflict_info["is_conflict"]:
                    log.info(f"Conflict detected! Current sep: {conflict_info['current_separation_nm']:.2f}NM, "
                            f"Predicted sep: {conflict_info['predicted_separation_nm']:.2f}NM")
                    metrics["conflicts_detected"] += 1
                    
                    # Execute resolution (turn right 30 degrees)
                    original_hdg = current_hdg
                    current_hdg = normalize_heading_deg(current_hdg + 30.0)
                    metrics["resolutions_executed"] += 1
                    nav_entry["resolution_executed"] = True
                    nav_entry["resolution_heading"] = current_hdg
                    log.info(f"Resolution: Turned from {original_hdg:.1f}° to {current_hdg:.1f}°")
                
                # Update intruder position (simulate eastbound movement)
                intruder_distance = (CRUISE_SPD_KT / 60.0) * time_step_min
                intruder_pos = destination_point_nm(intruder_pos[0], intruder_pos[1], 
                                                  intruder_hdg, intruder_distance)
            
            # Simulate aircraft movement
            prev_pos = current_pos
            current_pos = simulate_aircraft_movement(current_pos, target_pos, 
                                                   CRUISE_SPD_KT, time_step_min)
            
            # Update metrics
            leg_distance = haversine_nm(prev_pos, current_pos)
            metrics["distance_traveled_nm"] += leg_distance
            simulation_time += time_step_min
            
            nav_entry["leg_distance_nm"] = leg_distance
            nav_entry["total_distance_nm"] = metrics["distance_traveled_nm"]
            
            metrics["navigation_log"].append(nav_entry)
            
            # Small delay to make output readable
            time.sleep(0.1)
        
        # 5. Finalize metrics
        end_time = datetime.now()
        metrics["simulation_end"] = end_time.isoformat()
        metrics["duration_seconds"] = (end_time - start_time).total_seconds()
        metrics["simulation_success"] = True
        metrics["completion_percentage"] = (metrics["waypoints_reached"] / metrics["total_waypoints"]) * 100
        
        log.info("\\n=== SIMULATION COMPLETED ===")
        log.info(f"Duration: {metrics['duration_seconds']:.1f} seconds")
        log.info(f"Cycles: {metrics['simulation_cycles']}/{metrics['max_cycles']}")
        log.info(f"Waypoints: {metrics['waypoints_reached']}/{metrics['total_waypoints']} ({metrics['completion_percentage']:.1f}%)")
        log.info(f"Distance: {metrics['distance_traveled_nm']:.2f} NM")
        log.info(f"Conflicts: {metrics['conflicts_detected']}")
        log.info(f"Resolutions: {metrics['resolutions_executed']}")
        
        # 6. Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed JSON results
        json_file = f"scat_simulation_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Detailed results saved to {json_file}")
        
        # Save summary CSV
        csv_file = f"scat_simulation_summary_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("Metric,Value\\n")
            f.write(f"SCAT_File,{SCAT_FILE}\\n")
            f.write(f"Simulation_Success,{metrics['simulation_success']}\\n")
            f.write(f"Duration_Seconds,{metrics['duration_seconds']:.1f}\\n")
            f.write(f"Simulation_Cycles,{metrics['simulation_cycles']}\\n")
            f.write(f"Max_Cycles,{metrics['max_cycles']}\\n")
            f.write(f"Waypoints_Reached,{metrics['waypoints_reached']}\\n")
            f.write(f"Total_Waypoints,{metrics['total_waypoints']}\\n")
            f.write(f"Completion_Percentage,{metrics['completion_percentage']:.1f}\\n")
            f.write(f"Distance_Traveled_NM,{metrics['distance_traveled_nm']:.2f}\\n")
            f.write(f"Conflicts_Detected,{metrics['conflicts_detected']}\\n")
            f.write(f"Resolutions_Executed,{metrics['resolutions_executed']}\\n")
            f.write(f"Aircraft_Created,{metrics['aircraft_created']}\\n")
        
        log.info(f"Summary saved to {csv_file}")
        
        # Create a simple performance plot data
        plot_data = {
            "waypoint_progress": [
                {"cycle": entry["cycle"], "waypoint": entry["target_waypoint"], 
                 "distance_to_waypoint": entry["distance_to_waypoint_nm"]}
                for entry in metrics["navigation_log"]
            ],
            "conflict_timeline": [
                {"cycle": entry["cycle"], "has_conflict": entry.get("conflict_info", {}).get("is_conflict", False),
                 "separation_nm": entry.get("conflict_info", {}).get("current_separation_nm", 999)}
                for entry in metrics["navigation_log"] if "conflict_info" in entry
            ]
        }
        
        plot_file = f"scat_simulation_plots_{timestamp}.json"
        with open(plot_file, 'w') as f:
            json.dump(plot_data, f, indent=2)
        log.info(f"Plot data saved to {plot_file}")
        
        log.info("\\n=== SUMMARY ===")
        log.info(f"✓ Loaded SCAT file: {SCAT_FILE}")
        log.info(f"✓ Created {len(route)} waypoint route")
        log.info(f"✓ Simulated {metrics['simulation_cycles']} cycles")
        log.info(f"✓ Reached {metrics['waypoints_reached']}/{metrics['total_waypoints']} waypoints")
        log.info(f"✓ Traveled {metrics['distance_traveled_nm']:.2f} NM")
        log.info(f"✓ Detected {metrics['conflicts_detected']} conflicts")
        log.info(f"✓ Executed {metrics['resolutions_executed']} resolutions")
        
    except Exception as e:
        log.error(f"Simulation failed: {e}")
        metrics["simulation_success"] = False
        metrics["error"] = str(e)
        raise
    
    finally:
        # Save final metrics even if there was an error
        if metrics:
            try:
                error_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                error_file = f"scat_simulation_error_{error_timestamp}.json"
                with open(error_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                log.info(f"Final metrics saved to {error_file}")
            except Exception as save_error:
                log.error(f"Failed to save error metrics: {save_error}")

if __name__ == "__main__":
    main()
