#!/usr/bin/env python3
"""
SCAT LLM Runner - Run LLM-driven simulation with SCAT data

This module runs LLM-driven conflict detection and resolution simulations
using real SCAT trajectory data with BlueSky integration.

Usage:
    python scat_llm_run.py --ownship <file|id> [--intruders auto|file] [--realtime] [--dt-min 1]
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add the parent directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState, ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient, BSConfig
from src.cdr.llm_client import LlamaClient
from src.cdr.geodesy import haversine_nm, bearing_deg, destination_point_nm
from src.cdr.detect import predict_conflicts
from src.cdr.resolve import execute_resolution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log = logging.getLogger("scat_llm_run")


class SCATLLMRunner:
    """Run LLM-driven simulation with SCAT data and BlueSky motion."""
    
    def __init__(self, scat_root: str, realtime: bool = False, dt_min: float = 1.0):
        """Initialize runner with configuration."""
        self.scat_root = Path(scat_root)
        self.adapter = SCATAdapter(str(self.scat_root))
        self.realtime = realtime
        self.dt_min = dt_min
        
        # Initialize BlueSky
        self.bs_config = BSConfig()
        self.bluesky = BlueSkyClient(self.bs_config)
        
        # Initialize LLM
        self.llm_config = ConfigurationSettings(
            # Timing settings  
            polling_interval_min=self.dt_min,
            lookahead_time_min=10.0,
            snapshot_interval_min=1.0,
            
            # PromptBuilderV2 settings
            max_intruders_in_prompt=5,
            intruder_proximity_nm=100.0,
            intruder_altitude_diff_ft=5000.0,
            trend_analysis_window_min=2.0,
            
            # Separation standards
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            
            # LLM settings
            llm_enabled=True,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=512,
            
            # Safety settings
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
            
            # BlueSky integration
            bluesky_host="127.0.0.1",
            bluesky_port=5555,
            bluesky_timeout_sec=5.0,
            
            # Fast-time simulation
            fast_time=True,
            sim_accel_factor=1.0
        )
        self.llm_client = LlamaClient(self.llm_config)
        
        # Simulation state
        self.aircraft_routes = {}
        self.simulation_results = {
            "start_time": datetime.now().isoformat(),
            "configuration": {
                "realtime": realtime,
                "dt_min": dt_min,
                "llm_model": self.llm_config.llm_model_name
            },
            "events": [],
            "conflicts": [],
            "resolutions": [],
            "metrics": {}
        }
    
    def load_ownship(self, ownship_spec: str) -> Tuple[str, List[Tuple[float, float]]]:
        """Load ownship from file or ID specification.
        
        Args:
            ownship_spec: File path or aircraft ID
            
        Returns:
            Tuple of (callsign, route_waypoints)
        """
        if ownship_spec.endswith('.json'):
            # Load from file
            ownship_path = self.scat_root / ownship_spec
            if not ownship_path.exists():
                raise FileNotFoundError(f"Ownship file not found: {ownship_path}")
        else:
            # Treat as ID, find corresponding file
            ownship_path = self.scat_root / f"{ownship_spec}.json"
            if not ownship_path.exists():
                raise FileNotFoundError(f"Ownship file not found: {ownship_path}")
        
        # Load flight record
        record = self.adapter.load_flight_record(ownship_path)
        if not record:
            raise ValueError(f"Failed to load ownship data from {ownship_path}")
        
        # Extract states and build route
        states = self.adapter.extract_aircraft_states(record)
        if not states:
            raise ValueError(f"No aircraft states found in {ownship_path}")
        
        # Build route from states (sample every 10 NM)
        route = self._build_route_from_states(states, spacing_nm=10.0)
        
        log.info(f"Loaded ownship {record.callsign} with {len(route)} waypoints")
        
        return record.callsign, route
    
    def load_intruders(self, intruder_spec: str, ownship_route: List[Tuple[float, float]], 
                      max_intruders: int = 3) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Load intruder aircraft.
        
        Args:
            intruder_spec: 'auto' for automatic selection, or file path
            ownship_route: Ownship route for proximity analysis
            max_intruders: Maximum number of intruders to load
            
        Returns:
            List of (callsign, route) tuples
        """
        intruders = []
        
        if intruder_spec == 'auto':
            # Automatically find flights that come close to ownship route
            log.info("Auto-selecting intruder flights based on proximity...")
            intruders = self._auto_select_intruders(ownship_route, max_intruders)
        else:
            # Load specific intruder file
            intruder_path = self.scat_root / intruder_spec
            if not intruder_path.exists():
                raise FileNotFoundError(f"Intruder file not found: {intruder_path}")
            
            record = self.adapter.load_flight_record(intruder_path)
            if record:
                states = self.adapter.extract_aircraft_states(record)
                if states:
                    route = self._build_route_from_states(states, spacing_nm=10.0)
                    intruders.append((record.callsign, route))
        
        log.info(f"Loaded {len(intruders)} intruder aircraft")
        return intruders
    
    def run_simulation(self, ownship_spec: str, intruder_spec: str = 'auto', 
                      duration_min: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete LLM-driven simulation.
        
        Args:
            ownship_spec: Ownship specification (file or ID)
            intruder_spec: Intruder specification ('auto' or file)
            duration_min: Maximum simulation duration (None for full route)
            
        Returns:
            Simulation results dictionary
        """
        log.info("=== Starting SCAT LLM Simulation ===")
        
        try:
            # Connect to BlueSky
            if not self.bluesky.connect():
                raise RuntimeError("Failed to connect to BlueSky")
            
            # Reset simulation
            self.bluesky.sim_reset()
            self.bluesky.sim_realtime(self.realtime)
            if not self.realtime:
                self.bluesky.sim_set_dtmult(60.0)  # Speed up if not realtime
            
            # Extra safety: flush once more to ensure reset is processed
            self.bluesky.step_minutes(0.1)
            
            # Load ownship
            ownship_callsign, ownship_route = self.load_ownship(ownship_spec)
            self.ownship_callsign = ownship_callsign  # Store for later use
            self.aircraft_routes[ownship_callsign] = ownship_route
            
            # Load intruders
            intruders = self.load_intruders(intruder_spec, ownship_route)
            for callsign, route in intruders:
                self.aircraft_routes[callsign] = route
            
            # Create aircraft in BlueSky
            self._create_aircraft_in_bluesky(ownship_callsign, ownship_route, intruders)
            
            # Prime step: small step after creation to let internal state settle
            log.info("Performing prime step after aircraft creation...")
            self.bluesky.step_minutes(0.1)
            
            # Run simulation loop
            self._run_simulation_loop(duration_min)
            
            # Finalize results
            self.simulation_results["end_time"] = datetime.now().isoformat()
            self.simulation_results["duration_min"] = (
                datetime.fromisoformat(self.simulation_results["end_time"]) - 
                datetime.fromisoformat(self.simulation_results["start_time"])
            ).total_seconds() / 60.0
            
            log.info("=== Simulation Complete ===")
            return self.simulation_results
            
        except Exception as e:
            log.error(f"Simulation failed: {e}")
            self.simulation_results["error"] = str(e)
            raise
    
    def _build_route_from_states(self, states: List[AircraftState], spacing_nm: float) -> List[Tuple[float, float]]:
        """Build route waypoints from aircraft states."""
        if not states:
            return []
        
        # Sort by timestamp
        states = sorted(states, key=lambda s: s.timestamp)
        
        route = [(states[0].latitude, states[0].longitude)]
        last_pos = route[0]
        
        for state in states[1:]:
            current_pos = (state.latitude, state.longitude)
            distance = haversine_nm(last_pos, current_pos)
            if distance >= spacing_nm:
                route.append(current_pos)
                last_pos = current_pos
        
        # Always include final position
        final_pos = (states[-1].latitude, states[-1].longitude)
        if haversine_nm(route[-1], final_pos) > 2.0:
            route.append(final_pos)
        
        return route
    
    def _auto_select_intruders(self, ownship_route: List[Tuple[float, float]], 
                              max_intruders: int) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Automatically select intruder flights based on proximity to ownship route."""
        candidates = []
        
        # Find all SCAT files
        scat_files = list(self.scat_root.glob("*.json"))
        
        for scat_file in scat_files[:20]:  # Limit search for performance
            try:
                record = self.adapter.load_flight_record(scat_file)
                if not record:
                    continue
                
                states = self.adapter.extract_aircraft_states(record)
                if not states:
                    continue
                
                route = self._build_route_from_states(states, spacing_nm=15.0)
                if len(route) < 3:
                    continue
                
                # Check for proximity to ownship route
                min_distance = self._calculate_route_proximity(ownship_route, route)
                
                if min_distance < 50.0:  # Within 50 NM
                    candidates.append((record.callsign, route, min_distance))
                    
            except Exception as e:
                log.debug(f"Error processing {scat_file}: {e}")
                continue
        
        # Sort by proximity and take the closest ones
        candidates.sort(key=lambda x: x[2])
        selected = candidates[:max_intruders]
        
        return [(callsign, route) for callsign, route, _ in selected]
    
    def _calculate_route_proximity(self, route1: List[Tuple[float, float]], 
                                  route2: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance between two routes."""
        min_distance = float('inf')
        
        for pos1 in route1:
            for pos2 in route2:
                distance = haversine_nm(pos1, pos2)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _create_aircraft_in_bluesky(self, ownship_callsign: str, ownship_route: List[Tuple[float, float]],
                                   intruders: List[Tuple[str, List[Tuple[float, float]]]]):
        """Create all aircraft in BlueSky with route following."""
        aircraft_to_create = []
        
        # Prepare ownship
        lat0, lon0 = ownship_route[0]
        hdg0 = bearing_deg(lat0, lon0, ownship_route[1][0], ownship_route[1][1]) if len(ownship_route) > 1 else 90.0
        aircraft_to_create.append((ownship_callsign, "A320", lat0, lon0, hdg0, 35000.0, 420.0, ownship_route[1:]))
        
        # Prepare intruders
        for i, (callsign, route) in enumerate(intruders):
            if len(route) > 1:
                lat, lon = route[0]
                hdg = bearing_deg(lat, lon, route[1][0], route[1][1])
                # Stagger altitudes to avoid initial conflicts
                altitude = 35000.0 + (i * 2000.0)
                aircraft_to_create.append((callsign, "B737", lat, lon, hdg, altitude, 420.0, route[1:]))
        
        # Step 1: Create all aircraft first
        log.info(f"Creating {len(aircraft_to_create)} aircraft in BlueSky...")
        for callsign, actype, lat, lon, hdg, alt, spd, route in aircraft_to_create:
            success = self.bluesky.create_aircraft(callsign, actype, lat, lon, hdg, alt, spd)
            if not success:
                if callsign == ownship_callsign:
                    raise RuntimeError(f"Failed to create ownship {ownship_callsign}")
                else:
                    log.warning(f"Failed to create intruder {callsign}")
        
        # Step 2: Let BlueSky process aircraft creation
        log.info("Allowing BlueSky to process aircraft creation...")
        
        # Check aircraft status immediately after creation
        try:
            aircraft_states_after_creation = self.bluesky.get_aircraft_states()
            log.info(f"Aircraft after creation: {len(aircraft_states_after_creation)} aircraft found")
            for callsign in aircraft_states_after_creation.keys():
                log.info(f"  - {callsign}")
        except Exception as e:
            log.error(f"Error checking aircraft after creation: {e}")
        
        time.sleep(3.0)  # Wait for BlueSky to process aircraft creation
        
        # Check aircraft status after wait
        try:
            aircraft_states_after_wait = self.bluesky.get_aircraft_states()
            log.info(f"Aircraft after wait: {len(aircraft_states_after_wait)} aircraft found")
            for callsign in aircraft_states_after_wait.keys():
                log.info(f"  - {callsign}")
        except Exception as e:
            log.error(f"Error checking aircraft after wait: {e}")
        
        # Step 3: Add waypoints to all successfully created aircraft (now safe after RESET fix)
        log.info("Adding waypoints to aircraft...")
        for callsign, actype, lat, lon, hdg, alt, spd, route in aircraft_to_create:
            if route:  # Only add waypoints if route exists
                try:
                    success = self.bluesky.add_waypoints_from_route(callsign, route, alt)
                    if success:
                        log.info(f"Added route with {len(route)} waypoints to {callsign}")
                    else:
                        log.warning(f"Failed to add waypoints to {callsign}")
                except Exception as e:
                    log.error(f"Error adding waypoints to {callsign}: {e}")
        
        # Small step to settle the FMS after waypoint addition
        log.info("Small step to settle FMS after waypoint addition...")
        self.bluesky.step_minutes(0.1)
        
        # Step 4: Final initialization step and verification
        log.info("Final initialization and aircraft verification...")
        
        # Check aircraft status before simulation step
        try:
            aircraft_states_before = self.bluesky.get_aircraft_states()
            log.info(f"Aircraft before sim step: {len(aircraft_states_before)} aircraft found")
            for callsign in aircraft_states_before.keys():
                log.info(f"  - {callsign}")
        except Exception as e:
            log.error(f"Error checking aircraft before sim step: {e}")
        
        # Skip simulation step during initialization to preserve aircraft
        # self.bluesky.step_minutes(0.5)  # Prime the simulation
        
        # Verify aircraft were created successfully
        try:
            aircraft_states = self.bluesky.get_aircraft_states()
            log.info(f"Aircraft verification: {len(aircraft_states)} aircraft found in BlueSky:")
            for callsign in aircraft_states.keys():
                log.info(f"  - {callsign}")
        except Exception as e:
            log.warning(f"Could not verify aircraft creation: {e}")
    
    def _run_simulation_loop(self, duration_min: Optional[float]):
        """Run the main simulation loop with LLM conflict resolution."""
        simulation_time = 0.0
        max_duration = duration_min or 120.0  # Default 2 hours
        
        log.info(f"Starting simulation loop (dt={self.dt_min} min, max_duration={max_duration} min)")
        
        while simulation_time < max_duration:
            # Step BlueSky simulation by dt_min
            if not self.bluesky.step_minutes(self.dt_min):
                log.warning("BlueSky step failed")
                break
            
            simulation_time += self.dt_min
            
            # Get current aircraft states
            aircraft_states = self.bluesky.get_aircraft_states()
            
            if not aircraft_states:
                log.warning("No aircraft states available")
                continue
            
            # Convert to AircraftState objects
            current_states = []
            for callsign, state_dict in aircraft_states.items():
                aircraft_state = self._convert_to_aircraft_state(state_dict, callsign)
                current_states.append(aircraft_state)
            
            # Separate ownship from traffic
            ownship_state = None
            traffic_states = []
            
            for state in current_states:
                if state.aircraft_id == self.ownship_callsign:
                    ownship_state = state
                else:
                    traffic_states.append(state)
            
            if not ownship_state:
                log.warning(f"Ownship {self.ownship_callsign} not found in aircraft states")
                continue
            
            # Detect conflicts
            conflicts = predict_conflicts(ownship_state, traffic_states)
            
            if conflicts:
                log.info(f"Detected {len(conflicts)} conflicts at t={simulation_time:.1f} min")
                
                for conflict in conflicts:
                    # Record conflict
                    self.simulation_results["conflicts"].append({
                        "time_min": simulation_time,
                        "ownship": conflict.ownship_id,
                        "intruder": conflict.intruder_id,
                        "distance_nm": conflict.distance_at_cpa_nm,
                        "time_to_cpa_min": conflict.time_to_cpa_min,
                        "severity": conflict.severity_score
                    })
                    
                    # Get LLM resolution
                    try:
                        resolution = self.llm_client.resolve_conflict(conflict, current_states, self.llm_config)
                        
                        if resolution:
                            # Get aircraft states for resolution execution
                            ownship_state = next((s for s in current_states if s.aircraft_id == conflict.ownship_id), None)
                            intruder_state = next((s for s in current_states if s.aircraft_id == conflict.intruder_id), None)
                            
                            if ownship_state and intruder_state:
                                # Execute resolution with proper parameters
                                resolution_cmd = execute_resolution(resolution, ownship_state, intruder_state, conflict)
                                success = resolution_cmd is not None
                                
                                if success and resolution_cmd:
                                    # Apply the resolution command to BlueSky
                                    success = self.bluesky.execute_command(resolution_cmd)
                            else:
                                logger.error(f"Could not find aircraft states for conflict {conflict.ownship_id} vs {conflict.intruder_id}")
                                success = False
                            
                            # Record resolution
                            resolution_type_str = resolution.resolution_type
                            if hasattr(resolution.resolution_type, 'value'):
                                resolution_type_str = resolution.resolution_type.value
                            
                            self.simulation_results["resolutions"].append({
                                "time_min": simulation_time,
                                "conflict_id": f"{conflict.ownship_id}_{conflict.intruder_id}",
                                "resolution_type": resolution_type_str,
                                "target_aircraft": resolution.target_aircraft,
                                "new_heading": resolution.new_heading_deg,
                                "new_altitude": resolution.new_altitude_ft,
                                "success": success
                            })
                            
                            if success:
                                log.info(f"Applied {resolution_type_str} to {resolution.target_aircraft}")
                            else:
                                log.warning(f"Failed to apply resolution to {resolution.target_aircraft}")
                        else:
                            log.warning(f"No resolution generated for conflict {conflict.ownship_id}_{conflict.intruder_id}")
                            
                    except Exception as e:
                        log.error(f"LLM resolution failed: {e}")
            
            # Log progress
            if int(simulation_time) % 10 == 0:  # Every 10 minutes
                log.info(f"Simulation time: {simulation_time:.1f} min")
            
            # Check for early termination conditions
            if len(aircraft_states) < 2:
                log.info("Simulation ended - insufficient aircraft")
                break
    
    def _convert_to_aircraft_state(self, state_dict: Dict[str, Any], callsign: str) -> AircraftState:
        """Convert BlueSky state dict to AircraftState object."""
        return AircraftState(
            aircraft_id=callsign,
            timestamp=datetime.now(),  # BlueSky doesn't provide timestamps
            latitude=float(state_dict.get("lat", 0)),
            longitude=float(state_dict.get("lon", 0)),
            altitude_ft=float(state_dict.get("alt_ft", 0)),
            ground_speed_kt=float(state_dict.get("spd_kt", 0)),
            heading_deg=float(state_dict.get("hdg_deg", 0)),
            vertical_speed_fpm=float(state_dict.get("vs_fpm", 0))
        )
    
    def save_results(self, output_file: str):
        """Save simulation results to file."""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.simulation_results, f, indent=2)
        
        log.info(f"Saved simulation results: {output_path}")


def main():
    """Main entry point for SCAT LLM runner."""
    parser = argparse.ArgumentParser(description='Run LLM simulation with SCAT data')
    parser.add_argument('--root', required=True, help='Root directory containing SCAT files')
    parser.add_argument('--ownship', required=True, help='Ownship file or ID')
    parser.add_argument('--intruders', default='auto', help='Intruders: "auto" or specific file')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time simulation')
    parser.add_argument('--dt-min', type=float, default=1.0, help='Time step in minutes (default: 1.0)')
    parser.add_argument('--duration', type=float, help='Maximum simulation duration in minutes')
    parser.add_argument('--output', help='Output file for results (default: auto-generated)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create runner
        runner = SCATLLMRunner(args.root, args.realtime, args.dt_min)
        
        # Run simulation
        results = runner.run_simulation(args.ownship, args.intruders, args.duration)
        
        # Generate output filename if not provided
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ownship_name = Path(args.ownship).stem if args.ownship.endswith('.json') else args.ownship
            mode = "realtime" if args.realtime else "fast"
            output_file = f"scat_llm_results_{ownship_name}_{mode}_{timestamp}.json"
        
        # Save results
        runner.save_results(output_file)
        
        # Print summary
        print(f"[OK] LLM simulation completed successfully!")
        print(f"   Duration: {results.get('duration_min', 0):.1f} minutes")
        print(f"   Conflicts detected: {len(results.get('conflicts', []))}")
        print(f"   Resolutions applied: {len(results.get('resolutions', []))}")
        print(f"   Real-time mode: {args.realtime}")
        print(f"   Results saved: {output_file}")
        
    except Exception as e:
        log.error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
