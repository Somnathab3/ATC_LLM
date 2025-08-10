#!/usr/bin/env python3
"""
Enhanced BlueSky Demo with Time-Triggered Intruder Scenarios

This demonstration script shows the complete enhanced SCAT reproduction system
with time-triggered intruder spawning for conflict testing.
"""

import json
import logging
import time
from pathlib import Path
from src.cdr.bluesky_io import BlueSkyClient, BSConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def demo_enhanced_bluesky_scenarios():
    """Demonstrate enhanced BlueSky with time-triggered intruder scenarios."""
    
    log.info("=== Enhanced BlueSky Demo: Time-Triggered Intruder Scenarios ===")
    
    # Enhanced BlueSky configuration for conflict testing
    bs_config = BSConfig(
        headless=True,
        asas_enabled=False,      # ASAS OFF for baseline
        cdmethod="GEOMETRIC",    # CDMETHOD=GEOMETRIC  
        dtlook_sec=600.0,        # DTLOOK=600s (10-minute horizon)
        dtmult=10.0,             # DTMULT=10
        realtime=False           # Fast simulation
    )
    
    # Create output directory
    output_dir = Path("Output/enhanced_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bluesky_client = None
    
    try:
        # Initialize BlueSky client
        log.info("1. Initializing Enhanced BlueSky Client...")
        bluesky_client = BlueSkyClient(bs_config)
        
        if not bluesky_client.connect():
            log.error("Failed to connect to BlueSky")
            return False
        
        # Reset simulation
        if not bluesky_client.sim_reset():
            log.warning("Failed to reset simulation")
        
        # Configure enhanced baseline settings
        log.info("2. Configuring Enhanced Baseline Settings...")
        
        if not bluesky_client.set_asas(False):
            log.error("Failed to set ASAS OFF")
            return False
        log.info("✅ ASAS OFF: SUCCESS")
        
        if not bluesky_client.set_cdmethod("GEOMETRIC"):
            log.error("Failed to set CDMETHOD GEOMETRIC")
            return False
        log.info("✅ CDMETHOD GEOMETRIC: SUCCESS")
        
        if not bluesky_client.set_dtlook(600.0):
            log.error("Failed to set DTLOOK 600")
            return False
        log.info("✅ DTLOOK 600s (10-minute conflict horizon): SUCCESS")
        
        if not bluesky_client.set_dtmult(10.0):
            log.error("Failed to set DTMULT 10")
            return False
        log.info("✅ DTMULT 10: SUCCESS")
        
        # Create ownship aircraft
        log.info("3. Creating Ownship Aircraft...")
        ownship_success = bluesky_client.create_aircraft(
            cs="OWNSHIP1",
            actype="B738",
            lat=59.6519,      # Oslo area coordinates
            lon=10.7363,
            hdg_deg=90.0,     # Heading east
            alt_ft=35000.0,
            spd_kt=420.0
        )
        
        if ownship_success:
            log.info("✅ CRE OWNSHIP1: SUCCESS")
        else:
            log.error("Failed to create ownship")
            return False
        
        # Add some waypoints for ownship route
        log.info("4. Adding Ownship Route Waypoints...")
        waypoints = [
            (59.6519, 11.0363, 35000),  # Waypoint 1
            (59.6519, 11.5363, 35000),  # Waypoint 2  
            (59.6519, 12.0363, 35000),  # Waypoint 3
        ]
        
        for i, (lat, lon, alt) in enumerate(waypoints, 1):
            wp_success = bluesky_client.add_waypoint("OWNSHIP1", lat, lon, alt)
            if wp_success:
                log.info(f"✅ ADDWPT OWNSHIP1 waypoint {i}: SUCCESS")
            else:
                log.warning(f"Failed to add waypoint {i}")
        
        # Enable LNAV for ownship
        bluesky_client.stack("OWNSHIP1 LNAV ON")
        log.info("✅ OWNSHIP1 LNAV ON: SUCCESS")
        
        # Define intruder scenarios for time-triggered spawning
        log.info("5. Preparing Time-Triggered Intruder Scenarios...")
        
        intruder_scenarios = [
            {
                'aircraft_id': 'INTRUDER1',
                'scenario_type': 'head-on',
                'spawn_time_min': 1.0,
                'lat': 59.6519,
                'lon': 11.8363,  # Ahead of ownship
                'alt_ft': 35000.0,  # Same altitude for conflict
                'hdg_deg': 270.0,   # Opposite direction (west)
                'spd_kt': 450.0,
                'description': 'Head-on encounter - opposing traffic at same altitude'
            },
            {
                'aircraft_id': 'INTRUDER2', 
                'scenario_type': 'crossing',
                'spawn_time_min': 2.5,
                'lat': 59.8519,     # North of route
                'lon': 11.2363,
                'alt_ft': 35500.0,  # 500 ft above
                'hdg_deg': 180.0,   # Heading south to cross
                'spd_kt': 380.0,
                'description': 'Crossing traffic - perpendicular approach from north'
            },
            {
                'aircraft_id': 'INTRUDER3',
                'scenario_type': 'overtake', 
                'spawn_time_min': 4.0,
                'lat': 59.6519,
                'lon': 10.4363,     # Behind ownship
                'alt_ft': 34800.0,  # 200 ft below
                'hdg_deg': 90.0,    # Same direction
                'spd_kt': 500.0,    # Much faster for overtaking
                'description': 'Overtaking aircraft - faster traffic from behind'
            }
        ]
        
        log.info(f"Configured {len(intruder_scenarios)} intruder scenarios:")
        for scenario in intruder_scenarios:
            log.info(f"  {scenario['aircraft_id']} ({scenario['scenario_type']}): "
                    f"spawn at {scenario['spawn_time_min']:.1f}min")
        
        # Run enhanced simulation with time-triggered spawning
        log.info("6. Running Enhanced Simulation with Time-Triggered Intruders...")
        
        spawned_aircraft = {}
        spawn_events = []
        conflict_history = []
        simulation_data = []
        
        sim_time_min = 0.0
        max_duration_min = 12.0
        step_count = 0
        
        start_time = time.time()
        
        while sim_time_min < max_duration_min:
            # Step simulation
            if not bluesky_client.stack("STEP"):
                log.warning("Failed to step simulation")
                break
            
            # Time increment (0.1 second steps)
            sim_time_min += 0.1 / 60.0
            step_count += 1
            
            # Check for intruder spawns every second (every 10 steps)
            if step_count % 10 == 0:
                for scenario in intruder_scenarios:
                    aircraft_id = scenario['aircraft_id']
                    if (aircraft_id not in spawned_aircraft and 
                        sim_time_min >= scenario['spawn_time_min']):
                        
                        log.info(f"Spawning {aircraft_id} ({scenario['scenario_type']}) "
                                f"at {sim_time_min:.2f}min...")
                        
                        # Spawn intruder
                        spawn_success = bluesky_client.create_aircraft(
                            cs=aircraft_id,
                            actype="A320",
                            lat=scenario['lat'],
                            lon=scenario['lon'],
                            hdg_deg=scenario['hdg_deg'],
                            alt_ft=scenario['alt_ft'],
                            spd_kt=scenario['spd_kt']
                        )
                        
                        if spawn_success:
                            spawned_aircraft[aircraft_id] = sim_time_min
                            log.info(f"✅ Successfully spawned {aircraft_id} at "
                                    f"({scenario['lat']:.4f}, {scenario['lon']:.4f})")
                            
                            # Record spawn event
                            spawn_event = {
                                'timestamp': time.time(),
                                'sim_time_min': sim_time_min,
                                'aircraft_id': aircraft_id,
                                'scenario_type': scenario['scenario_type'],
                                'position': {
                                    'lat': scenario['lat'],
                                    'lon': scenario['lon'],
                                    'alt_ft': scenario['alt_ft'],
                                    'hdg_deg': scenario['hdg_deg'],
                                    'spd_kt': scenario['spd_kt']
                                },
                                'success': True,
                                'description': scenario['description']
                            }
                            spawn_events.append(spawn_event)
                        else:
                            log.error(f"Failed to spawn {aircraft_id}")
            
            # Collect simulation data every 10 seconds (every 100 steps)
            if step_count % 100 == 0:
                try:
                    current_states = bluesky_client.get_aircraft_states()
                    
                    # Record states for all aircraft
                    for ac_id, state in current_states.items():
                        sim_data = {
                            'timestamp': time.time(),
                            'sim_time_min': sim_time_min,
                            'aircraft_id': ac_id,
                            'lat_deg': state['lat'],
                            'lon_deg': state['lon'],
                            'alt_ft': state['alt_ft'],
                            'hdg_deg': state['hdg_deg'],
                            'spd_kt': state['spd_kt'],
                            'is_ownship': ac_id == 'OWNSHIP1',
                            'is_intruder': ac_id in spawned_aircraft
                        }
                        simulation_data.append(sim_data)
                    
                    # Simple conflict detection (aircraft within 5 NM)
                    aircraft_list = list(current_states.keys())
                    for i in range(len(aircraft_list)):
                        for j in range(i + 1, len(aircraft_list)):
                            ac1 = aircraft_list[i]
                            ac2 = aircraft_list[j]
                            
                            state1 = current_states[ac1]
                            state2 = current_states[ac2]
                            
                            # Simple distance calculation (lat/lon difference)
                            lat_diff = abs(state1['lat'] - state2['lat'])
                            lon_diff = abs(state1['lon'] - state2['lon'])
                            distance_approx = (lat_diff**2 + lon_diff**2)**0.5 * 60  # Rough NM
                            
                            alt_diff = abs(state1['alt_ft'] - state2['alt_ft'])
                            
                            if distance_approx < 5.0 and alt_diff < 1000.0:
                                conflict = {
                                    'sim_time_min': sim_time_min,
                                    'aircraft_1': ac1,
                                    'aircraft_2': ac2,
                                    'distance_approx_nm': distance_approx,
                                    'altitude_diff_ft': alt_diff,
                                    'severity': 'high' if distance_approx < 2.0 else 'medium'
                                }
                                conflict_history.append(conflict)
                                
                                log.warning(f"🚨 CONFLICT DETECTED at {sim_time_min:.2f}min: "
                                          f"{ac1} vs {ac2}, ~{distance_approx:.1f}NM, "
                                          f"alt_diff={alt_diff:.0f}ft")
                    
                except Exception as e:
                    log.debug(f"Data collection error: {e}")
            
            # Progress logging every minute
            if step_count % 600 == 0:  # 600 steps = 1 minute
                current_states = bluesky_client.get_aircraft_states()
                log.info(f"Simulation progress: {sim_time_min:.1f}min, "
                        f"active aircraft: {len(current_states)}, "
                        f"spawned: {len(spawned_aircraft)}")
        
        elapsed = time.time() - start_time
        log.info(f"Enhanced simulation completed: {sim_time_min:.1f}min simulated in {elapsed:.1f}s")
        
        # Save enhanced outputs
        log.info("7. Saving Enhanced Outputs...")
        
        # Save scenarios.json
        scenarios_data = {
            'generation_time': time.time(),
            'ownship_id': 'OWNSHIP1',
            'simulation_config': {
                'asas_enabled': False,
                'cdmethod': 'GEOMETRIC',
                'dtlook_sec': 600.0,
                'dtmult': 10.0,
                'simulation_duration_min': sim_time_min
            },
            'intruder_scenarios': intruder_scenarios,
            'scenario_summary': {
                'total_intruders': len(intruder_scenarios),
                'successful_spawns': len(spawned_aircraft),
                'scenario_types': [s['scenario_type'] for s in intruder_scenarios],
                'spawn_times_min': [s['spawn_time_min'] for s in intruder_scenarios]
            }
        }
        
        scenarios_file = output_dir / "scenarios.json"
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios_data, f, indent=2)
        log.info(f"✅ Saved scenarios.json to {scenarios_file}")
        
        # Save run log with spawn events
        run_log_data = {
            'simulation_start': time.time(),
            'ownship_id': 'OWNSHIP1',
            'spawn_events': spawn_events,
            'performance_summary': {
                'total_spawn_events': len(spawn_events),
                'successful_spawns': len([e for e in spawn_events if e['success']]),
                'conflicts_detected': len(conflict_history),
                'conflicts_in_10min': len([c for c in conflict_history if c['sim_time_min'] <= 10.0]),
                'simulation_records': len(simulation_data)
            },
            'conflict_summary': {
                'total_conflicts': len(conflict_history),
                'conflicts_by_severity': {
                    'high': len([c for c in conflict_history if c['severity'] == 'high']),
                    'medium': len([c for c in conflict_history if c['severity'] == 'medium'])
                },
                'conflicts_in_horizon': [c for c in conflict_history if c['sim_time_min'] <= 10.0]
            }
        }
        
        run_log_file = output_dir / "run_log.json"
        with open(run_log_file, 'w') as f:
            json.dump(run_log_data, f, indent=2)
        log.info(f"✅ Saved run_log.json to {run_log_file}")
        
        # Save simulation data as JSONL
        sim_data_file = output_dir / "enhanced_demo_min_sep.jsonl"
        with open(sim_data_file, 'w') as f:
            for record in simulation_data:
                f.write(json.dumps(record) + '\n')
        log.info(f"✅ Saved {len(simulation_data)} simulation records to {sim_data_file}")
        
        # Save conflict data as JSONL
        conflicts_file = output_dir / "enhanced_demo_conflicts.jsonl"
        with open(conflicts_file, 'w') as f:
            for record in conflict_history:
                f.write(json.dumps(record) + '\n')
        log.info(f"✅ Saved {len(conflict_history)} conflict records to {conflicts_file}")
        
        # Summary report
        log.info("=== ENHANCED DEMO SUMMARY ===")
        log.info(f"✅ Ownship created and routed successfully")
        log.info(f"✅ {len(spawned_aircraft)} intruders spawned at scheduled times:")
        for aircraft_id, spawn_time in spawned_aircraft.items():
            scenario_type = next(s['scenario_type'] for s in intruder_scenarios 
                               if s['aircraft_id'] == aircraft_id)
            log.info(f"    {aircraft_id} ({scenario_type}) at {spawn_time:.2f}min")
        
        conflicts_in_horizon = [c for c in conflict_history if c['sim_time_min'] <= 10.0]
        log.info(f"✅ {len(conflicts_in_horizon)} conflicts detected within 10-min horizon")
        
        if conflicts_in_horizon:
            log.info("    Conflicts detected:")
            for conflict in conflicts_in_horizon[:3]:  # Show first 3
                log.info(f"    {conflict['aircraft_1']} vs {conflict['aircraft_2']} "
                        f"at {conflict['sim_time_min']:.2f}min "
                        f"(~{conflict['distance_approx_nm']:.1f}NM)")
        
        log.info(f"✅ All outputs saved to {output_dir}/")
        log.info("    - scenarios.json (intruder configurations)")
        log.info("    - run_log.json (spawn events and performance)")
        log.info("    - enhanced_demo_min_sep.jsonl (time series data)")
        log.info("    - enhanced_demo_conflicts.jsonl (conflict records)")
        
        return True
        
    except Exception as e:
        log.exception(f"Enhanced demo failed: {e}")
        return False
        
    finally:
        if bluesky_client:
            try:
                bluesky_client.close()
            except Exception as e:
                log.debug(f"BlueSky cleanup error: {e}")


if __name__ == "__main__":
    success = demo_enhanced_bluesky_scenarios()
    if success:
        print("\n🎉 Enhanced BlueSky demo completed successfully!")
        print("📁 Check Output/enhanced_demo/ for scenarios.json and logs")
        exit(0)
    else:
        print("\n❌ Enhanced BlueSky demo failed")
        exit(1)
