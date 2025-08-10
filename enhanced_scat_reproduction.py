#!/usr/bin/env python3
"""
Enhanced BlueSky SCAT Flight Reproduction with Dynamic Intruder Scenarios

This script extends the basic SCAT reproduction to include time-triggered
intruder spawning with three conflict scenarios:
1. Head-on encounter
2. Crossing traffic  
3. Overtaking aircraft

Intruders are spawned at calculated times and positions to create realistic
conflict scenarios within a 10-minute horizon.
"""

import json
import logging
import sys
import time
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

# Import our BlueSky interface
from src.cdr.bluesky_io import BlueSkyClient, BSConfig
from src.cdr.scat_adapter import SCATAdapter, VicinityIndex
from src.cdr.schemas import AircraftState

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

@dataclass
class IntruderScenario:
    """Configuration for a time-triggered intruder aircraft."""
    
    # Identification
    aircraft_id: str
    scenario_type: str  # "head-on", "crossing", "overtake"
    
    # Timing
    spawn_time_min: float  # Minutes after ownship start
    
    # Initial position and state
    initial_lat: float
    initial_lon: float
    initial_alt_ft: float
    initial_hdg_deg: float
    initial_spd_kt: float
    
    # Target encounter parameters
    encounter_time_min: float  # Expected encounter time
    encounter_lat: float       # Expected encounter position
    encounter_lon: float
    
    # Metadata
    description: str
    expected_conflict: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SpawnEvent:
    """Record of an intruder spawn event."""
    
    timestamp: float
    sim_time_min: float
    aircraft_id: str
    scenario_type: str
    position: Dict[str, float]
    success: bool
    description: str


class EnhancedScatBlueSkyReproduction:
    """Enhanced SCAT flight reproduction with dynamic intruder scenarios."""
    
    def __init__(self, scat_data_dir: str, output_dir: str = "Output/enhanced_demo"):
        self.scat_data_dir = Path(scat_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # BlueSky configuration for enhanced scenarios
        self.bs_config = BSConfig(
            headless=True,
            asas_enabled=False,      # ASAS OFF for baseline
            cdmethod="GEOMETRIC",    # CDMETHOD=GEOMETRIC
            dtlook_sec=600.0,        # DTLOOK=600s (10 min horizon)
            dtmult=10.0,             # DTMULT=10
            realtime=False           # Fast simulation
        )
        
        self.bluesky_client: Optional[BlueSkyClient] = None
        self.scat_adapter: Optional[SCATAdapter] = None
        self.flight_data: Dict[str, List[AircraftState]] = {}
        
        # Enhanced scenario tracking
        self.intruder_scenarios: List[IntruderScenario] = []
        self.spawn_events: List[SpawnEvent] = []
        self.spawned_aircraft: Dict[str, float] = {}  # aircraft_id -> spawn_time
        self.run_log: List[Dict[str, Any]] = []
        
        # Conflict detection data
        self.min_sep_history: List[Dict[str, Any]] = []
        self.cd_outputs: List[Dict[str, Any]] = []
        self.predicted_conflicts: List[Dict[str, Any]] = []
    
    def calculate_encounter_position(self, ownship_states: List[AircraftState], 
                                   encounter_time_min: float) -> Tuple[float, float]:
        """Calculate where ownship will be at encounter time."""
        if not ownship_states:
            return 0.0, 0.0
        
        # Simple linear interpolation for encounter position
        total_duration = len(ownship_states) * 5.0 / 60.0  # Assume 5-second intervals
        time_fraction = min(encounter_time_min / total_duration, 1.0)
        
        state_index = min(int(time_fraction * len(ownship_states)), len(ownship_states) - 1)
        encounter_state = ownship_states[state_index]
        
        return encounter_state.latitude, encounter_state.longitude
    
    def calculate_intruder_start_position(self, encounter_lat: float, encounter_lon: float,
                                        scenario_type: str, ownship_hdg: float) -> Tuple[float, float, float]:
        """Calculate intruder starting position and heading for conflict scenario."""
        
        # Distance calculations (approximate, in degrees)
        distance_nm = 15.0  # Start 15 NM away
        distance_deg = distance_nm / 60.0  # Rough conversion
        
        if scenario_type == "head-on":
            # Position intruder directly ahead on reciprocal heading
            opposite_hdg = (ownship_hdg + 180) % 360
            lat_offset = distance_deg * math.cos(math.radians(ownship_hdg))
            lon_offset = distance_deg * math.sin(math.radians(ownship_hdg))
            
            start_lat = encounter_lat + lat_offset
            start_lon = encounter_lon + lon_offset
            intruder_hdg = opposite_hdg
            
        elif scenario_type == "crossing":
            # Position intruder to cross from the right at 90 degrees
            crossing_hdg = (ownship_hdg - 90) % 360
            lat_offset = distance_deg * math.cos(math.radians(crossing_hdg))
            lon_offset = distance_deg * math.sin(math.radians(crossing_hdg))
            
            start_lat = encounter_lat + lat_offset
            start_lon = encounter_lon + lon_offset
            intruder_hdg = (ownship_hdg + 90) % 360  # Cross from right to left
            
        else:  # overtake
            # Position intruder behind and slightly to the side
            overtake_hdg = (ownship_hdg + 180 + 15) % 360  # Behind and 15 degrees offset
            lat_offset = distance_deg * math.cos(math.radians(overtake_hdg))
            lon_offset = distance_deg * math.sin(math.radians(overtake_hdg))
            
            start_lat = encounter_lat + lat_offset
            start_lon = encounter_lon + lon_offset
            intruder_hdg = ownship_hdg  # Same direction but faster
        
        return start_lat, start_lon, intruder_hdg
    
    def create_intruder_scenarios(self, ownship_id: str) -> List[IntruderScenario]:
        """Create three intruder scenarios for conflict testing."""
        
        ownship_states = self.flight_data.get(ownship_id, [])
        if not ownship_states:
            log.error(f"No flight data for {ownship_id}")
            return []
        
        scenarios = []
        
        # Get ownship initial heading
        ownship_hdg = ownship_states[0].heading_deg
        base_alt = ownship_states[0].altitude_ft
        
        # Scenario 1: Head-on encounter at 3 minutes
        encounter_time_1 = 3.0
        encounter_lat_1, encounter_lon_1 = self.calculate_encounter_position(ownship_states, encounter_time_1)
        start_lat_1, start_lon_1, hdg_1 = self.calculate_intruder_start_position(
            encounter_lat_1, encounter_lon_1, "head-on", ownship_hdg
        )
        
        scenario_1 = IntruderScenario(
            aircraft_id="INTRUDER1",
            scenario_type="head-on",
            spawn_time_min=1.0,  # Spawn 1 minute after start
            initial_lat=start_lat_1,
            initial_lon=start_lon_1,
            initial_alt_ft=base_alt,  # Same altitude for conflict
            initial_hdg_deg=hdg_1,
            initial_spd_kt=420.0,  # Faster for head-on approach
            encounter_time_min=encounter_time_1,
            encounter_lat=encounter_lat_1,
            encounter_lon=encounter_lon_1,
            description="Head-on encounter - opposing traffic at same altitude"
        )
        scenarios.append(scenario_1)
        
        # Scenario 2: Crossing traffic at 5 minutes
        encounter_time_2 = 5.0
        encounter_lat_2, encounter_lon_2 = self.calculate_encounter_position(ownship_states, encounter_time_2)
        start_lat_2, start_lon_2, hdg_2 = self.calculate_intruder_start_position(
            encounter_lat_2, encounter_lon_2, "crossing", ownship_hdg
        )
        
        scenario_2 = IntruderScenario(
            aircraft_id="INTRUDER2",
            scenario_type="crossing", 
            spawn_time_min=2.5,  # Spawn 2.5 minutes after start
            initial_lat=start_lat_2,
            initial_lon=start_lon_2,
            initial_alt_ft=base_alt + 500,  # 500 ft above for vertical conflict
            initial_hdg_deg=hdg_2,
            initial_spd_kt=350.0,
            encounter_time_min=encounter_time_2,
            encounter_lat=encounter_lat_2,
            encounter_lon=encounter_lon_2,
            description="Crossing traffic - perpendicular approach from right"
        )
        scenarios.append(scenario_2)
        
        # Scenario 3: Overtaking aircraft at 7 minutes
        encounter_time_3 = 7.0
        encounter_lat_3, encounter_lon_3 = self.calculate_encounter_position(ownship_states, encounter_time_3)
        start_lat_3, start_lon_3, hdg_3 = self.calculate_intruder_start_position(
            encounter_lat_3, encounter_lon_3, "overtake", ownship_hdg
        )
        
        scenario_3 = IntruderScenario(
            aircraft_id="INTRUDER3",
            scenario_type="overtake",
            spawn_time_min=4.0,  # Spawn 4 minutes after start
            initial_lat=start_lat_3,
            initial_lon=start_lon_3, 
            initial_alt_ft=base_alt - 200,  # 200 ft below
            initial_hdg_deg=hdg_3,
            initial_spd_kt=480.0,  # Much faster for overtaking
            encounter_time_min=encounter_time_3,
            encounter_lat=encounter_lat_3,
            encounter_lon=encounter_lon_3,
            description="Overtaking aircraft - faster traffic from behind"
        )
        scenarios.append(scenario_3)
        
        log.info(f"Created {len(scenarios)} intruder scenarios")
        for scenario in scenarios:
            log.info(f"  {scenario.aircraft_id} ({scenario.scenario_type}): spawn at {scenario.spawn_time_min:.1f}min, encounter at {scenario.encounter_time_min:.1f}min")
        
        return scenarios
    
    def setup_bluesky(self) -> bool:
        """Initialize BlueSky client and configure baseline settings."""
        log.info("Setting up BlueSky client for enhanced SCAT reproduction...")
        
        try:
            self.bluesky_client = BlueSkyClient(self.bs_config)
            
            # Connect to BlueSky
            if not self.bluesky_client.connect():
                log.error("Failed to connect to BlueSky")
                return False
            
            # Reset simulation to clean state
            if not self.bluesky_client.sim_reset():
                log.warning("Failed to reset simulation")
            
            # Configure baseline settings as required
            log.info("Configuring BlueSky enhanced scenario settings:")
            
            # Apply configuration in required order
            if not self.bluesky_client.set_asas(False):
                log.error("Failed to set ASAS OFF")
                return False
            log.info("✅ ASAS OFF")
            
            if not self.bluesky_client.set_cdmethod("GEOMETRIC"):
                log.error("Failed to set CDMETHOD GEOMETRIC")
                return False
            log.info("✅ CDMETHOD GEOMETRIC")
            
            if not self.bluesky_client.set_dtlook(600.0):
                log.error("Failed to set DTLOOK 600")
                return False
            log.info("✅ DTLOOK 600s (10-minute conflict horizon)")
            
            if not self.bluesky_client.set_dtmult(10.0):
                log.error("Failed to set DTMULT 10")
                return False
            log.info("✅ DTMULT 10")
            
            return True
            
        except Exception as e:
            log.exception(f"Failed to setup BlueSky: {e}")
            return False
    
    def load_scat_flight(self, aircraft_id: str) -> bool:
        """Load SCAT flight data for the specified aircraft."""
        log.info(f"Loading SCAT flight data for {aircraft_id}...")
        
        try:
            # Initialize SCAT adapter with dataset path
            self.scat_adapter = SCATAdapter(str(self.scat_data_dir))
            
            # Load scenario with all flights to find our target
            all_states = self.scat_adapter.load_scenario(
                max_flights=100,  # Load enough to ensure we get our target
                time_window_minutes=0  # No time filtering
            )
            
            if not all_states:
                log.error("No SCAT data loaded")
                return False
            
            # Find our target aircraft
            target_states = []
            for state in all_states:
                if state.aircraft_id == aircraft_id:
                    target_states.append(state)
            
            if not target_states:
                log.error(f"Aircraft {aircraft_id} not found in SCAT data")
                # Show available aircraft for debugging
                available_aircraft = list(set(s.aircraft_id for s in all_states[:50]))
                log.info(f"Available aircraft: {available_aircraft}")
                return False
            
            # Sort by timestamp  
            target_states.sort(key=lambda s: s.timestamp)
            self.flight_data[aircraft_id] = target_states
            
            log.info(f"Loaded {len(target_states)} states for {aircraft_id}")
            
            # Calculate flight duration in seconds
            start_time = target_states[0].timestamp
            end_time = target_states[-1].timestamp
            
            # Handle both datetime and float timestamps
            if hasattr(start_time, 'timestamp'):  # datetime object
                duration_sec = (end_time - start_time).total_seconds()
            else:  # assume float timestamp
                duration_sec = end_time - start_time
                
            log.info(f"Flight duration: {duration_sec:.1f} seconds")
            
            return True
            
        except Exception as e:
            log.exception(f"Failed to load SCAT flight: {e}")
            return False
    
    def create_ownship(self, aircraft_id: str) -> bool:
        """Create ownship aircraft in BlueSky using first SCAT state."""
        states = self.flight_data.get(aircraft_id, [])
        if not states:
            log.error(f"No flight data for {aircraft_id}")
            return False
        
        first_state = states[0]
        
        # Create aircraft using CRE command
        success = self.bluesky_client.create_aircraft(
            cs=aircraft_id,
            actype="B738",  # Default aircraft type
            lat=first_state.latitude,
            lon=first_state.longitude,
            hdg_deg=first_state.heading_deg,
            alt_ft=first_state.altitude_ft,
            spd_kt=first_state.ground_speed_kt
        )
        
        if success:
            log.info(f"✅ CRE {aircraft_id} at ({first_state.latitude:.6f}, {first_state.longitude:.6f})")
            
            # Log ownship creation event
            self.run_log.append({
                'timestamp': time.time(),
                'sim_time_min': 0.0,
                'event_type': 'ownship_created',
                'aircraft_id': aircraft_id,
                'position': {
                    'lat': first_state.latitude,
                    'lon': first_state.longitude,
                    'alt_ft': first_state.altitude_ft,
                    'hdg_deg': first_state.heading_deg,
                    'spd_kt': first_state.ground_speed_kt
                }
            })
        else:
            log.error(f"Failed to create aircraft {aircraft_id}")
        
        return success
    
    def add_waypoints(self, aircraft_id: str, waypoint_spacing: int = 10) -> bool:
        """Add waypoints from SCAT track using ADDWPT commands."""
        states = self.flight_data.get(aircraft_id, [])
        if not states:
            log.error(f"No flight data for {aircraft_id}")
            return False
        
        log.info(f"Adding waypoints for {aircraft_id} (every {waypoint_spacing} states)...")
        
        waypoint_count = 0
        for i in range(1, len(states), waypoint_spacing):  # Skip first state (already at start position)
            state = states[i]
            
            # Add waypoint using ADDWPT command
            success = self.bluesky_client.add_waypoint(
                cs=aircraft_id,
                lat=state.latitude,
                lon=state.longitude,
                alt_ft=state.altitude_ft
            )
            
            if success:
                waypoint_count += 1
                if waypoint_count <= 5:  # Log first few for verification
                    log.info(f"✅ ADDWPT {aircraft_id} waypoint {waypoint_count} at ({state.latitude:.6f}, {state.longitude:.6f})")
            else:
                log.warning(f"Failed to add waypoint {waypoint_count + 1}")
        
        log.info(f"Added {waypoint_count} waypoints for {aircraft_id}")
        return waypoint_count > 0
    
    def spawn_intruder(self, scenario: IntruderScenario, sim_time_min: float) -> bool:
        """Spawn an intruder aircraft according to scenario parameters."""
        
        if not self.bluesky_client:
            log.error("BlueSky client not available for intruder spawn")
            return False
        
        log.info(f"Spawning {scenario.aircraft_id} ({scenario.scenario_type}) at sim_time={sim_time_min:.2f}min")
        
        # Create intruder aircraft
        success = self.bluesky_client.create_aircraft(
            cs=scenario.aircraft_id,
            actype="A320",  # Different type for intruders
            lat=scenario.initial_lat,
            lon=scenario.initial_lon,
            hdg_deg=scenario.initial_hdg_deg,
            alt_ft=scenario.initial_alt_ft,
            spd_kt=scenario.initial_spd_kt
        )
        
        # Record spawn event
        spawn_event = SpawnEvent(
            timestamp=time.time(),
            sim_time_min=sim_time_min,
            aircraft_id=scenario.aircraft_id,
            scenario_type=scenario.scenario_type,
            position={
                'lat': scenario.initial_lat,
                'lon': scenario.initial_lon,
                'alt_ft': scenario.initial_alt_ft,
                'hdg_deg': scenario.initial_hdg_deg,
                'spd_kt': scenario.initial_spd_kt
            },
            success=success,
            description=scenario.description
        )
        self.spawn_events.append(spawn_event)
        
        if success:
            self.spawned_aircraft[scenario.aircraft_id] = sim_time_min
            log.info(f"✅ Successfully spawned {scenario.aircraft_id} at ({scenario.initial_lat:.6f}, {scenario.initial_lon:.6f})")
            
            # Add to run log
            self.run_log.append({
                'timestamp': time.time(),
                'sim_time_min': sim_time_min,
                'event_type': 'intruder_spawned',
                'aircraft_id': scenario.aircraft_id,
                'scenario_type': scenario.scenario_type,
                'position': spawn_event.position,
                'description': scenario.description
            })
        else:
            log.error(f"Failed to spawn {scenario.aircraft_id}")
        
        return success
    
    def detect_conflicts(self, sim_time_min: float) -> List[Dict[str, Any]]:
        """Detect potential conflicts using BlueSky conflict detection."""
        
        if not self.bluesky_client:
            return []
        
        # Get current aircraft states
        try:
            aircraft_states = self.bluesky_client.get_aircraft_states()
            
            conflicts = []
            
            # Simple conflict detection: check distances between all aircraft pairs
            aircraft_list = list(aircraft_states.keys())
            
            for i in range(len(aircraft_list)):
                for j in range(i + 1, len(aircraft_list)):
                    ac1_id = aircraft_list[i]
                    ac2_id = aircraft_list[j]
                    
                    ac1_state = aircraft_states[ac1_id]
                    ac2_state = aircraft_states[ac2_id]
                    
                    # Calculate horizontal distance (simple great circle approximation)
                    lat1, lon1 = math.radians(ac1_state['lat']), math.radians(ac1_state['lon'])
                    lat2, lon2 = math.radians(ac2_state['lat']), math.radians(ac2_state['lon'])
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    c = 2 * math.asin(math.sqrt(a))
                    distance_nm = 3440.065 * c  # Earth radius in NM
                    
                    # Calculate vertical separation
                    alt_diff_ft = abs(ac1_state['alt_ft'] - ac2_state['alt_ft'])
                    
                    # Check for conflict (within 5 NM and 1000 ft)
                    if distance_nm < 5.0 and alt_diff_ft < 1000.0:
                        conflict = {
                            'sim_time_min': sim_time_min,
                            'aircraft_1': ac1_id,
                            'aircraft_2': ac2_id,
                            'horizontal_distance_nm': distance_nm,
                            'vertical_separation_ft': alt_diff_ft,
                            'conflict_severity': 'high' if distance_nm < 2.0 else 'medium'
                        }
                        conflicts.append(conflict)
                        
                        # Log conflict detection
                        log.warning(f"CONFLICT DETECTED at {sim_time_min:.2f}min: {ac1_id} vs {ac2_id}, "
                                  f"distance={distance_nm:.2f}NM, alt_diff={alt_diff_ft:.0f}ft")
            
            return conflicts
            
        except Exception as e:
            log.debug(f"Conflict detection failed: {e}")
            return []
    
    def run_enhanced_simulation(self, aircraft_id: str, max_duration_min: float = 15.0) -> bool:
        """Run enhanced BlueSky simulation with time-triggered intruder spawning."""
        log.info(f"Starting enhanced simulation for {aircraft_id} (max {max_duration_min:.1f}min)...")
        
        if not self.bluesky_client:
            log.error("BlueSky client not available")
            return False
        
        # Create intruder scenarios
        self.intruder_scenarios = self.create_intruder_scenarios(aircraft_id)
        
        start_time = time.time()
        sim_time_min = 0.0
        step_count = 0
        
        try:
            # Enable LNAV for ownship to follow waypoints
            self.bluesky_client.stack(f"{aircraft_id} LNAV ON")
            
            while sim_time_min < max_duration_min:
                # Step simulation forward
                if not self.bluesky_client.stack("STEP"):
                    log.warning("Failed to step simulation")
                    break
                
                # Small time step
                sim_time_min += 0.1 / 60.0  # 0.1 second = 0.0167 minutes
                
                # Check for intruder spawns every second (every 10 steps)
                if step_count % 10 == 0:
                    for scenario in self.intruder_scenarios:
                        if (scenario.aircraft_id not in self.spawned_aircraft and 
                            sim_time_min >= scenario.spawn_time_min):
                            self.spawn_intruder(scenario, sim_time_min)
                
                # Collect data every 10 seconds (every 100 steps)
                if step_count % 100 == 0:
                    # Get current aircraft states
                    current_states = self.bluesky_client.get_aircraft_states()
                    
                    # Record min-sep data for all aircraft
                    for ac_id, state in current_states.items():
                        min_sep_data = {
                            'timestamp': time.time(),
                            'sim_time_min': sim_time_min,
                            'aircraft_id': ac_id,
                            'lat_deg': state['lat'],
                            'lon_deg': state['lon'],
                            'alt_ft': state['alt_ft'],
                            'hdg_deg': state['hdg_deg'],
                            'spd_kt': state['spd_kt'],
                            'is_ownship': ac_id == aircraft_id,
                            'is_intruder': ac_id in self.spawned_aircraft
                        }
                        self.min_sep_history.append(min_sep_data)
                    
                    # Detect conflicts
                    conflicts = self.detect_conflicts(sim_time_min)
                    self.predicted_conflicts.extend(conflicts)
                    
                    # Record CD outputs
                    cd_data = {
                        'timestamp': time.time(),
                        'sim_time_min': sim_time_min,
                        'cd_method': 'GEOMETRIC',
                        'dtlook_sec': 600.0,
                        'active_aircraft': list(current_states.keys()),
                        'conflicts_detected': conflicts,
                        'asas_active': False
                    }
                    self.cd_outputs.append(cd_data)
                
                step_count += 1
                
                # Log progress every minute of sim time
                if step_count % 600 == 0:  # 600 steps = 1 minute
                    current_states = self.bluesky_client.get_aircraft_states()
                    log.info(f"Simulation progress: {sim_time_min:.1f}min, active aircraft: {len(current_states)}")
                
                # Check if ownship is still active
                if step_count % 100 == 0:
                    current_states = self.bluesky_client.get_aircraft_states()
                    if aircraft_id.upper() not in current_states:
                        log.info(f"Ownship {aircraft_id} no longer active at {sim_time_min:.1f}min")
                        if sim_time_min < 2.0:  # If it disappeared too quickly
                            log.warning("Ownship disappeared very quickly - continuing with intruders only")
            
            elapsed = time.time() - start_time
            log.info(f"Enhanced simulation completed: {sim_time_min:.1f}min simulated in {elapsed:.1f}s real time")
            log.info(f"Spawned {len(self.spawned_aircraft)} intruders")
            log.info(f"Detected {len(self.predicted_conflicts)} potential conflicts")
            log.info(f"Collected {len(self.min_sep_history)} state records")
            
            return True
            
        except Exception as e:
            log.exception(f"Enhanced simulation failed: {e}")
            return False
    
    def save_enhanced_outputs(self, aircraft_id: str) -> bool:
        """Save enhanced outputs including scenarios.json and run logs."""
        log.info(f"Saving enhanced outputs for {aircraft_id}...")
        
        try:
            # Save scenarios.json
            scenarios_data = {
                'ownship_id': aircraft_id,
                'generation_time': datetime.now(timezone.utc).isoformat(),
                'simulation_config': {
                    'asas_enabled': False,
                    'cdmethod': 'GEOMETRIC',
                    'dtlook_sec': 600.0,
                    'dtmult': 10.0
                },
                'intruder_scenarios': [scenario.to_dict() for scenario in self.intruder_scenarios],
                'scenario_summary': {
                    'total_intruders': len(self.intruder_scenarios),
                    'scenario_types': list(set(s.scenario_type for s in self.intruder_scenarios)),
                    'spawn_times_min': [s.spawn_time_min for s in self.intruder_scenarios],
                    'encounter_times_min': [s.encounter_time_min for s in self.intruder_scenarios]
                }
            }
            
            scenarios_file = self.output_dir / "scenarios.json"
            with open(scenarios_file, 'w') as f:
                json.dump(scenarios_data, f, indent=2)
            log.info(f"✅ Saved scenarios configuration to {scenarios_file}")
            
            # Save run log
            run_log_data = {
                'ownship_id': aircraft_id,
                'simulation_start': datetime.now(timezone.utc).isoformat(),
                'events': self.run_log,
                'spawn_events': [asdict(event) for event in self.spawn_events],
                'performance_summary': {
                    'total_events': len(self.run_log),
                    'successful_spawns': len([e for e in self.spawn_events if e.success]),
                    'failed_spawns': len([e for e in self.spawn_events if not e.success]),
                    'conflicts_detected': len(self.predicted_conflicts),
                    'state_records': len(self.min_sep_history)
                }
            }
            
            run_log_file = self.output_dir / "run_log.json"
            with open(run_log_file, 'w') as f:
                json.dump(run_log_data, f, indent=2)
            log.info(f"✅ Saved run log to {run_log_file}")
            
            # Save min-sep time series
            min_sep_file = self.output_dir / f"enhanced_{aircraft_id}_min_sep.jsonl"
            with open(min_sep_file, 'w') as f:
                for record in self.min_sep_history:
                    f.write(json.dumps(record) + '\n')
            log.info(f"✅ Saved {len(self.min_sep_history)} min-sep records to {min_sep_file}")
            
            # Save CD outputs with conflicts
            cd_file = self.output_dir / f"enhanced_{aircraft_id}_cd_outputs.jsonl"
            with open(cd_file, 'w') as f:
                for record in self.cd_outputs:
                    f.write(json.dumps(record) + '\n')
            log.info(f"✅ Saved {len(self.cd_outputs)} CD output records to {cd_file}")
            
            # Save conflict analysis
            conflict_analysis = {
                'analysis_time': datetime.now(timezone.utc).isoformat(),
                'ownship_id': aircraft_id,
                'total_conflicts': len(self.predicted_conflicts),
                'conflicts_by_type': {},
                'conflicts_in_horizon': [c for c in self.predicted_conflicts if c['sim_time_min'] <= 10.0],
                'detailed_conflicts': self.predicted_conflicts
            }
            
            # Group conflicts by intruder scenario type
            for conflict in self.predicted_conflicts:
                for ac_name in [conflict['aircraft_1'], conflict['aircraft_2']]:
                    if ac_name.startswith('INTRUDER'):
                        # Find scenario type
                        scenario_type = 'unknown'
                        for scenario in self.intruder_scenarios:
                            if scenario.aircraft_id == ac_name:
                                scenario_type = scenario.scenario_type
                                break
                        
                        if scenario_type not in conflict_analysis['conflicts_by_type']:
                            conflict_analysis['conflicts_by_type'][scenario_type] = 0
                        conflict_analysis['conflicts_by_type'][scenario_type] += 1
                        break
            
            conflicts_file = self.output_dir / f"enhanced_{aircraft_id}_conflicts.json"
            with open(conflicts_file, 'w') as f:
                json.dump(conflict_analysis, f, indent=2)
            log.info(f"✅ Saved conflict analysis to {conflicts_file}")
            
            # Summary
            summary = {
                'aircraft_id': aircraft_id,
                'analysis_time': datetime.now(timezone.utc).isoformat(),
                'simulation_type': 'enhanced_with_intruders',
                'intruder_scenarios': len(self.intruder_scenarios),
                'successful_spawns': len([e for e in self.spawn_events if e.success]),
                'conflicts_detected': len(self.predicted_conflicts),
                'conflicts_in_10min_horizon': len([c for c in self.predicted_conflicts if c['sim_time_min'] <= 10.0]),
                'files_generated': [
                    'scenarios.json',
                    'run_log.json',
                    f'enhanced_{aircraft_id}_min_sep.jsonl',
                    f'enhanced_{aircraft_id}_cd_outputs.jsonl',
                    f'enhanced_{aircraft_id}_conflicts.json'
                ]
            }
            
            summary_file = self.output_dir / f"enhanced_{aircraft_id}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            log.info(f"✅ Saved summary to {summary_file}")
            
            return True
            
        except Exception as e:
            log.exception(f"Failed to save enhanced outputs: {e}")
            return False
    
    def reproduce_enhanced_flight(self, aircraft_id: str) -> bool:
        """Complete enhanced flight reproduction workflow with intruders."""
        log.info(f"Starting enhanced SCAT flight reproduction for {aircraft_id}")
        
        # 1. Setup BlueSky with enhanced configuration
        if not self.setup_bluesky():
            log.error("Failed to setup BlueSky")
            return False
        
        # 2. Load SCAT flight data
        if not self.load_scat_flight(aircraft_id):
            log.error("Failed to load SCAT flight data")
            return False
        
        # 3. Create ownship aircraft
        if not self.create_ownship(aircraft_id):
            log.error("Failed to create ownship")
            return False
        
        # 4. Add flight route waypoints
        if not self.add_waypoints(aircraft_id):
            log.error("Failed to add waypoints")
            return False
        
        # 5. Run enhanced simulation with intruder spawning
        if not self.run_enhanced_simulation(aircraft_id):
            log.error("Failed to run enhanced simulation")
            return False
        
        # 6. Save enhanced outputs
        if not self.save_enhanced_outputs(aircraft_id):
            log.error("Failed to save enhanced outputs")
            return False
        
        log.info(f"✅ Successfully completed enhanced reproduction for {aircraft_id}")
        return True
    
    def cleanup(self):
        """Clean up resources."""
        if self.bluesky_client:
            try:
                self.bluesky_client.close()
            except Exception as e:
                log.debug(f"BlueSky cleanup error: {e}")


def main():
    """Main entry point for enhanced SCAT flight reproduction."""
    parser = argparse.ArgumentParser(description='Enhanced SCAT flight reproduction with intruder scenarios')
    parser.add_argument('scat_data_dir', help='Directory containing SCAT data')
    parser.add_argument('aircraft_id', help='Aircraft ID to reproduce (e.g., SAS117)')
    parser.add_argument('--output-dir', default='Output/enhanced_demo', 
                       help='Output directory for enhanced files')
    
    args = parser.parse_args()
    
    log.info("=== Enhanced BlueSky SCAT Flight Reproduction ===")
    log.info(f"SCAT data: {args.scat_data_dir}")
    log.info(f"Aircraft: {args.aircraft_id}")
    log.info(f"Output: {args.output_dir}")
    log.info("Features: Time-triggered intruder spawning, conflict detection")
    
    reproducer = EnhancedScatBlueSkyReproduction(args.scat_data_dir, args.output_dir)
    
    try:
        success = reproducer.reproduce_enhanced_flight(args.aircraft_id)
        if success:
            log.info("🎉 Enhanced flight reproduction completed successfully!")
            log.info(f"📁 Check {args.output_dir} for scenarios.json and run logs")
            return 0
        else:
            log.error("❌ Enhanced flight reproduction failed")
            return 1
    
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 1
    
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return 1
    
    finally:
        reproducer.cleanup()


if __name__ == "__main__":
    sys.exit(main())
