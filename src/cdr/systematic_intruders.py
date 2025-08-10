"""Systematic Intruder Generator for Parameterized Conflict Scenarios.

This module provides a systematic approach to generating reproducible intruder
scenarios with specific conflict patterns (crossing, head-on, overtake) and
controllable CPA timing for comprehensive testing and evaluation.
"""

import logging
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import json

from .geodesy import haversine_nm, bearing_deg, destination_point_nm
from .schemas import AircraftState, IntruderScenario, FlightRecord

logger = logging.getLogger(__name__)

class ConflictPattern(Enum):
    """Types of conflict patterns that can be generated."""
    CROSSING = "crossing"          # Aircraft paths intersect
    HEAD_ON = "head_on"           # Aircraft approaching head-on
    OVERTAKE = "overtake"         # Faster aircraft overtaking slower
    PARALLEL = "parallel"         # Parallel paths with potential convergence
    CONVERGING = "converging"     # Paths converging at an angle
    RANDOM = "random"             # Random encounter (existing behavior)

class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = "low"                   # CPA > 8 minutes, distance 3-5 NM
    MEDIUM = "medium"             # CPA 4-8 minutes, distance 1-3 NM
    HIGH = "high"                 # CPA 2-4 minutes, distance 0.5-1 NM
    CRITICAL = "critical"         # CPA < 2 minutes, distance < 0.5 NM

@dataclass
class ScenarioParameters:
    """Parameters for generating a specific conflict scenario."""
    pattern: ConflictPattern
    severity: ConflictSeverity
    cpa_time_min: float           # Time to CPA in minutes
    cpa_distance_nm: float        # Horizontal separation at CPA
    altitude_difference_ft: float = 0.0  # Vertical separation (0 = same level)
    intruder_speed_kt: float = 450.0     # Intruder ground speed
    spawn_distance_nm: float = 50.0      # Distance from ownship at spawn
    seed: Optional[int] = None            # Reproducibility seed

@dataclass
class SystematicScenarioSet:
    """A systematic set of scenarios for comprehensive testing."""
    name: str
    description: str
    scenarios: List[ScenarioParameters] = field(default_factory=list)
    base_seed: int = 12345
    
    def add_scenario(self, pattern: ConflictPattern, severity: ConflictSeverity,
                    cpa_time_min: float, cpa_distance_nm: float,
                    altitude_difference_ft: float = 0.0,
                    intruder_speed_kt: float = 450.0,
                    spawn_distance_nm: float = 50.0,
                    seed: Optional[int] = None) -> None:
        """Add a scenario to the set."""
        scenario = ScenarioParameters(
            pattern=pattern,
            severity=severity, 
            cpa_time_min=cpa_time_min,
            cpa_distance_nm=cpa_distance_nm,
            altitude_difference_ft=altitude_difference_ft,
            intruder_speed_kt=intruder_speed_kt,
            spawn_distance_nm=spawn_distance_nm,
            seed=seed
        )
        self.scenarios.append(scenario)

class SystematicIntruderGenerator:
    """Systematic generator for parameterized intruder scenarios."""
    
    def __init__(self, base_seed: int = 12345):
        """Initialize the systematic generator.
        
        Args:
            base_seed: Base seed for reproducible random generation
        """
        self.base_seed = base_seed
        self.rng = random.Random(base_seed)
        
    def generate_scenario_set(self, name: str = "comprehensive") -> SystematicScenarioSet:
        """Generate a comprehensive set of systematic scenarios.
        
        Args:
            name: Name for the scenario set
            
        Returns:
            SystematicScenarioSet with predefined scenarios
        """
        scenario_set = SystematicScenarioSet(
            name=name,
            description="Comprehensive systematic scenario set for CDR testing",
            base_seed=self.base_seed
        )
        
        # Define systematic test matrix
        patterns = [ConflictPattern.CROSSING, ConflictPattern.HEAD_ON, ConflictPattern.OVERTAKE]
        severities = [ConflictSeverity.LOW, ConflictSeverity.MEDIUM, ConflictSeverity.HIGH]
        
        scenario_id = 0
        for pattern in patterns:
            for severity in severities:
                cpa_time, cpa_distance = self._get_severity_parameters(severity)
                
                scenario_set.add_scenario(
                    pattern=pattern,
                    severity=severity,
                    cpa_time_min=cpa_time,
                    cpa_distance_nm=cpa_distance,
                    seed=self.base_seed + scenario_id
                )
                scenario_id += 1
        
        # Add some special test cases
        self._add_edge_cases(scenario_set, scenario_id)
        
        return scenario_set
    
    def _get_severity_parameters(self, severity: ConflictSeverity) -> Tuple[float, float]:
        """Get CPA time and distance parameters for severity level."""
        if severity == ConflictSeverity.LOW:
            return 10.0, 4.0  # 10 min to CPA, 4 NM separation
        elif severity == ConflictSeverity.MEDIUM:
            return 6.0, 2.0   # 6 min to CPA, 2 NM separation
        elif severity == ConflictSeverity.HIGH:
            return 3.0, 0.8   # 3 min to CPA, 0.8 NM separation
        else:  # CRITICAL
            return 1.5, 0.3   # 1.5 min to CPA, 0.3 NM separation
    
    def _add_edge_cases(self, scenario_set: SystematicScenarioSet, start_id: int) -> None:
        """Add edge case scenarios for comprehensive testing."""
        edge_cases = [
            # Vertical conflicts
            (ConflictPattern.HEAD_ON, ConflictSeverity.HIGH, 4.0, 0.0, 500.0),  # Vertical conflict
            (ConflictPattern.CROSSING, ConflictSeverity.MEDIUM, 6.0, 0.0, 800.0),
            
            # High-speed scenarios
            (ConflictPattern.OVERTAKE, ConflictSeverity.HIGH, 2.0, 1.0, 0.0),
            
            # Multiple altitude levels
            (ConflictPattern.PARALLEL, ConflictSeverity.LOW, 8.0, 3.0, 2000.0),
        ]
        
        for i, (pattern, severity, cpa_time, cpa_distance, alt_diff) in enumerate(edge_cases):
            scenario_set.add_scenario(
                pattern=pattern,
                severity=severity,
                cpa_time_min=cpa_time,
                cpa_distance_nm=cpa_distance,
                altitude_difference_ft=alt_diff,
                seed=scenario_set.base_seed + start_id + i
            )
    
    def generate_intruder_for_scenario(self, ownship: AircraftState, 
                                     params: ScenarioParameters) -> AircraftState:
        """Generate a single intruder aircraft based on scenario parameters.
        
        Args:
            ownship: Current ownship state
            params: Scenario parameters
            
        Returns:
            Generated intruder aircraft state
        """
        if params.seed is not None:
            self.rng.seed(params.seed)
        
        # Calculate intruder position and heading based on pattern
        if params.pattern == ConflictPattern.CROSSING:
            return self._generate_crossing_intruder(ownship, params)
        elif params.pattern == ConflictPattern.HEAD_ON:
            return self._generate_head_on_intruder(ownship, params)
        elif params.pattern == ConflictPattern.OVERTAKE:
            return self._generate_overtake_intruder(ownship, params)
        elif params.pattern == ConflictPattern.PARALLEL:
            return self._generate_parallel_intruder(ownship, params)
        elif params.pattern == ConflictPattern.CONVERGING:
            return self._generate_converging_intruder(ownship, params)
        else:  # RANDOM
            return self._generate_random_intruder(ownship, params)
    
    def _generate_crossing_intruder(self, ownship: AircraftState, 
                                  params: ScenarioParameters) -> AircraftState:
        """Generate intruder that will cross ownship path."""
        # Calculate intersection point based on CPA parameters
        cpa_distance_ahead = ownship.ground_speed_kt * (params.cpa_time_min / 60.0)
        
        # CPA point along ownship's path
        cpa_lat, cpa_lon = destination_point_nm(
            ownship.latitude, ownship.longitude,
            ownship.heading_deg,
            cpa_distance_ahead
        )
        
        # Choose crossing angle (typically 30-150 degrees)
        crossing_angle = self.rng.uniform(30, 150)
        intruder_heading = (ownship.heading_deg + crossing_angle) % 360
        
        # Calculate intruder starting position
        # Work backwards from CPA point
        intruder_distance_to_cpa = params.intruder_speed_kt * (params.cpa_time_min / 60.0)
        
        # Offset from CPA point to achieve desired separation
        offset_bearing = (intruder_heading + 180) % 360  # Opposite direction
        intruder_start_lat, intruder_start_lon = destination_point_nm(
            cpa_lat, cpa_lon,
            offset_bearing,
            intruder_distance_to_cpa
        )
        
        # Adjust for desired CPA separation
        if params.cpa_distance_nm > 0:
            perpendicular_bearing = (intruder_heading + 90) % 360
            intruder_start_lat, intruder_start_lon = destination_point_nm(
                intruder_start_lat, intruder_start_lon,
                perpendicular_bearing,
                params.cpa_distance_nm
            )
        
        return AircraftState(
            aircraft_id=f"INTRUDER_CROSSING",
            timestamp=ownship.timestamp,
            latitude=intruder_start_lat,
            longitude=intruder_start_lon,
            altitude_ft=ownship.altitude_ft + params.altitude_difference_ft,
            ground_speed_kt=params.intruder_speed_kt,
            heading_deg=intruder_heading,
            vertical_speed_fpm=0.0,
            aircraft_type="B737",
            spawn_offset_min=0.0
        )
    
    def _generate_head_on_intruder(self, ownship: AircraftState,
                                 params: ScenarioParameters) -> AircraftState:
        """Generate intruder approaching head-on."""
        # Head-on: opposite direction with slight offset for desired CPA distance
        intruder_heading = (ownship.heading_deg + 180) % 360
        
        # Calculate meeting point
        relative_speed = ownship.ground_speed_kt + params.intruder_speed_kt
        total_distance = relative_speed * (params.cpa_time_min / 60.0)
        ownship_distance = (ownship.ground_speed_kt / relative_speed) * total_distance
        
        # Meeting point along ownship path
        meeting_lat, meeting_lon = destination_point_nm(
            ownship.latitude, ownship.longitude,
            ownship.heading_deg,
            ownship_distance
        )
        
        # Intruder starting position (opposite direction from meeting point)
        intruder_distance = total_distance - ownship_distance
        intruder_start_lat, intruder_start_lon = destination_point_nm(
            meeting_lat, meeting_lon,
            intruder_heading,
            intruder_distance
        )
        
        # Add lateral offset for desired CPA separation
        if params.cpa_distance_nm > 0:
            offset_bearing = (ownship.heading_deg + 90) % 360
            intruder_start_lat, intruder_start_lon = destination_point_nm(
                intruder_start_lat, intruder_start_lon,
                offset_bearing,
                params.cpa_distance_nm
            )
        
        return AircraftState(
            aircraft_id=f"INTRUDER_HEAD_ON",
            timestamp=ownship.timestamp,
            latitude=intruder_start_lat,
            longitude=intruder_start_lon,
            altitude_ft=ownship.altitude_ft + params.altitude_difference_ft,
            ground_speed_kt=params.intruder_speed_kt,
            heading_deg=intruder_heading,
            vertical_speed_fpm=0.0,
            aircraft_type="B737",
            spawn_offset_min=0.0
        )
    
    def _generate_overtake_intruder(self, ownship: AircraftState,
                                  params: ScenarioParameters) -> AircraftState:
        """Generate intruder that will overtake ownship."""
        # Overtaking: same general direction but faster
        intruder_speed = max(params.intruder_speed_kt, ownship.ground_speed_kt + 50)
        
        # Slight heading offset for interesting overtake
        heading_offset = self.rng.uniform(-15, 15)
        intruder_heading = (ownship.heading_deg + heading_offset) % 360
        
        # Start behind ownship
        start_distance = params.spawn_distance_nm
        start_bearing = (ownship.heading_deg + 180) % 360  # Behind ownship
        
        intruder_start_lat, intruder_start_lon = destination_point_nm(
            ownship.latitude, ownship.longitude,
            start_bearing,
            start_distance
        )
        
        # Add lateral offset
        lateral_offset = params.cpa_distance_nm + self.rng.uniform(0, 2)
        offset_bearing = (ownship.heading_deg + 90) % 360
        intruder_start_lat, intruder_start_lon = destination_point_nm(
            intruder_start_lat, intruder_start_lon,
            offset_bearing,
            lateral_offset
        )
        
        return AircraftState(
            aircraft_id=f"INTRUDER_OVERTAKE",
            timestamp=ownship.timestamp,
            latitude=intruder_start_lat,
            longitude=intruder_start_lon,
            altitude_ft=ownship.altitude_ft + params.altitude_difference_ft,
            ground_speed_kt=intruder_speed,
            heading_deg=intruder_heading,
            vertical_speed_fpm=0.0,
            aircraft_type="B737",
            spawn_offset_min=0.0
        )
    
    def _generate_parallel_intruder(self, ownship: AircraftState,
                                  params: ScenarioParameters) -> AircraftState:
        """Generate intruder on parallel path."""
        # Parallel: same direction with lateral offset
        intruder_heading = ownship.heading_deg + self.rng.uniform(-5, 5)
        
        # Lateral offset
        offset_bearing = (ownship.heading_deg + 90) % 360
        intruder_start_lat, intruder_start_lon = destination_point_nm(
            ownship.latitude, ownship.longitude,
            offset_bearing,
            params.cpa_distance_nm + 5.0  # Start further apart
        )
        
        return AircraftState(
            aircraft_id=f"INTRUDER_PARALLEL",
            timestamp=ownship.timestamp,
            latitude=intruder_start_lat,
            longitude=intruder_start_lon,
            altitude_ft=ownship.altitude_ft + params.altitude_difference_ft,
            ground_speed_kt=params.intruder_speed_kt,
            heading_deg=intruder_heading,
            vertical_speed_fpm=0.0,
            aircraft_type="B737",
            spawn_offset_min=0.0
        )
    
    def _generate_converging_intruder(self, ownship: AircraftState,
                                    params: ScenarioParameters) -> AircraftState:
        """Generate intruder on converging path."""
        # Converging: paths will meet at an angle
        convergence_angle = self.rng.uniform(20, 60)
        intruder_heading = (ownship.heading_deg + convergence_angle) % 360
        
        # Calculate convergence point and work backwards
        convergence_distance = ownship.ground_speed_kt * (params.cpa_time_min / 60.0)
        convergence_lat, convergence_lon = destination_point_nm(
            ownship.latitude, ownship.longitude,
            ownship.heading_deg,
            convergence_distance
        )
        
        # Intruder distance to convergence point
        intruder_distance = params.intruder_speed_kt * (params.cpa_time_min / 60.0)
        start_bearing = (intruder_heading + 180) % 360
        
        intruder_start_lat, intruder_start_lon = destination_point_nm(
            convergence_lat, convergence_lon,
            start_bearing,
            intruder_distance
        )
        
        return AircraftState(
            aircraft_id=f"INTRUDER_CONVERGING",
            timestamp=ownship.timestamp,
            latitude=intruder_start_lat,
            longitude=intruder_start_lon,
            altitude_ft=ownship.altitude_ft + params.altitude_difference_ft,
            ground_speed_kt=params.intruder_speed_kt,
            heading_deg=intruder_heading,
            vertical_speed_fpm=0.0,
            aircraft_type="B737",
            spawn_offset_min=0.0
        )
    
    def _generate_random_intruder(self, ownship: AircraftState,
                                params: ScenarioParameters) -> AircraftState:
        """Generate random intruder (existing behavior)."""
        # Random position within spawn distance
        bearing = self.rng.uniform(0, 360)
        distance = self.rng.uniform(10, params.spawn_distance_nm)
        
        intruder_lat, intruder_lon = destination_point_nm(
            ownship.latitude, ownship.longitude,
            bearing,
            distance
        )
        
        # Random heading
        intruder_heading = self.rng.uniform(0, 360)
        
        return AircraftState(
            aircraft_id=f"INTRUDER_RANDOM",
            timestamp=ownship.timestamp,
            latitude=intruder_lat,
            longitude=intruder_lon,
            altitude_ft=ownship.altitude_ft + params.altitude_difference_ft,
            ground_speed_kt=params.intruder_speed_kt,
            heading_deg=intruder_heading,
            vertical_speed_fpm=0.0,
            aircraft_type="B737",
            spawn_offset_min=0.0
        )
    
    def save_scenario_set(self, scenario_set: SystematicScenarioSet, 
                         filepath: Path) -> None:
        """Save scenario set to JSON file for reproducibility."""
        data = {
            "name": scenario_set.name,
            "description": scenario_set.description,
            "base_seed": scenario_set.base_seed,
            "scenarios": []
        }
        
        for scenario in scenario_set.scenarios:
            scenario_data = {
                "pattern": scenario.pattern.value,
                "severity": scenario.severity.value,
                "cpa_time_min": scenario.cpa_time_min,
                "cpa_distance_nm": scenario.cpa_distance_nm,
                "altitude_difference_ft": scenario.altitude_difference_ft,
                "intruder_speed_kt": scenario.intruder_speed_kt,
                "spawn_distance_nm": scenario.spawn_distance_nm,
                "seed": scenario.seed
            }
            data["scenarios"].append(scenario_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(scenario_set.scenarios)} scenarios to {filepath}")
    
    def load_scenario_set(self, filepath: Path) -> SystematicScenarioSet:
        """Load scenario set from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        scenario_set = SystematicScenarioSet(
            name=data["name"],
            description=data["description"],
            base_seed=data["base_seed"]
        )
        
        for scenario_data in data["scenarios"]:
            scenario = ScenarioParameters(
                pattern=ConflictPattern(scenario_data["pattern"]),
                severity=ConflictSeverity(scenario_data["severity"]),
                cpa_time_min=scenario_data["cpa_time_min"],
                cpa_distance_nm=scenario_data["cpa_distance_nm"],
                altitude_difference_ft=scenario_data["altitude_difference_ft"],
                intruder_speed_kt=scenario_data["intruder_speed_kt"],
                spawn_distance_nm=scenario_data["spawn_distance_nm"],
                seed=scenario_data["seed"]
            )
            scenario_set.scenarios.append(scenario)
        
        logger.info(f"Loaded {len(scenario_set.scenarios)} scenarios from {filepath}")
        return scenario_set
