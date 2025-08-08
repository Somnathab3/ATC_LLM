"""Stress testing framework for multi-intruder scenarios and Monte Carlo perturbations.

This module implements:
- Multi-intruder conflict scenarios (2-4 aircraft)
- Monte Carlo perturbations for robustness testing
- Comprehensive failure mode analysis
- Performance metrics collection across stress scenarios
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .schemas import AircraftState, ConflictPrediction
from .detect import detect_conflicts
from .resolve import execute_resolution
from .metrics import MetricsCollector
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class StressTestScenario:
    """Define a stress test scenario with multiple aircraft."""
    scenario_id: str
    description: str
    ownship: AircraftState
    intruders: List[AircraftState]
    expected_conflicts: int
    duration_minutes: float = 10.0
    perturbation_std: float = 0.1  # Standard deviation for Monte Carlo perturbations


@dataclass
class StressTestResult:
    """Results from a stress test run."""
    scenario_id: str
    run_id: str
    timestamp: datetime
    
    # Performance metrics
    total_conflicts_detected: int
    conflicts_resolved: int
    resolution_success_rate: float
    
    # Timing metrics
    avg_detection_time_sec: float
    avg_resolution_time_sec: float
    total_processing_time_sec: float
    
    # Safety metrics
    min_separation_nm: float
    safety_violations: int
    near_misses: int  # < 1nm separation
    
    # Failure analysis
    late_detections: int
    missed_conflicts: int
    unsafe_resolutions: int
    oscillations: int
    
    # Raw data for detailed analysis
    conflict_timeline: List[Dict[str, Any]] = field(default_factory=list)
    resolution_timeline: List[Dict[str, Any]] = field(default_factory=list)
    metrics_data: Optional[Dict[str, Any]] = None


class StressTestFramework:
    """Framework for conducting stress tests with multiple intruders."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize stress test framework.
        
        Args:
            llm_client: LLM client for conflict resolution
        """
        self.llm_client = llm_client
        self.metrics_collector = MetricsCollector()
        self.test_results: List[StressTestResult] = []
        
    def create_converging_scenario(
        self,
        scenario_id: str,
        num_intruders: int = 2,
        convergence_point: Tuple[float, float] = (0.0, 0.0),
        altitude_spread_ft: float = 500.0
    ) -> StressTestScenario:
        """Create a converging aircraft scenario.
        
        Args:
            scenario_id: Unique identifier for scenario
            num_intruders: Number of intruder aircraft (2-4)
            convergence_point: Lat/lon where aircraft converge
            altitude_spread_ft: Altitude variation between aircraft
            
        Returns:
            Configured stress test scenario
        """
        # Create ownship at convergence point
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            latitude=convergence_point[0],
            longitude=convergence_point[1],
            altitude_ft=10000.0,
            heading_deg=90.0,  # Heading east
            ground_speed_kt=250.0,
            vertical_speed_fpm=0.0,
            timestamp=datetime.now()
        )
        
        # Create intruders approaching from different directions
        intruders = []
        base_distance_nm = 15.0  # Start 15nm away
        
        for i in range(num_intruders):
            # Distribute intruders around the convergence point
            approach_angle = (i + 1) * (360.0 / (num_intruders + 1))
            
            # Calculate intruder position
            lat_offset = base_distance_nm * np.cos(np.radians(approach_angle)) / 60.0
            lon_offset = base_distance_nm * np.sin(np.radians(approach_angle)) / 60.0
            
            # Heading toward convergence point
            heading_to_convergence = (approach_angle + 180) % 360
            
            # Altitude with some spread
            altitude = 10000.0 + (i - num_intruders/2) * altitude_spread_ft
            
            intruder = AircraftState(
                aircraft_id=f"INTRUDER_{i+1}",
                latitude=convergence_point[0] + lat_offset,
                longitude=convergence_point[1] + lon_offset,
                altitude_ft=altitude,
                heading_deg=heading_to_convergence,
                ground_speed_kt=250.0,
                vertical_speed_fpm=0.0,
                timestamp=datetime.now()
            )
            intruders.append(intruder)
        
        return StressTestScenario(
            scenario_id=scenario_id,
            description=f"Converging scenario with {num_intruders} intruders",
            ownship=ownship,
            intruders=intruders,
            expected_conflicts=num_intruders  # Expect conflict with each intruder
        )
    
    def create_crossing_scenario(
        self,
        scenario_id: str,
        num_intruders: int = 3
    ) -> StressTestScenario:
        """Create a crossing traffic scenario.
        
        Args:
            scenario_id: Unique identifier for scenario
            num_intruders: Number of crossing aircraft
            
        Returns:
            Configured stress test scenario
        """
        # Ownship flying north
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            latitude=0.0,
            longitude=0.0,
            altitude_ft=10000.0,
            heading_deg=0.0,  # North
            ground_speed_kt=250.0,
            vertical_speed_fpm=0.0,
            timestamp=datetime.now()
        )
        
        intruders = []
        for i in range(num_intruders):
            # Create aircraft crossing from different angles
            if i == 0:
                # Aircraft crossing from east to west
                intruder = AircraftState(
                    aircraft_id="CROSS_EW",
                    latitude_deg=0.1,  # Slightly north
                    longitude_deg=-0.2,  # West
                    altitude_ft=10000.0,
                    heading_deg=90.0,  # East
                    ground_speed_kt=250.0,
                    timestamp=datetime.now()
                )
            elif i == 1:
                # Aircraft crossing diagonally
                intruder = AircraftState(
                    aircraft_id="CROSS_DIAG",
                    latitude_deg=-0.1,  # South
                    longitude_deg=-0.1,  # West
                    altitude_ft=10500.0,
                    heading_deg=45.0,  # Northeast
                    ground_speed_kt=280.0,
                    timestamp=datetime.now()
                )
            else:
                # Head-on traffic
                intruder = AircraftState(
                    aircraft_id="HEAD_ON",
                    latitude_deg=0.3,  # North
                    longitude_deg=0.0,
                    altitude_ft=10000.0,
                    heading_deg=180.0,  # South (head-on)
                    ground_speed_kt=240.0,
                    timestamp=datetime.now()
                )
            
            intruders.append(intruder)
        
        return StressTestScenario(
            scenario_id=scenario_id,
            description=f"Crossing scenario with {num_intruders} aircraft",
            ownship=ownship,
            intruders=intruders,
            expected_conflicts=num_intruders
        )
    
    def apply_monte_carlo_perturbations(
        self,
        scenario: StressTestScenario,
        perturbation_std: float = 0.1
    ) -> StressTestScenario:
        """Apply random perturbations to aircraft states for Monte Carlo testing.
        
        Args:
            scenario: Base scenario to perturb
            perturbation_std: Standard deviation for perturbations
            
        Returns:
            Perturbed scenario
        """
        # Perturb ownship
        perturbed_ownship = self._perturb_aircraft_state(scenario.ownship, perturbation_std)
        
        # Perturb intruders
        perturbed_intruders = [
            self._perturb_aircraft_state(intruder, perturbation_std)
            for intruder in scenario.intruders
        ]
        
        return StressTestScenario(
            scenario_id=f"{scenario.scenario_id}_mc_{random.randint(1000, 9999)}",
            description=f"{scenario.description} (Monte Carlo perturbed)",
            ownship=perturbed_ownship,
            intruders=perturbed_intruders,
            expected_conflicts=scenario.expected_conflicts,
            duration_minutes=scenario.duration_minutes,
            perturbation_std=perturbation_std
        )
    
    def _perturb_aircraft_state(
        self,
        aircraft: AircraftState,
        std_dev: float
    ) -> AircraftState:
        """Apply random perturbations to an aircraft state."""
        # Perturbation ranges
        position_std = std_dev * 0.05  # ~3nm at std=0.1
        heading_std = std_dev * 10.0   # ~1 degree at std=0.1
        speed_std = std_dev * 25.0     # ~2.5 knots at std=0.1
        altitude_std = std_dev * 200.0 # ~20 feet at std=0.1
        
        return AircraftState(
            aircraft_id=aircraft.aircraft_id,
            latitude_deg=aircraft.latitude_deg + np.random.normal(0, position_std),
            longitude_deg=aircraft.longitude_deg + np.random.normal(0, position_std),
            altitude_ft=aircraft.altitude_ft + np.random.normal(0, altitude_std),
            heading_deg=(aircraft.heading_deg + np.random.normal(0, heading_std)) % 360,
            ground_speed_kt=max(100, aircraft.ground_speed_kt + np.random.normal(0, speed_std)),
            timestamp=aircraft.timestamp
        )
    
    def run_stress_test(
        self,
        scenario: StressTestScenario,
        simulation_step_sec: float = 30.0
    ) -> StressTestResult:
        """Run a stress test scenario.
        
        Args:
            scenario: Scenario to test
            simulation_step_sec: Time step for simulation
            
        Returns:
            Test results with metrics and failure analysis
        """
        start_time = datetime.now()
        run_id = f"run_{int(start_time.timestamp())}"
        
        logger.info(f"Starting stress test: {scenario.scenario_id}")
        
        # Initialize result tracking
        result = StressTestResult(
            scenario_id=scenario.scenario_id,
            run_id=run_id,
            timestamp=start_time,
            total_conflicts_detected=0,
            conflicts_resolved=0,
            resolution_success_rate=0.0,
            avg_detection_time_sec=0.0,
            avg_resolution_time_sec=0.0,
            total_processing_time_sec=0.0,
            min_separation_nm=float('inf'),
            safety_violations=0,
            near_misses=0,
            late_detections=0,
            missed_conflicts=0,
            unsafe_resolutions=0,
            oscillations=0
        )
        
        # Simulation state
        current_ownship = scenario.ownship
        current_intruders = scenario.intruders.copy()
        simulation_time = 0.0
        
        detection_times = []
        resolution_times = []
        
        # Run simulation - ensure at least one cycle executes
        simulation_step_sec = max(10.0, simulation_step_sec)  # Minimum 10 seconds per step
        min_simulation_time = max(60.0, scenario.duration_minutes * 60.0)  # At least 1 minute
        
        while simulation_time < min_simulation_time:
            step_start = datetime.now()
            
            # Detect conflicts
            conflicts = []
            for intruder in current_intruders:
                try:
                    conflict = detect_conflicts(current_ownship, [intruder])
                    if conflict:
                        conflicts.extend(conflict)
                        result.total_conflicts_detected += 1
                        
                        # Track detection timing
                        detection_time = (datetime.now() - step_start).total_seconds()
                        detection_times.append(detection_time)
                        
                        # Log conflict details
                        result.conflict_timeline.append({
                            'time': simulation_time,
                            'conflict_id': f"conflict_{len(result.conflict_timeline)}",
                            'intruder_id': intruder.aircraft_id,
                            'time_to_cpa': conflict[0].time_to_cpa_min if conflict else None,
                            'min_separation': conflict[0].min_separation_nm if conflict else None
                        })
                        
                except Exception as e:
                    logger.error(f"Conflict detection failed: {e}")
                    result.missed_conflicts += 1
            
            # Process resolutions if conflicts found
            if conflicts and self.llm_client:
                for conflict in conflicts:
                    resolution_start = datetime.now()
                    
                    try:
                        # Get LLM resolution
                        llm_resolution = self.llm_client.resolve_conflict(
                            current_ownship, conflict.intruder, conflict
                        )
                        
                        if llm_resolution:
                            # Execute resolution
                            resolution_cmd = execute_resolution(
                                llm_resolution, current_ownship, conflict.intruder, conflict
                            )
                            
                            if resolution_cmd and resolution_cmd.is_validated:
                                result.conflicts_resolved += 1
                                
                                # Apply resolution to ownship state
                                current_ownship = self._apply_resolution_to_state(
                                    current_ownship, resolution_cmd
                                )
                                
                                # Track resolution timing
                                resolution_time = (datetime.now() - resolution_start).total_seconds()
                                resolution_times.append(resolution_time)
                                
                                # Log resolution details
                                result.resolution_timeline.append({
                                    'time': simulation_time,
                                    'resolution_id': resolution_cmd.resolution_id,
                                    'action': llm_resolution.action,
                                    'rationale': llm_resolution.rationale,
                                    'processing_time': resolution_time
                                })
                                
                            else:
                                result.unsafe_resolutions += 1
                        
                    except Exception as e:
                        logger.error(f"Resolution processing failed: {e}")
                        result.unsafe_resolutions += 1
            
            # Update aircraft positions (simplified)
            current_ownship = self._update_aircraft_position(
                current_ownship, simulation_step_sec
            )
            current_intruders = [
                self._update_aircraft_position(intruder, simulation_step_sec)
                for intruder in current_intruders
            ]
            
            # Calculate separations and check for violations
            for intruder in current_intruders:
                separation = self._calculate_separation(current_ownship, intruder)
                result.min_separation_nm = min(result.min_separation_nm, separation)
                
                if separation < 3.0:  # 3nm minimum separation
                    result.safety_violations += 1
                if separation < 1.0:  # Near miss
                    result.near_misses += 1
            
            simulation_time += simulation_step_sec
        
        # Calculate final metrics
        result.total_processing_time_sec = (datetime.now() - start_time).total_seconds()
        result.resolution_success_rate = (
            result.conflicts_resolved / max(1, result.total_conflicts_detected)
        )
        result.avg_detection_time_sec = np.mean(detection_times) if detection_times else 0.0
        result.avg_resolution_time_sec = np.mean(resolution_times) if resolution_times else 0.0
        
        if result.min_separation_nm == float('inf'):
            result.min_separation_nm = 999.0  # No conflicts occurred
        
        logger.info(f"Stress test completed: {scenario.scenario_id}")
        logger.info(f"  Conflicts detected: {result.total_conflicts_detected}")
        logger.info(f"  Conflicts resolved: {result.conflicts_resolved}")
        logger.info(f"  Success rate: {result.resolution_success_rate:.2%}")
        logger.info(f"  Min separation: {result.min_separation_nm:.1f}nm")
        
        self.test_results.append(result)
        return result
    
    def _apply_resolution_to_state(
        self,
        aircraft: AircraftState,
        resolution: Any
    ) -> AircraftState:
        """Apply resolution command to aircraft state."""
        # This is a simplified implementation
        # In practice, this would integrate with the flight dynamics model
        new_heading = aircraft.heading_deg
        new_altitude = aircraft.altitude_ft
        new_speed = aircraft.ground_speed_kt
        
        if hasattr(resolution, 'new_heading_deg') and resolution.new_heading_deg:
            new_heading = resolution.new_heading_deg
        if hasattr(resolution, 'new_altitude_ft') and resolution.new_altitude_ft:
            new_altitude = resolution.new_altitude_ft
        if hasattr(resolution, 'new_speed_kt') and resolution.new_speed_kt:
            new_speed = resolution.new_speed_kt
        
        return AircraftState(
            aircraft_id=aircraft.aircraft_id,
            latitude_deg=aircraft.latitude_deg,
            longitude_deg=aircraft.longitude_deg,
            altitude_ft=new_altitude,
            heading_deg=new_heading,
            ground_speed_kt=new_speed,
            timestamp=datetime.now()
        )
    
    def _update_aircraft_position(
        self,
        aircraft: AircraftState,
        time_step_sec: float
    ) -> AircraftState:
        """Update aircraft position based on current state and time step."""
        # Simple dead reckoning - convert speed to distance
        distance_nm = aircraft.ground_speed_kt * (time_step_sec / 3600.0)
        
        # Convert to lat/lon changes
        lat_change = distance_nm * np.cos(np.radians(aircraft.heading_deg)) / 60.0
        lon_change = distance_nm * np.sin(np.radians(aircraft.heading_deg)) / 60.0
        
        return AircraftState(
            aircraft_id=aircraft.aircraft_id,
            latitude_deg=aircraft.latitude_deg + lat_change,
            longitude_deg=aircraft.longitude_deg + lon_change,
            altitude_ft=aircraft.altitude_ft,
            heading_deg=aircraft.heading_deg,
            ground_speed_kt=aircraft.ground_speed_kt,
            timestamp=datetime.now()
        )
    
    def _calculate_separation(
        self,
        aircraft1: AircraftState,
        aircraft2: AircraftState
    ) -> float:
        """Calculate horizontal separation between two aircraft."""
        # Simplified calculation - use haversine for accuracy
        lat1, lon1 = np.radians(aircraft1.latitude_deg), np.radians(aircraft1.longitude_deg)
        lat2, lon2 = np.radians(aircraft2.latitude_deg), np.radians(aircraft2.longitude_deg)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Convert to nautical miles
        return c * 3440.065  # Earth radius in nautical miles
    
    def run_monte_carlo_analysis(
        self,
        base_scenario: StressTestScenario,
        num_runs: int = 100,
        perturbation_std: float = 0.1
    ) -> List[StressTestResult]:
        """Run Monte Carlo analysis with multiple perturbed scenarios.
        
        Args:
            base_scenario: Base scenario to perturb
            num_runs: Number of Monte Carlo runs
            perturbation_std: Standard deviation for perturbations
            
        Returns:
            List of results from all Monte Carlo runs
        """
        logger.info(f"Starting Monte Carlo analysis: {num_runs} runs")
        
        results = []
        for i in range(num_runs):
            logger.info(f"Monte Carlo run {i+1}/{num_runs}")
            
            # Create perturbed scenario
            perturbed_scenario = self.apply_monte_carlo_perturbations(
                base_scenario, perturbation_std
            )
            
            # Run test
            result = self.run_stress_test(perturbed_scenario)
            results.append(result)
        
        logger.info("Monte Carlo analysis completed")
        return results
    
    def get_failure_mode_analysis(self) -> Dict[str, Any]:
        """Analyze failure modes across all test results.
        
        Returns:
            Comprehensive failure mode analysis
        """
        if not self.test_results:
            return {"error": "No test results available"}
        
        total_tests = len(self.test_results)
        
        # Aggregate failure statistics
        analysis = {
            "total_tests": total_tests,
            "late_detections": {
                "total": sum(r.late_detections for r in self.test_results),
                "rate": sum(r.late_detections for r in self.test_results) / total_tests,
                "scenarios_affected": len([r for r in self.test_results if r.late_detections > 0])
            },
            "missed_conflicts": {
                "total": sum(r.missed_conflicts for r in self.test_results),
                "rate": sum(r.missed_conflicts for r in self.test_results) / total_tests,
                "scenarios_affected": len([r for r in self.test_results if r.missed_conflicts > 0])
            },
            "unsafe_resolutions": {
                "total": sum(r.unsafe_resolutions for r in self.test_results),
                "rate": sum(r.unsafe_resolutions for r in self.test_results) / total_tests,
                "scenarios_affected": len([r for r in self.test_results if r.unsafe_resolutions > 0])
            },
            "oscillations": {
                "total": sum(r.oscillations for r in self.test_results),
                "rate": sum(r.oscillations for r in self.test_results) / total_tests,
                "scenarios_affected": len([r for r in self.test_results if r.oscillations > 0])
            },
            "safety_violations": {
                "total": sum(r.safety_violations for r in self.test_results),
                "rate": sum(r.safety_violations for r in self.test_results) / total_tests,
                "near_misses": sum(r.near_misses for r in self.test_results)
            },
            "performance_metrics": {
                "avg_resolution_success_rate": np.mean([r.resolution_success_rate for r in self.test_results]),
                "avg_detection_time": np.mean([r.avg_detection_time_sec for r in self.test_results]),
                "avg_resolution_time": np.mean([r.avg_resolution_time_sec for r in self.test_results]),
                "min_separation_observed": min(r.min_separation_nm for r in self.test_results)
            }
        }
        
        return analysis
