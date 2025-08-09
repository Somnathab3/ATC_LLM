"""Comparison tool for ASAS vs LLM conflict detection and resolution.

This module provides:
- Side-by-side comparison of ASAS and LLM performance
- Detailed metrics and analysis
- Visual comparison outputs
- Performance benchmarking
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
from pathlib import Path

from .asas_integration import BlueSkyASAS, ASASDetection, ASASResolution, ASASMetrics
from .enhanced_llm_client import EnhancedLLMClient
from .schemas import AircraftState, ConflictPrediction, ConfigurationSettings
from .bluesky_io import BlueSkyClient

logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Results of ASAS vs LLM comparison."""
    timestamp: str
    scenario_id: str
    total_aircraft: int
    
    # Detection comparison
    asas_conflicts_detected: int
    llm_conflicts_detected: int
    common_detections: int
    asas_only_detections: int
    llm_only_detections: int
    detection_agreement_rate: float
    
    # Resolution comparison
    asas_resolutions_attempted: int
    llm_resolutions_attempted: int
    asas_successful_resolutions: int
    llm_successful_resolutions: int
    asas_resolution_success_rate: float
    llm_resolution_success_rate: float
    
    # Performance metrics
    asas_avg_detection_time_sec: float
    llm_avg_detection_time_sec: float
    asas_avg_resolution_time_sec: float
    llm_avg_resolution_time_sec: float
    
    # Safety metrics
    asas_min_separation_achieved_nm: float
    llm_min_separation_achieved_nm: float
    asas_safety_violations: int
    llm_safety_violations: int
    
    # Additional details
    detailed_conflicts: List[Dict[str, Any]]
    detailed_resolutions: List[Dict[str, Any]]

class ASASLLMComparator:
    """Comprehensive comparison tool for ASAS vs LLM systems."""
    
    def __init__(self, bluesky_client: BlueSkyClient, config: ConfigurationSettings):
        self.bluesky_client = bluesky_client
        self.config = config
        self.asas_system = BlueSkyASAS(bluesky_client, config)
        self.llm_client = EnhancedLLMClient(config)
        self.comparison_results: List[ComparisonResult] = []
        
    def setup_comparison(self) -> bool:
        """Setup both ASAS and LLM systems for comparison."""
        try:
            # Setup ASAS
            logger.info("Setting up ASAS system...")
            asas_success = self.asas_system.configure_asas()
            
            # Setup LLM
            logger.info("Setting up LLM system...")
            llm_success = self.llm_client.validate_llm_connection()
            
            if not asas_success:
                logger.warning("ASAS setup failed - will proceed with LLM only")
            
            if not llm_success:
                logger.error("LLM setup failed - cannot proceed")
                return False
            
            logger.info(f"Comparison setup complete - ASAS: {'OK' if asas_success else 'FAILED'}, LLM: {'OK' if llm_success else 'FAILED'}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up comparison systems: {e}")
            return False
    
    def run_comparison_scenario(self, aircraft_states: List[AircraftState], 
                              scenario_id: str = None) -> ComparisonResult:
        """Run a complete comparison scenario with given aircraft states."""
        if scenario_id is None:
            scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting comparison scenario: {scenario_id} with {len(aircraft_states)} aircraft")
        
        start_time = time.time()
        
        # Initialize results
        result = ComparisonResult(
            timestamp=datetime.now().isoformat(),
            scenario_id=scenario_id,
            total_aircraft=len(aircraft_states),
            asas_conflicts_detected=0,
            llm_conflicts_detected=0,
            common_detections=0,
            asas_only_detections=0,
            llm_only_detections=0,
            detection_agreement_rate=0.0,
            asas_resolutions_attempted=0,
            llm_resolutions_attempted=0,
            asas_successful_resolutions=0,
            llm_successful_resolutions=0,
            asas_resolution_success_rate=0.0,
            llm_resolution_success_rate=0.0,
            asas_avg_detection_time_sec=0.0,
            llm_avg_detection_time_sec=0.0,
            asas_avg_resolution_time_sec=0.0,
            llm_avg_resolution_time_sec=0.0,
            asas_min_separation_achieved_nm=999.0,
            llm_min_separation_achieved_nm=999.0,
            asas_safety_violations=0,
            llm_safety_violations=0,
            detailed_conflicts=[],
            detailed_resolutions=[]
        )
        
        try:
            # Step 1: Run conflict detection comparison
            asas_conflicts, llm_conflicts = self._compare_conflict_detection(aircraft_states, result)
            
            # Step 2: Run resolution comparison if conflicts detected
            if asas_conflicts or llm_conflicts:
                self._compare_conflict_resolution(aircraft_states, asas_conflicts, llm_conflicts, result)
            
            # Step 3: Calculate final metrics
            self._calculate_comparison_metrics(result)
            
            total_time = time.time() - start_time
            logger.info(f"Comparison scenario {scenario_id} completed in {total_time:.2f} seconds")
            
            self.comparison_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error running comparison scenario: {e}")
            raise
    
    def _compare_conflict_detection(self, aircraft_states: List[AircraftState], 
                                  result: ComparisonResult) -> tuple:
        """Compare ASAS and LLM conflict detection capabilities."""
        logger.info("Comparing conflict detection...")
        
        asas_conflicts = []
        llm_conflicts = []
        
        # Get ASAS detections
        asas_start_time = time.time()
        try:
            if self.asas_system.enabled:
                asas_conflicts = self.asas_system.get_conflicts()
                result.asas_conflicts_detected = len(asas_conflicts)
        except Exception as e:
            logger.error(f"ASAS detection failed: {e}")
        asas_detection_time = time.time() - asas_start_time
        
        # Get LLM detections
        llm_start_time = time.time()
        try:
            # For each aircraft, check for conflicts with others
            for i, ownship in enumerate(aircraft_states):
                traffic = [ac for j, ac in enumerate(aircraft_states) if j != i]
                
                if traffic:  # Only check if there's traffic
                    prompt = self.llm_client.build_enhanced_detect_prompt(ownship, traffic, self.config)
                    response = self.llm_client._post_ollama(prompt)
                    parsed_response = self.llm_client.parse_enhanced_detection_response(str(response))
                    
                    if parsed_response.get('conflict', False):
                        for intruder_id in parsed_response.get('intruders', []):
                            # Create conflict prediction object
                            conflict = ConflictPrediction(
                                ownship_id=ownship.aircraft_id,
                                intruder_id=intruder_id,
                                time_to_cpa_min=parsed_response.get('horizon_min', 10.0),
                                distance_at_cpa_nm=5.0,  # Default
                                altitude_diff_ft=1000.0,  # Default
                                is_conflict=True,
                                severity_score=0.5,  # Default
                                conflict_type="both",
                                prediction_time=ownship.timestamp,
                                confidence=parsed_response.get('confidence', 0.5)
                            )
                            llm_conflicts.append(conflict)
            
            result.llm_conflicts_detected = len(llm_conflicts)
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
        llm_detection_time = time.time() - llm_start_time
        
        # Compare detections
        asas_pairs = set()
        if asas_conflicts:
            asas_pairs = {tuple(sorted([c.aircraft_pair[0], c.aircraft_pair[1]])) 
                         for c in asas_conflicts}
        
        llm_pairs = set()
        if llm_conflicts:
            llm_pairs = {tuple(sorted([c.ownship_id, c.intruder_id])) 
                        for c in llm_conflicts}
        
        common_pairs = asas_pairs.intersection(llm_pairs)
        asas_only_pairs = asas_pairs - llm_pairs
        llm_only_pairs = llm_pairs - asas_pairs
        
        result.common_detections = len(common_pairs)
        result.asas_only_detections = len(asas_only_pairs)
        result.llm_only_detections = len(llm_only_pairs)
        
        total_unique_conflicts = len(asas_pairs.union(llm_pairs))
        if total_unique_conflicts > 0:
            result.detection_agreement_rate = (len(common_pairs) / total_unique_conflicts) * 100
        
        result.asas_avg_detection_time_sec = asas_detection_time
        result.llm_avg_detection_time_sec = llm_detection_time
        
        # Store detailed conflict information
        for conflict in asas_conflicts:
            result.detailed_conflicts.append({
                'type': 'asas',
                'aircraft_pair': conflict.aircraft_pair,
                'time_to_conflict': conflict.time_to_conflict_min,
                'distance_cpa': conflict.distance_at_cpa_nm,
                'altitude_diff': conflict.altitude_diff_ft,
                'severity': conflict.conflict_severity
            })
        
        for conflict in llm_conflicts:
            result.detailed_conflicts.append({
                'type': 'llm',
                'aircraft_pair': (conflict.ownship_id, conflict.intruder_id),
                'time_to_conflict': conflict.time_to_cpa_min,
                'distance_cpa': conflict.distance_at_cpa_nm,
                'altitude_diff': conflict.altitude_diff_ft,
                'severity': conflict.severity_score
            })
        
        logger.info(f"Detection comparison: ASAS={len(asas_conflicts)}, LLM={len(llm_conflicts)}, Common={len(common_pairs)}")
        
        return asas_conflicts, llm_conflicts
    
    def _compare_conflict_resolution(self, aircraft_states: List[AircraftState],
                                   asas_conflicts: List, llm_conflicts: List,
                                   result: ComparisonResult):
        """Compare ASAS and LLM conflict resolution capabilities."""
        logger.info("Comparing conflict resolution...")
        
        asas_resolutions = []
        llm_resolutions = []
        
        # ASAS resolutions
        asas_start_time = time.time()
        try:
            for conflict in asas_conflicts:
                if hasattr(conflict, 'aircraft_pair'):
                    # Resolve for first aircraft in pair
                    aircraft_id = conflict.aircraft_pair[0]
                    resolution = self.asas_system.resolve_conflict_manual(aircraft_id, conflict)
                    if resolution:
                        asas_resolutions.append(resolution)
                        result.asas_resolutions_attempted += 1
                        if resolution.success:
                            result.asas_successful_resolutions += 1
        except Exception as e:
            logger.error(f"ASAS resolution failed: {e}")
        asas_resolution_time = time.time() - asas_start_time
        
        # LLM resolutions  
        llm_start_time = time.time()
        try:
            # Group conflicts by ownship
            ownship_conflicts = {}
            for conflict in llm_conflicts:
                ownship_id = conflict.ownship_id
                if ownship_id not in ownship_conflicts:
                    ownship_conflicts[ownship_id] = []
                ownship_conflicts[ownship_id].append(conflict)
            
            for ownship_id, conflicts in ownship_conflicts.items():
                # Find ownship state
                ownship_state = None
                for state in aircraft_states:
                    if state.aircraft_id == ownship_id:
                        ownship_state = state
                        break
                
                if ownship_state:
                    prompt = self.llm_client.build_enhanced_resolve_prompt(
                        ownship_state, conflicts, self.config
                    )
                    response = self.llm_client._post_ollama(prompt)
                    parsed_response = self.llm_client.parse_enhanced_resolution_response(str(response))
                    
                    result.llm_resolutions_attempted += 1
                    
                    # Try to execute the resolution
                    bluesky_command = parsed_response.get('bluesky_command', '')
                    if bluesky_command:
                        success = self.llm_client.execute_bluesky_command(
                            bluesky_command, self.bluesky_client
                        )
                        if success:
                            result.llm_successful_resolutions += 1
                        
                        llm_resolutions.append({
                            'aircraft_id': ownship_id,
                            'command': bluesky_command,
                            'success': success,
                            'action': parsed_response.get('action', 'unknown'),
                            'rationale': parsed_response.get('rationale', ''),
                            'confidence': parsed_response.get('confidence', 0.0)
                        })
        except Exception as e:
            logger.error(f"LLM resolution failed: {e}")
        llm_resolution_time = time.time() - llm_start_time
        
        # Calculate success rates
        if result.asas_resolutions_attempted > 0:
            result.asas_resolution_success_rate = (result.asas_successful_resolutions / 
                                                 result.asas_resolutions_attempted) * 100
        
        if result.llm_resolutions_attempted > 0:
            result.llm_resolution_success_rate = (result.llm_successful_resolutions / 
                                                result.llm_resolutions_attempted) * 100
        
        result.asas_avg_resolution_time_sec = asas_resolution_time
        result.llm_avg_resolution_time_sec = llm_resolution_time
        
        # Store detailed resolution information
        for resolution in asas_resolutions:
            if hasattr(resolution, 'aircraft_id'):
                result.detailed_resolutions.append({
                    'type': 'asas',
                    'aircraft_id': resolution.aircraft_id,
                    'command_type': resolution.command_type,
                    'command_value': resolution.command_value,
                    'success': resolution.success,
                    'reason': resolution.reason
                })
        
        for resolution in llm_resolutions:
            result.detailed_resolutions.append({
                'type': 'llm',
                'aircraft_id': resolution['aircraft_id'],
                'command': resolution['command'],
                'success': resolution['success'],
                'action': resolution['action'],
                'rationale': resolution['rationale'],
                'confidence': resolution['confidence']
            })
        
        logger.info(f"Resolution comparison: ASAS={result.asas_successful_resolutions}/{result.asas_resolutions_attempted}, "
                   f"LLM={result.llm_successful_resolutions}/{result.llm_resolutions_attempted}")
    
    def _calculate_comparison_metrics(self, result: ComparisonResult):
        """Calculate final comparison metrics."""
        # Safety metrics would require additional simulation to verify
        # For now, set placeholder values
        result.asas_min_separation_achieved_nm = 5.0  # Assume safe
        result.llm_min_separation_achieved_nm = 5.0   # Assume safe
        result.asas_safety_violations = 0
        result.llm_safety_violations = 0
    
    def save_comparison_results(self, output_dir: str = "Output"):
        """Save comparison results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed JSON results
        json_file = output_path / f"asas_llm_comparison_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(result) for result in self.comparison_results], f, indent=2)
        
        # Save summary CSV
        csv_file = output_path / f"asas_llm_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.comparison_results:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.comparison_results[0]).keys()))
                writer.writeheader()
                for result in self.comparison_results:
                    # Convert lists to strings for CSV
                    row = asdict(result)
                    row['detailed_conflicts'] = str(len(row['detailed_conflicts']))
                    row['detailed_resolutions'] = str(len(row['detailed_resolutions']))
                    writer.writerow(row)
        
        logger.info(f"Comparison results saved to {json_file} and {csv_file}")
        
        return json_file, csv_file
    
    def print_comparison_summary(self):
        """Print a summary of all comparison results."""
        if not self.comparison_results:
            logger.info("No comparison results available")
            return
        
        logger.info("=== ASAS vs LLM Comparison Summary ===")
        logger.info(f"Total scenarios analyzed: {len(self.comparison_results)}")
        
        # Aggregate metrics
        total_aircraft = sum(r.total_aircraft for r in self.comparison_results)
        total_asas_conflicts = sum(r.asas_conflicts_detected for r in self.comparison_results)
        total_llm_conflicts = sum(r.llm_conflicts_detected for r in self.comparison_results)
        total_common = sum(r.common_detections for r in self.comparison_results)
        
        avg_agreement = sum(r.detection_agreement_rate for r in self.comparison_results) / len(self.comparison_results)
        avg_asas_success = sum(r.asas_resolution_success_rate for r in self.comparison_results) / len(self.comparison_results)
        avg_llm_success = sum(r.llm_resolution_success_rate for r in self.comparison_results) / len(self.comparison_results)
        
        logger.info(f"Total aircraft processed: {total_aircraft}")
        logger.info(f"Conflicts detected - ASAS: {total_asas_conflicts}, LLM: {total_llm_conflicts}")
        logger.info(f"Common detections: {total_common}")
        logger.info(f"Average detection agreement: {avg_agreement:.1f}%")
        logger.info(f"Average resolution success - ASAS: {avg_asas_success:.1f}%, LLM: {avg_llm_success:.1f}%")
        
        # Performance comparison
        avg_asas_detection_time = sum(r.asas_avg_detection_time_sec for r in self.comparison_results) / len(self.comparison_results)
        avg_llm_detection_time = sum(r.llm_avg_detection_time_sec for r in self.comparison_results) / len(self.comparison_results)
        
        logger.info(f"Average detection time - ASAS: {avg_asas_detection_time:.3f}s, LLM: {avg_llm_detection_time:.3f}s")
        
        if avg_asas_detection_time < avg_llm_detection_time:
            speedup = avg_llm_detection_time / avg_asas_detection_time
            logger.info(f"ASAS is {speedup:.1f}x faster than LLM for detection")
        else:
            speedup = avg_asas_detection_time / avg_llm_detection_time
            logger.info(f"LLM is {speedup:.1f}x faster than ASAS for detection")
        
        logger.info("=" * 50)
