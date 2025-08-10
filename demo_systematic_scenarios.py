#!/usr/bin/env python3
"""Demonstration of Systematic Intruder Generation and Enhanced Metrics Collection.

This script demonstrates the complete workflow for:
1. Generating systematic, reproducible intruder scenarios
2. Running conflict detection with enhanced CPA calculations  
3. Collecting comprehensive Wolfgang (2011) metrics
4. Analyzing and reporting results

Example Usage:
    python demo_systematic_scenarios.py --scenarios comprehensive --runs 20
    python demo_systematic_scenarios.py --scenarios edge_cases --seed 12345
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.cdr.systematic_intruders import (
    SystematicIntruderGenerator, 
    ConflictPattern, 
    ConflictSeverity,
    ScenarioParameters,
    SystematicScenarioSet
)
from src.cdr.enhanced_metrics import (
    EnhancedMetricsCollector,
    ConflictType,
    MetricType
)
from src.cdr.schemas import AircraftState
from src.cdr.enhanced_cpa import calculate_enhanced_cpa, CPAResult
from src.cdr.geodesy import haversine_nm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystematicScenarioDemo:
    """Demonstration of systematic scenario generation and metrics collection."""
    
    def __init__(self, output_dir: Path = Path("demo_output")):
        """Initialize the demonstration.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.intruder_generator = SystematicIntruderGenerator()
        self.metrics_collector = EnhancedMetricsCollector(self.output_dir / "metrics")
        
        # Demo configuration
        self.ownship_state = AircraftState(
            aircraft_id="OWNSHIP_001",
            timestamp=datetime.now(),
            latitude=59.3293,    # Stockholm
            longitude=18.0686,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,      # Eastbound
            vertical_speed_fpm=0,
            aircraft_type="A320",
            spawn_offset_min=0
        )
    
    def run_comprehensive_demo(self, num_runs: int = 15) -> Dict[str, Any]:
        """Run comprehensive systematic scenario demonstration.
        
        Args:
            num_runs: Number of runs to execute (uses first N scenarios)
            
        Returns:
            Summary results dictionary
        """
        logger.info(f"Starting comprehensive systematic scenario demo - {num_runs} runs")
        
        # Generate systematic scenario set
        scenario_set = self.intruder_generator.generate_scenario_set("comprehensive_demo")
        logger.info(f"Generated {len(scenario_set.scenarios)} systematic scenarios")
        
        # Save scenario set for reproducibility
        scenario_file = self.output_dir / "comprehensive_scenarios.json"
        self.intruder_generator.save_scenario_set(scenario_set, scenario_file)
        
        # Execute scenarios
        results = []
        scenarios_to_run = scenario_set.scenarios[:num_runs]
        
        for i, scenario_params in enumerate(scenarios_to_run):
            logger.info(f"Running scenario {i+1}/{len(scenarios_to_run)}: "
                       f"{scenario_params.pattern.value} - {scenario_params.severity.value}")
            
            result = self._run_single_scenario(scenario_params)
            results.append(result)
            
            # Brief pause between scenarios
            time.sleep(0.1)
        
        # Generate final reports
        summary = self._generate_demo_summary(results)
        self._save_demo_results(results, summary)
        
        logger.info(f"Demo completed: {summary['successful_scenarios']}/{summary['total_scenarios']} successful")
        return summary
    
    def run_custom_scenarios(self, patterns: List[str], severities: List[str], 
                           cpa_times: List[float], seeds: List[int]) -> Dict[str, Any]:
        """Run custom-defined scenarios.
        
        Args:
            patterns: List of conflict patterns
            severities: List of severity levels  
            cpa_times: List of CPA times in minutes
            seeds: List of seeds for reproducibility
            
        Returns:
            Summary results dictionary
        """
        logger.info(f"Running {len(patterns)} custom scenarios")
        
        # Create custom scenario set
        scenario_set = SystematicScenarioSet(
            name="custom_scenarios",
            description="User-defined custom scenarios"
        )
        
        for i, (pattern_str, severity_str, cpa_time, seed) in enumerate(
            zip(patterns, severities, cpa_times, seeds)):
            
            pattern = ConflictPattern(pattern_str)
            severity = ConflictSeverity(severity_str)
            
            # Get distance from severity
            _, cpa_distance = self._get_severity_parameters(severity)
            
            scenario_set.add_scenario(
                pattern=pattern,
                severity=severity,
                cpa_time_min=cpa_time,
                cpa_distance_nm=cpa_distance,
                seed=seed
            )
        
        # Save custom scenarios
        scenario_file = self.output_dir / "custom_scenarios.json"
        self.intruder_generator.save_scenario_set(scenario_set, scenario_file)
        
        # Execute scenarios
        results = []
        for i, scenario_params in enumerate(scenario_set.scenarios):
            logger.info(f"Running custom scenario {i+1}: "
                       f"{scenario_params.pattern.value} - CPA {scenario_params.cpa_time_min}min")
            
            result = self._run_single_scenario(scenario_params)
            results.append(result)
        
        # Generate results
        summary = self._generate_demo_summary(results)
        self._save_demo_results(results, summary, "custom")
        
        return summary
    
    def _run_single_scenario(self, scenario_params: ScenarioParameters) -> Dict[str, Any]:
        """Run a single scenario and collect metrics.
        
        Args:
            scenario_params: Scenario parameters
            
        Returns:
            Scenario result dictionary
        """
        start_time = time.time()
        
        # Generate intruder for scenario
        intruder = self.intruder_generator.generate_intruder_for_scenario(
            self.ownship_state, scenario_params
        )
        
        # Start metrics collection
        scenario_dict = {
            "scenario_id": f"{scenario_params.pattern.value}_{scenario_params.severity.value}",
            "pattern": scenario_params.pattern.value,
            "seed": scenario_params.seed,
            "initial_separation": haversine_nm(
                (self.ownship_state.latitude, self.ownship_state.longitude),
                (intruder.latitude, intruder.longitude)
            ),
            "closure_rate": abs(self.ownship_state.ground_speed_kt - intruder.ground_speed_kt)
        }
        
        run_id = self.metrics_collector.start_run(
            self.ownship_state.aircraft_id,
            intruder.aircraft_id,
            scenario_dict
        )
        
        # Simulate conflict detection
        detection_start = time.time()
        cpa_result = calculate_enhanced_cpa(
            self.ownship_state, intruder, 
            look_ahead_min=15.0
        )
        detection_time = (time.time() - detection_start) * 1000  # milliseconds
        
        # Determine conflict type
        is_conflict = cpa_result.horizontal_separation_nm < 5.0
        expected_conflict = scenario_params.cpa_distance_nm < 5.0
        
        if is_conflict and expected_conflict:
            conflict_type = ConflictType.TRUE_POSITIVE
        elif is_conflict and not expected_conflict:
            conflict_type = ConflictType.FALSE_POSITIVE
        elif not is_conflict and expected_conflict:
            conflict_type = ConflictType.FALSE_NEGATIVE
        else:
            conflict_type = ConflictType.TRUE_NEGATIVE
        
        # Record detection metrics
        self.metrics_collector.record_detection(
            run_id=run_id,
            detection_time=detection_time / 1000.0,  # Convert to seconds
            cpa_time=cpa_result.time_to_cpa_min,
            cpa_distance=cpa_result.horizontal_separation_nm,
            confidence=cpa_result.confidence_score,
            conflict_type=conflict_type
        )
        
        # Simulate resolution if conflict detected
        resolution_success = True
        min_separation = cpa_result.horizontal_separation_nm
        
        if is_conflict:
            # Simulate resolution action
            resolution_start = time.time()
            
            # Simple resolution: turn away from intruder
            maneuver_type = "HDG"
            deviation_amount = 20.0  # 20 degree turn
            
            # Simulate resolution effectiveness
            if scenario_params.severity in [ConflictSeverity.LOW, ConflictSeverity.MEDIUM]:
                resolution_success = True
                min_separation = 5.5  # Achieved safe separation
            else:
                resolution_success = scenario_params.cpa_distance_nm > 0.5
                min_separation = max(scenario_params.cpa_distance_nm, 1.0)
            
            resolution_time = (time.time() - resolution_start) * 1000
            
            self.metrics_collector.record_resolution(
                run_id=run_id,
                action_time=resolution_time / 1000.0,
                maneuver_type=maneuver_type,
                deviation_amount=deviation_amount,
                success=resolution_success,
                intrusion=max(0, 5.0 - min_separation)  # Intrusion into protected zone
            )
        
        # Record system performance
        total_processing_time = (time.time() - start_time) * 1000
        self.metrics_collector.record_system_performance(
            run_id=run_id,
            processing_time=total_processing_time
        )
        
        # Record final outcome
        safety_maintained = min_separation >= 5.0
        self.metrics_collector.record_final_outcome(
            run_id=run_id,
            min_separation=min_separation,
            conflict_resolved=resolution_success if is_conflict else True,
            safety_maintained=safety_maintained
        )
        
        # Complete the run
        run_metrics = self.metrics_collector.complete_run(run_id)
        
        return {
            "scenario_params": scenario_params,
            "intruder_state": intruder,
            "cpa_result": cpa_result,
            "run_metrics": run_metrics,
            "conflict_type": conflict_type,
            "min_separation": min_separation,
            "resolution_success": resolution_success
        }
    
    def _get_severity_parameters(self, severity: ConflictSeverity) -> tuple:
        """Get CPA time and distance for severity level."""
        if severity == ConflictSeverity.LOW:
            return 10.0, 4.0
        elif severity == ConflictSeverity.MEDIUM:
            return 6.0, 2.0
        elif severity == ConflictSeverity.HIGH:
            return 3.0, 0.8
        else:  # CRITICAL
            return 1.5, 0.3
    
    def _generate_demo_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of demo results.
        
        Args:
            results: List of scenario results
            
        Returns:
            Summary dictionary
        """
        # Get session summary from metrics collector
        session_summary = self.metrics_collector.get_session_summary()
        
        # Pattern-specific analysis
        pattern_stats = {}
        for result in results:
            pattern = result["scenario_params"].pattern.value
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    "count": 0,
                    "successful": 0,
                    "avg_min_separation": 0.0,
                    "detection_accuracy": 0.0
                }
            
            stats = pattern_stats[pattern]
            stats["count"] += 1
            stats["successful"] += 1 if result["resolution_success"] else 0
            stats["avg_min_separation"] += result["min_separation"]
            
            # Detection accuracy
            if result["conflict_type"] in [ConflictType.TRUE_POSITIVE, ConflictType.TRUE_NEGATIVE]:
                stats["detection_accuracy"] += 1.0
        
        # Calculate averages
        for pattern, stats in pattern_stats.items():
            stats["success_rate"] = stats["successful"] / stats["count"]
            stats["avg_min_separation"] /= stats["count"]
            stats["detection_accuracy"] /= stats["count"]
        
        return {
            "demo_timestamp": datetime.now().isoformat(),
            "total_scenarios": len(results),
            "successful_scenarios": sum(1 for r in results if r["resolution_success"]),
            "pattern_statistics": pattern_stats,
            "session_metrics": session_summary,
            "ownship_config": {
                "aircraft_id": self.ownship_state.aircraft_id,
                "initial_position": (self.ownship_state.latitude, self.ownship_state.longitude),
                "altitude_ft": self.ownship_state.altitude_ft,
                "speed_kt": self.ownship_state.ground_speed_kt,
                "heading_deg": self.ownship_state.heading_deg
            }
        }
    
    def _save_demo_results(self, results: List[Dict[str, Any]], 
                          summary: Dict[str, Any], prefix: str = "comprehensive") -> None:
        """Save demo results to files.
        
        Args:
            results: List of scenario results
            summary: Summary dictionary
            prefix: Filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"{prefix}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects to dicts
            serializable_results = []
            for result in results:
                serializable_result = {
                    "scenario_params": {
                        "pattern": result["scenario_params"].pattern.value,
                        "severity": result["scenario_params"].severity.value,
                        "cpa_time_min": result["scenario_params"].cpa_time_min,
                        "cpa_distance_nm": result["scenario_params"].cpa_distance_nm,
                        "seed": result["scenario_params"].seed
                    },
                    "conflict_type": result["conflict_type"].value,
                    "min_separation": result["min_separation"],
                    "resolution_success": result["resolution_success"],
                    "cpa_result": {
                        "time_to_cpa_min": result["cpa_result"].time_to_cpa_min,
                        "horizontal_separation_nm": result["cpa_result"].horizontal_separation_nm,
                        "vertical_separation_ft": result["cpa_result"].vertical_separation_ft,
                        "confidence_score": result["cpa_result"].confidence_score
                    }
                }
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / f"{prefix}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = self.metrics_collector.save_session_metrics(
            f"{prefix}_metrics_{timestamp}.json"
        )
        
        logger.info(f"Demo results saved:")
        logger.info(f"  Detailed results: {results_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Metrics: {metrics_file}")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Systematic Intruder Generation Demo")
    parser.add_argument("--scenarios", choices=["comprehensive", "custom"], 
                       default="comprehensive", help="Type of scenarios to run")
    parser.add_argument("--runs", type=int, default=15, 
                       help="Number of runs for comprehensive mode")
    parser.add_argument("--seed", type=int, default=12345, 
                       help="Base seed for reproducibility")
    parser.add_argument("--output", type=str, default="demo_output",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize demo
    demo = SystematicScenarioDemo(Path(args.output))
    demo.intruder_generator.base_seed = args.seed
    
    try:
        if args.scenarios == "comprehensive":
            summary = demo.run_comprehensive_demo(args.runs)
        else:
            # Custom scenarios example
            patterns = ["crossing", "head_on", "overtake"]
            severities = ["medium", "high", "medium"]
            cpa_times = [6.0, 3.0, 8.0]
            seeds = [args.seed + i for i in range(3)]
            
            summary = demo.run_custom_scenarios(patterns, severities, cpa_times, seeds)
        
        # Print summary
        print("\\n" + "="*60)
        print("SYSTEMATIC SCENARIO DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Total scenarios: {summary['total_scenarios']}")
        print(f"Successful scenarios: {summary['successful_scenarios']}")
        print(f"Success rate: {summary['successful_scenarios']/summary['total_scenarios']*100:.1f}%")
        
        if 'session_metrics' in summary:
            metrics = summary['session_metrics']['wolfgang_2011_kpis']
            print(f"\\nWolfgang (2011) KPIs:")
            print(f"  TBAS (avg): {metrics['tbas_avg']:.3f}")
            print(f"  LAT (avg): {metrics['lat_avg_min']:.2f} min")
            print(f"  DAT (avg): {metrics['dat_avg_min']:.2f} min")
            print(f"  DFA (avg): {metrics['dfa_avg']:.3f}")
            print(f"  RE (avg): {metrics['re_avg']:.3f}")
            print(f"  RI (avg): {metrics['ri_avg_nm']:.2f} NM")
            print(f"  RAT (avg): {metrics['rat_avg_sec']:.2f} sec")
        
        print(f"\\nOutput directory: {args.output}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
