"""
Enhanced SCAT simulation with batch flight processing and Monte Carlo intruder generation.

This script implements:
1. Batch flight input support from multiple SCAT files [OK]
2. Monte Carlo intruder generation using KDTree-based spatial search [OK]
3. LLM-based conflict detection and resolution [OK]
4. Comprehensive metrics collection and analysis [OK]
5. Integration with existing pipeline infrastructure [OK]
"""

import sys
from pathlib import Path

# Add the parent directory to Python path so we can import src modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import logging
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import glob
import os

from src.cdr.pipeline import CDRPipeline
from src.cdr.schemas import (
    ConfigurationSettings, FlightRecord, MonteCarloParameters, 
    BatchSimulationResult
)
from src.cdr.scat_adapter import SCATAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_scat_simulation")


class SCATFlightLoader:
    """Loads multiple SCAT flight records for batch processing."""
    
    def __init__(self, scat_directory: str):
        self.scat_directory = Path(scat_directory)
        # Note: Pass dummy path since we'll load individual files directly
        self.scat_adapter = SCATAdapter(str(self.scat_directory))
    
    def load_flight_from_file(self, scat_file_path: str) -> Optional[FlightRecord]:
        """Load a single flight record from SCAT file."""
        try:
            # Load SCAT flight record using the adapter
            flight_record = self.scat_adapter.load_flight_record(Path(scat_file_path))
            
            if not flight_record:
                logger.warning(f"Failed to parse SCAT file {scat_file_path}")
                return None
            
            # Extract aircraft states
            aircraft_states = self.scat_adapter.extract_aircraft_states(flight_record)
            
            if not aircraft_states:
                logger.warning(f"No aircraft states found in {scat_file_path}")
                return None
            
            # Sort by timestamp
            aircraft_states.sort(key=lambda s: s.timestamp)
            
            # Build flight record from SCAT data
            waypoints = [(state.latitude, state.longitude) for state in aircraft_states]
            altitudes = [state.altitude_ft for state in aircraft_states]
            timestamps = [state.timestamp for state in aircraft_states]
            
            # Use SCAT data for aircraft info
            aircraft_type = flight_record.aircraft_type or "B737"
            flight_id = flight_record.callsign or Path(scat_file_path).stem
            
            # Calculate cruise speed from track data
            if len(aircraft_states) > 1:
                total_speed = sum(state.ground_speed_kt for state in aircraft_states)
                cruise_speed = total_speed / len(aircraft_states)
            else:
                cruise_speed = 420.0
            
            # Determine scenario complexity based on flight characteristics
            complexity_level = self._assess_complexity(aircraft_states, waypoints)
            
            flight_record = FlightRecord(
                flight_id=flight_id,
                callsign=aircraft_states[0].callsign or flight_id,
                aircraft_type=aircraft_type,
                waypoints=waypoints,
                altitudes_ft=altitudes,
                timestamps=timestamps,
                cruise_speed_kt=cruise_speed,
                climb_rate_fpm=2000.0,
                descent_rate_fpm=-1500.0,
                scenario_type="scat_real_data",
                complexity_level=complexity_level
            )
            
            logger.info(f"Loaded flight {flight_id} with {len(waypoints)} waypoints, complexity {complexity_level}")
            return flight_record
            
        except Exception as e:
            logger.error(f"Error loading SCAT file {scat_file_path}: {e}")
            return None
    
    def _assess_complexity(self, aircraft_states: List[Any], waypoints: List[Any]) -> int:
        """Assess scenario complexity on a scale of 1-5."""
        complexity_score = 1
        
        # Factor 1: Number of waypoints
        if len(waypoints) > 20:
            complexity_score += 1
        if len(waypoints) > 50:
            complexity_score += 1
        
        # Factor 2: Altitude variations
        if len(aircraft_states) > 1:
            alt_changes = sum(1 for i in range(len(aircraft_states)-1)
                            if abs(aircraft_states[i+1].altitude_ft - aircraft_states[i].altitude_ft) > 2000)
            if alt_changes > 2:
                complexity_score += 1
        
        # Factor 3: Path irregularity (heading changes)
        if len(waypoints) > 2:
            from src.cdr.monte_carlo_intruders import bearing_deg
            heading_changes = 0
            for i in range(len(waypoints) - 2):
                bearing1 = bearing_deg(waypoints[i][0], waypoints[i][1], 
                                     waypoints[i+1][0], waypoints[i+1][1])
                bearing2 = bearing_deg(waypoints[i+1][0], waypoints[i+1][1], 
                                     waypoints[i+2][0], waypoints[i+2][1])
                heading_change = abs(bearing2 - bearing1)
                if heading_change > 180:
                    heading_change = 360 - heading_change
                if heading_change > 30:  # Significant course change
                    heading_changes += 1
            
            if heading_changes > 3:
                complexity_score += 1
        
        return min(complexity_score, 5)
    
    def load_multiple_flights(self, file_pattern: str = "*.json", 
                            max_flights: Optional[int] = None) -> List[FlightRecord]:
        """Load multiple flight records from SCAT directory."""
        pattern_path = self.scat_directory / file_pattern
        scat_files = glob.glob(str(pattern_path))
        
        if max_flights:
            scat_files = scat_files[:max_flights]
        
        logger.info(f"Loading {len(scat_files)} SCAT files from {self.scat_directory}")
        
        flight_records = []
        for scat_file in scat_files:
            flight_record = self.load_flight_from_file(scat_file)
            if flight_record:
                flight_records.append(flight_record)
        
        logger.info(f"Successfully loaded {len(flight_records)} flight records")
        return flight_records


class BatchSimulationRunner:
    """Coordinates batch simulation execution with Monte Carlo analysis."""
    
    def __init__(self, config: ConfigurationSettings):
        self.config = config
        self.pipeline = CDRPipeline(config)
        
    def run_batch_simulation(
        self,
        flight_records: List[FlightRecord],
        monte_carlo_params: MonteCarloParameters,
        output_dir: str = "batch_simulation_output"
    ) -> BatchSimulationResult:
        """Run complete batch simulation with analysis."""
        
        logger.info(f"Starting batch simulation with {len(flight_records)} flights")
        logger.info(f"Monte Carlo parameters: {monte_carlo_params.scenarios_per_flight} scenarios per flight")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Initialize pipeline
        if not self.pipeline.bluesky_client.connect():
            raise RuntimeError("Failed to connect to BlueSky simulator")
        
        try:
            # Run batch simulation
            batch_result = self.pipeline.run_for_flights(
                flight_records=flight_records,
                max_cycles=120,  # Max cycles per scenario
                monte_carlo_params=monte_carlo_params
            )
            
            # Save detailed results
            self._save_results(batch_result, output_path)
            
            # Generate analysis report
            self._generate_analysis_report(batch_result, output_path)
            
            logger.info(f"Batch simulation completed successfully")
            logger.info(f"Results saved to {output_path}")
            
            return batch_result
            
        finally:
            self.pipeline.stop()
    
    def _save_results(self, batch_result: BatchSimulationResult, output_path: Path):
        """Save simulation results to files."""
        
        # Save main results as JSON
        results_file = output_path / f"{batch_result.simulation_id}_results.json"
        with open(results_file, 'w') as f:
            # Convert to dict for JSON serialization
            result_dict = batch_result.model_dump()
            # Handle datetime serialization
            result_dict['start_time'] = batch_result.start_time.isoformat()
            if batch_result.end_time:
                result_dict['end_time'] = batch_result.end_time.isoformat()
            
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        # Save scenario results as CSV for easy analysis
        import pandas as pd
        
        if batch_result.scenario_results:
            df = pd.DataFrame(batch_result.scenario_results)
            csv_file = output_path / f"{batch_result.simulation_id}_scenarios.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved scenario data to {csv_file}")
        
        # Save flight-level summary
        if batch_result.flight_results:
            flight_df = pd.DataFrame([
                {'flight_id': fid, **metrics} 
                for fid, metrics in batch_result.flight_results.items()
            ])
            flight_csv = output_path / f"{batch_result.simulation_id}_flights.csv"
            flight_df.to_csv(flight_csv, index=False)
            logger.info(f"Saved flight summary to {flight_csv}")
    
    def _generate_analysis_report(self, batch_result: BatchSimulationResult, output_path: Path):
        """Generate comprehensive analysis report."""
        
        report_file = output_path / f"{batch_result.simulation_id}_analysis_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Batch Simulation Analysis Report\n\n")
            f.write(f"**Simulation ID:** {batch_result.simulation_id}\n")
            f.write(f"**Start Time:** {batch_result.start_time}\n")
            f.write(f"**End Time:** {batch_result.end_time}\n")
            f.write(f"**Duration:** {batch_result.end_time - batch_result.start_time if batch_result.end_time else 'N/A'}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Flights:** {len(batch_result.flight_records)}\n")
            f.write(f"- **Total Scenarios:** {batch_result.total_scenarios}\n")
            f.write(f"- **Scenarios per Flight:** {batch_result.scenarios_per_flight}\n")
            f.write(f"- **Conflicts Detected:** {batch_result.total_conflicts_detected}\n")
            f.write(f"- **Resolutions Attempted:** {batch_result.total_resolutions_attempted}\n")
            f.write(f"- **Successful Resolutions:** {batch_result.successful_resolutions}\n\n")
            
            f.write("## Performance Metrics\n\n")
            success_rate = (batch_result.successful_resolutions / max(1, batch_result.total_resolutions_attempted)) * 100
            f.write(f"- **Resolution Success Rate:** {success_rate:.1f}%\n")
            f.write(f"- **False Positive Rate:** {batch_result.false_positive_rate:.3f}\n")
            f.write(f"- **Average Resolution Time:** {batch_result.average_resolution_time_sec:.2f} seconds\n")
            f.write(f"- **Minimum Separation Achieved:** {batch_result.minimum_separation_achieved_nm:.2f} NM\n")
            f.write(f"- **Safety Violations:** {batch_result.safety_violations}\n\n")
            
            # Flight-level analysis
            if batch_result.flight_results:
                f.write("## Flight-Level Analysis\n\n")
                f.write("| Flight ID | Scenarios | Conflicts | Resolutions | Success Rate |\n")
                f.write("|-----------|-----------|-----------|-------------|---------------|\n")
                
                for flight_id, metrics in batch_result.flight_results.items():
                    scenarios = metrics.get('scenarios_processed', 0)
                    conflicts = metrics.get('conflicts_detected', 0)
                    resolutions = metrics.get('resolutions_attempted', 0)
                    successful = metrics.get('successful_resolutions', 0)
                    success_rate = (successful / max(1, resolutions)) * 100
                    
                    f.write(f"| {flight_id} | {scenarios} | {conflicts} | {resolutions} | {success_rate:.1f}% |\n")
            
            f.write("\n## Recommendations\n\n")
            
            if batch_result.false_positive_rate > 0.1:
                f.write("- **High False Positive Rate:** Consider tuning conflict detection sensitivity\n")
            
            if success_rate < 80:
                f.write("- **Low Resolution Success Rate:** Review LLM prompts and resolution validation\n")
            
            if batch_result.safety_violations > 0:
                f.write("- **Safety Violations Detected:** Immediate review of resolution algorithms required\n")
            
            if batch_result.average_resolution_time_sec > 10:
                f.write("- **Slow Resolution Times:** Consider LLM optimization or caching strategies\n")
        
        logger.info(f"Generated analysis report: {report_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch SCAT simulation with Monte Carlo analysis")
    
    parser.add_argument(
        "--scat-dir", 
        type=str, 
        default=r"F:\SCAT_extracted",
        help="Directory containing SCAT JSON files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_simulation_output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="Maximum number of flights to process"
    )
    
    parser.add_argument(
        "--scenarios-per-flight",
        type=int,
        default=10,
        help="Number of Monte Carlo scenarios per flight"
    )
    
    parser.add_argument(
        "--conflict-probability",
        type=float,
        default=0.3,
        help="Probability of generating conflict scenarios"
    )
    
    parser.add_argument(
        "--max-intruders",
        type=int,
        default=5,
        help="Maximum number of intruder aircraft per scenario"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for batch simulation."""
    
    args = parse_arguments()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting Enhanced SCAT Batch Simulation")
    logger.info(f"SCAT Directory: {args.scat_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Scenarios per Flight: {args.scenarios_per_flight}")
    
    # Initialize configuration
    config = ConfigurationSettings(
        polling_interval_min=1.0,  # Faster for simulation
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1,
        fast_time=True,
        sim_accel_factor=1.0
    )
    
    # Monte Carlo parameters
    monte_carlo_params = MonteCarloParameters(
        scenarios_per_flight=args.scenarios_per_flight,
        intruder_count_range=(1, args.max_intruders),
        conflict_zone_radius_nm=50.0,
        non_conflict_zone_radius_nm=200.0,
        altitude_spread_ft=10000.0,
        time_window_min=60.0,
        conflict_timing_variance_min=10.0,
        conflict_probability=args.conflict_probability,
        speed_variance_kt=50.0,
        heading_variance_deg=45.0,
        realistic_aircraft_types=True,
        airway_based_generation=False,
        weather_influence=False
    )
    
    try:
        # Load flight records
        flight_loader = SCATFlightLoader(args.scat_dir)
        flight_records = flight_loader.load_multiple_flights(
            max_flights=args.max_flights
        )
        
        if not flight_records:
            logger.error("No flight records loaded. Check SCAT directory and file format.")
            return 1
        
        # Run batch simulation
        simulation_runner = BatchSimulationRunner(config)
        batch_result = simulation_runner.run_batch_simulation(
            flight_records=flight_records,
            monte_carlo_params=monte_carlo_params,
            output_dir=args.output_dir
        )
        
        # Print summary
        logger.info("=== SIMULATION COMPLETED ===")
        logger.info(f"Processed {len(flight_records)} flights")
        logger.info(f"Generated {batch_result.total_scenarios} scenarios")
        logger.info(f"Detected {batch_result.total_conflicts_detected} conflicts")
        logger.info(f"Attempted {batch_result.total_resolutions_attempted} resolutions")
        logger.info(f"Success rate: {(batch_result.successful_resolutions/max(1,batch_result.total_resolutions_attempted))*100:.1f}%")
        logger.info(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
