"""Main conflict detection and resolution pipeline.

This module implements the core 5-minute polling loop that:
- Fetches current aircraft states from BlueSky
- Runs conflict detection for 10-minute horizon  
- Generates resolutions using LLM reasoning
- Validates and executes safe resolution commands
- Logs all decisions and maintains execution state
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json

from .bluesky_io import BlueSkyClient
from .detect import predict_conflicts
from .resolve import generate_horizontal_resolution, generate_vertical_resolution, validate_resolution
from .llm_client import LlamaClient
from .metrics import MetricsCollector
from .schemas import AircraftState, ConflictPrediction, ResolutionCommand, ConfigurationSettings

logger = logging.getLogger(__name__)


class CDRPipeline:
    """Main conflict detection and resolution pipeline."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize CDR pipeline with configuration.
        
        Args:
            config: System configuration parameters
        """
        self.config = config
        self.running = False
        self.cycle_count = 0
        
        # Initialize components
        self.bluesky_client = BlueSkyClient(config)
        self.llm_client = LlamaClient(config)
        self.metrics = MetricsCollector()
        
        # State tracking
        self.active_resolutions: Dict[str, ResolutionCommand] = {}
        self.conflict_history: List[ConflictPrediction] = []
        
        logger.info("CDR Pipeline initialized")
    
    def run(self, max_cycles: Optional[int] = None, ownship_id: str = "OWNSHIP") -> None:
        """Run the main CDR loop.
        
        Args:
            max_cycles: Maximum cycles to run (None = infinite)
            ownship_id: Ownship aircraft identifier
        """
        logger.info(f"Starting CDR pipeline for ownship: {ownship_id}")
        self.running = True
        
        try:
            while self.running and (max_cycles is None or self.cycle_count < max_cycles):
                cycle_start = datetime.now()
                
                # Execute one pipeline cycle
                self._execute_cycle(ownship_id)
                
                # Calculate sleep time for next cycle
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.config.polling_interval_min * 60 - cycle_duration)
                
                logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f}s, "
                           f"sleeping {sleep_time:.2f}s")
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.cycle_count += 1
                
        except KeyboardInterrupt:
            logger.info("CDR pipeline stopped by user")
        except Exception as e:
            logger.error(f"CDR pipeline error: {e}")
            raise
        finally:
            self.running = False
            self._cleanup()
    
    def _execute_cycle(self, ownship_id: str) -> None:
        """Execute one complete CDR cycle.
        
        Args:
            ownship_id: Ownship aircraft identifier
        """
        logger.debug(f"Starting cycle {self.cycle_count}")
        
        # Step 1: Fetch current aircraft states
        aircraft_states = self._fetch_aircraft_states()
        if not aircraft_states:
            logger.warning("No aircraft states available, skipping cycle")
            return
        
        # Step 2: Identify ownship and traffic
        ownship = self._find_ownship(aircraft_states, ownship_id)
        if not ownship:
            logger.warning(f"Ownship {ownship_id} not found, skipping cycle")
            return
        
        traffic = [ac for ac in aircraft_states if ac.aircraft_id != ownship_id]
        logger.info(f"Processing ownship {ownship_id} with {len(traffic)} traffic aircraft")
        
        # Step 3: Predict conflicts
        conflicts = self._predict_conflicts(ownship, traffic)
        logger.info(f"Detected {len(conflicts)} potential conflicts")
        
        # Step 4: Generate and execute resolutions
        for conflict in conflicts:
            if conflict.is_conflict:
                self._handle_conflict(conflict, ownship, traffic)
        
        # Step 5: Update metrics
        self._update_metrics(ownship, traffic, conflicts)
    
    def _fetch_aircraft_states(self) -> List[AircraftState]:
        """Fetch current aircraft states from BlueSky.
        
        Returns:
            List of current aircraft states
        """
        logger.debug("Fetching aircraft states from BlueSky")
        if not self.bluesky_client.connected:
            if not self.bluesky_client.connect():
                logger.warning("Failed to connect to BlueSky")
                return []
        
        return self.bluesky_client.get_aircraft_states()
    
    def _find_ownship(self, aircraft_states: List[AircraftState], ownship_id: str) -> Optional[AircraftState]:
        """Find ownship in aircraft states list.
        
        Args:
            aircraft_states: List of all aircraft states
            ownship_id: Ownship identifier
            
        Returns:
            Ownship state or None if not found
        """
        for aircraft in aircraft_states:
            if aircraft.aircraft_id == ownship_id:
                return aircraft
        return None
    
    def _predict_conflicts(self, ownship: AircraftState, traffic: List[AircraftState]) -> List[ConflictPrediction]:
        """Predict conflicts using deterministic algorithms.
        
        Args:
            ownship: Ownship current state
            traffic: Traffic aircraft states
            
        Returns:
            List of predicted conflicts
        """
        # TODO: Implement in Sprint 1
        logger.debug(f"Predicting conflicts for {len(traffic)} traffic aircraft")
        return []
    
    def _handle_conflict(
        self, 
        conflict: ConflictPrediction, 
        ownship: AircraftState, 
        traffic: List[AircraftState]
    ) -> None:
        """Handle detected conflict by generating and executing resolution.
        
        Args:
            conflict: Predicted conflict details
            ownship: Current ownship state
            traffic: Current traffic states
        """
        logger.warning(f"Handling conflict with {conflict.intruder_id}, "
                      f"CPA in {conflict.time_to_cpa_min:.1f} min, "
                      f"separation {conflict.distance_at_cpa_nm:.2f} NM")
        
        # Generate resolution options
        resolution = self._generate_resolution(conflict, ownship, traffic)
        
        if resolution and self._validate_and_execute_resolution(resolution, ownship, traffic):
            logger.info(f"Successfully executed resolution {resolution.resolution_id}")
            self.active_resolutions[resolution.resolution_id] = resolution
        else:
            logger.error(f"Failed to resolve conflict with {conflict.intruder_id}")
    
    def _generate_resolution(
        self, 
        conflict: ConflictPrediction, 
        ownship: AircraftState, 
        traffic: List[AircraftState]
    ) -> Optional[ResolutionCommand]:
        """Generate conflict resolution using available methods.
        
        Args:
            conflict: Conflict to resolve
            ownship: Current ownship state
            traffic: Current traffic states
            
        Returns:
            Resolution command or None
        """
        # TODO: Implement LLM-based resolution in Sprint 3
        # For now, use deterministic algorithms
        
        # Try horizontal resolution first
        resolution = generate_horizontal_resolution(conflict, ownship)
        if resolution:
            return resolution
        
        # Fallback to vertical resolution
        return generate_vertical_resolution(conflict, ownship)
    
    def _validate_and_execute_resolution(
        self, 
        resolution: ResolutionCommand, 
        ownship: AircraftState, 
        traffic: List[AircraftState]
    ) -> bool:
        """Validate and execute resolution command.
        
        Args:
            resolution: Resolution to validate and execute
            ownship: Current ownship state
            traffic: Current traffic states
            
        Returns:
            True if successfully executed
        """
        # Validate resolution safety
        if not validate_resolution(resolution, ownship, traffic):
            logger.error(f"Resolution {resolution.resolution_id} failed safety validation")
            return False
        
        # Execute via BlueSky
        success = self.bluesky_client.execute_command(resolution)
        if success:
            logger.info(f"Executed resolution: {resolution.resolution_type.value}")
            return True
        else:
            logger.error(f"Failed to execute resolution {resolution.resolution_id}")
            return False
    
    def _update_metrics(
        self, 
        ownship: AircraftState, 
        traffic: List[AircraftState], 
        conflicts: List[ConflictPrediction]
    ) -> None:
        """Update performance metrics.
        
        Args:
            ownship: Current ownship state
            traffic: Current traffic states
            conflicts: Detected conflicts
        """
        # TODO: Implement in Sprint 4
        pass
    
    def _cleanup(self) -> None:
        """Clean up resources and save final state."""
        logger.info("Cleaning up CDR pipeline")
        
        # Save metrics
        self.metrics.save_report(f"reports/sprint_0/cycle_{self.cycle_count}_metrics.json")
        
        # Close connections
        if hasattr(self.bluesky_client, 'close'):
            self.bluesky_client.close()
    
    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        logger.info("Stopping CDR pipeline")
        self.running = False


def main():
    """Main entry point for CDR pipeline."""
    # TODO: Add command-line argument parsing
    config = ConfigurationSettings()
    pipeline = CDRPipeline(config)
    
    try:
        pipeline.run(ownship_id="OWNSHIP")
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
