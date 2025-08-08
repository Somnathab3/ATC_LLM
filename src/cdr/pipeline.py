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
from datetime import datetime
from typing import List, Optional, Dict, Any

from .bluesky_io import BlueSkyClient
from .detect import predict_conflicts
from .resolve import generate_horizontal_resolution, generate_vertical_resolution, validate_resolution
from .llm_client import LlamaClient
from .metrics import MetricsCollector
from .schemas import AircraftState, ConflictPrediction, ResolutionCommand, ResolutionType, ConfigurationSettings

logger = logging.getLogger(__name__)


def _asdict_state(s: Any) -> Dict[str, Any]:
    """Convert any state object to dict for backward compatibility."""
    if isinstance(s, dict):
        return s
    if hasattr(s, "model_dump"):  # Pydantic v2
        return s.model_dump()
    if hasattr(s, "dict"):        # Pydantic v1
        return s.dict()
    if hasattr(s, "__dict__"):    # dataclass / simple object
        return dict(vars(s))
    raise TypeError(f"Unsupported state type: {type(s)}")


def _dict_to_aircraft_state(state_dict: Dict[str, Any]) -> AircraftState:
    """Convert BlueSky state dict to AircraftState object."""
    return AircraftState(
        aircraft_id=state_dict["id"],
        timestamp=datetime.now(),
        latitude=state_dict["lat"],
        longitude=state_dict["lon"],
        altitude_ft=state_dict["alt_ft"],
        ground_speed_kt=state_dict["spd_kt"],
        heading_deg=state_dict["hdg_deg"],
        vertical_speed_fpm=state_dict["roc_fpm"]
    )


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
        
        # Aliases for compatibility
        self.bs = self.bluesky_client
        self.log = logger
        
        # State tracking
        self.active_resolutions: Dict[str, ResolutionCommand] = {}
        self.conflict_history: List[ConflictPrediction] = []
        
        logger.info("CDR Pipeline initialized")
    
    def run(self, max_cycles: Optional[int] = None, ownship_id: str = "OWNSHIP") -> bool:
        """Run the main CDR loop.
        
        Args:
            max_cycles: Maximum cycles to run (None = infinite)
            ownship_id: Ownship aircraft identifier
            
        Returns:
            True if completed successfully
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
                
            # Return True if successfully completed a full 5-minute cycle
            return self.cycle_count > 0
                
        except KeyboardInterrupt:
            logger.info("CDR pipeline stopped by user")
            return False
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
        
        # Step 1: Fetch current aircraft states and split into ownship/traffic
        ownship, traffic = self._fetch_aircraft_states(ownship_id)
        if ownship is None:
            logger.warning(f"Ownship {ownship_id} not found, skipping cycle")
            return
        
        logger.info(f"Processing ownship {ownship_id} with {len(traffic)} traffic aircraft")
        
        # Step 2: Predict conflicts
        conflicts = self._predict_conflicts(ownship, traffic)
        logger.info(f"Detected {len(conflicts)} potential conflicts")
        
        # Step 3: Generate and execute resolutions
        for conflict in conflicts:
            if conflict.is_conflict:
                self._handle_conflict(conflict, ownship, traffic)
        
        # Step 4: Update metrics
        self._update_metrics(ownship, traffic, conflicts)
    
    def _fetch_aircraft_states(self, ownship_id: str):
        """Fetch current aircraft states from BlueSky and split into ownship/traffic.
        
        Args:
            ownship_id: Ownship aircraft identifier
            
        Returns:
            Tuple of (ownship_state, traffic_states) where ownship_state is dict or None,
            and traffic_states is list of dicts
        """
        raw = self.bs.get_aircraft_states()
        own = next((r for r in raw if r["id"] == ownship_id), None)
        traffic = [r for r in raw if r["id"] != ownship_id]
        if own is None:
            self.log.warning("Ownship %s not found in BS state", ownship_id)
        return own, traffic
    
    def _find_ownship(self, states: List[Dict[str, Any]], ownship_id: str) -> Optional[Dict[str, Any]]:
        """Find ownship in states list."""
        for s in states:
            d = _asdict_state(s)
            if d.get("id") == ownship_id:
                return d
        return None
    

    def _predict_conflicts(self, own: Optional[Dict[str, Any]], traffic: List[Dict[str, Any]]) -> List[ConflictPrediction]:
        """Predict conflicts using deterministic algorithms.
        
        Args:
            own: Ownship current state dict
            traffic: Traffic aircraft state dicts
            
        Returns:
            List of predicted conflicts
        """
        if own is None:
            logger.warning("Cannot predict conflicts: ownship state is None")
            return []
            
        try:
            # Convert dict states to AircraftState objects
            own_s = _dict_to_aircraft_state(own)
            traf_s = [_dict_to_aircraft_state(t) for t in traffic]
            
            # Use the conflict detection algorithm
            conflicts = predict_conflicts(own_s, traf_s, lookahead_minutes=self.config.lookahead_time_min)
            
            logger.debug(f"Predicted {len(conflicts)} conflicts for {len(traffic)} traffic aircraft")
            return conflicts
            
        except Exception as e:
            logger.exception(f"Error in conflict prediction: {e}")
            return []
    
    def _handle_conflict(
        self, 
        conflict: ConflictPrediction, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]]
    ) -> None:
        """Handle detected conflict by generating and executing resolution.
        
        Args:
            conflict: Predicted conflict details
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
        """
        logger.warning(f"Handling conflict with {conflict.intruder_id}, "
                      f"CPA in {conflict.time_to_cpa_min:.1f} min, "
                      f"separation {conflict.distance_at_cpa_nm:.2f} NM")
        
        # Build a compact prompt for the LLM (ownship + intruder + next 10 min task)
        prompt = self._llm_client_prompt_builder(conflict, ownship, traffic)
        
        # Ask LLM for action
        llm_json = self.llm_client.generate_resolution(prompt)
        
        # Turn JSON into a ResolutionCommand (with your existing create/validate)
        # Then push via BlueSky stack (HDG/ALT), and record metrics.
        resolution = self._create_resolution_from_llm(llm_json, conflict, ownship)
        
        if resolution and self._validate_and_execute_resolution(resolution, ownship, traffic):
            logger.info(f"Successfully executed resolution {resolution.resolution_id}")
            self.active_resolutions[resolution.resolution_id] = resolution
        else:
            logger.error(f"Failed to resolve conflict with {conflict.intruder_id}")

    def _llm_client_prompt_builder(self, conflict: ConflictPrediction, ownship: Dict[str, Any], traffic: List[Dict[str, Any]]) -> str:
        """Build a compact prompt for the LLM with ownship + intruder + next 10 min task.
        
        Args:
            conflict: Predicted conflict details
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            
        Returns:
            Formatted prompt string for LLM
        """
        # Find the intruder aircraft
        intruder = next((t for t in traffic if t["id"] == conflict.intruder_id), None)
        if intruder is None:
            logger.warning(f"Intruder {conflict.intruder_id} not found in traffic list")
            intruder = {"id": conflict.intruder_id, "lat": 0, "lon": 0, "alt_ft": 0, "hdg_deg": 0, "spd_kt": 0}
        
        prompt = f"""
Air Traffic Control Conflict Resolution Task:

OWNSHIP: {ownship['id']}
- Position: ({ownship['lat']:.6f}, {ownship['lon']:.6f})
- Altitude: {ownship['alt_ft']:.0f} ft
- Heading: {ownship['hdg_deg']:.0f}°
- Speed: {ownship['spd_kt']:.0f} kts

INTRUDER: {intruder['id']}
- Position: ({intruder['lat']:.6f}, {intruder['lon']:.6f})
- Altitude: {intruder['alt_ft']:.0f} ft
- Heading: {intruder['hdg_deg']:.0f}°
- Speed: {intruder['spd_kt']:.0f} kts

CONFLICT PREDICTION:
- Time to CPA: {conflict.time_to_cpa_min:.1f} minutes
- Distance at CPA: {conflict.distance_at_cpa_nm:.2f} NM
- Altitude separation at CPA: {conflict.altitude_diff_ft:.0f} ft

TASK: Generate a conflict resolution command for the ownship. 
Prefer horizontal maneuvers (heading changes) over vertical maneuvers (altitude changes) as they are typically less intrusive.
Provide your resolution as JSON with fields: resolution_type (either "heading" or "altitude"), new_heading_deg (if heading change), new_altitude_ft (if altitude change), target_aircraft (ownship ID).
"""
        return prompt.strip()

    def _create_resolution_from_llm(self, llm_json: Dict[str, Any], conflict: ConflictPrediction, ownship: Dict[str, Any]) -> Optional[ResolutionCommand]:
        """Convert LLM JSON response to ResolutionCommand object.
        
        Args:
            llm_json: LLM response JSON
            conflict: Original conflict prediction
            ownship: Current ownship state dict
            
        Returns:
            ResolutionCommand object or None if invalid
        """
        try:
            from datetime import datetime
            
            # Extract resolution details from LLM response
            resolution_type = llm_json.get("resolution_type", "").lower()
            target_aircraft = llm_json.get("target_aircraft", ownship["id"])
            
            # Create resolution command based on type
            if resolution_type == "heading":
                new_heading = llm_json.get("new_heading_deg")
                if new_heading is None:
                    logger.error("LLM response missing new_heading_deg for heading resolution")
                    return None
                    
                return ResolutionCommand(
                    resolution_id=f"hdg_{target_aircraft}_{int(datetime.now().timestamp())}",
                    target_aircraft=target_aircraft,
                    resolution_type=ResolutionType.HEADING_CHANGE,
                    new_heading_deg=float(new_heading),
                    new_speed_kt=None,
                    new_altitude_ft=None,
                    issue_time=datetime.now(),
                    is_validated=False,
                    safety_margin_nm=5.0  # Default safety margin
                )
                
            elif resolution_type == "altitude":
                new_altitude = llm_json.get("new_altitude_ft")
                if new_altitude is None:
                    logger.error("LLM response missing new_altitude_ft for altitude resolution")
                    return None
                    
                return ResolutionCommand(
                    resolution_id=f"alt_{target_aircraft}_{int(datetime.now().timestamp())}",
                    target_aircraft=target_aircraft,
                    resolution_type=ResolutionType.ALTITUDE_CHANGE,
                    new_heading_deg=None,
                    new_speed_kt=None,
                    new_altitude_ft=float(new_altitude),
                    issue_time=datetime.now(),
                    is_validated=False,
                    safety_margin_nm=5.0  # Default safety margin
                )
                
            else:
                logger.error(f"Unknown resolution type from LLM: {resolution_type}")
                return None
                
        except Exception as e:
            logger.exception(f"Error creating resolution from LLM response: {e}")
            return None
    
    def _generate_resolution(
        self, 
        conflict: ConflictPrediction, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]]
    ) -> Optional[ResolutionCommand]:
        """Generate conflict resolution using available methods.
        
        Args:
            conflict: Conflict to resolve
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            
        Returns:
            Resolution command or None
        """
        # TODO: Implement LLM-based resolution in Sprint 3
        # For now, use deterministic algorithms
        
        try:
            # Convert dicts to AircraftState for legacy resolution methods
            ownship_state = _dict_to_aircraft_state(ownship)
            
            # Try horizontal resolution first
            resolution = generate_horizontal_resolution(conflict, ownship_state)
            if resolution:
                return resolution
            
            # Fallback to vertical resolution
            return generate_vertical_resolution(conflict, ownship_state)
            
        except Exception as e:
            logger.exception(f"Error in fallback resolution generation: {e}")
            return None
    
    def _validate_and_execute_resolution(
        self, 
        resolution: ResolutionCommand, 
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]]
    ) -> bool:
        """Validate and execute resolution command.
        
        Args:
            resolution: Resolution to validate and execute
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
            
        Returns:
            True if successfully executed
        """
        # Convert dicts to AircraftState for validation
        try:
            ownship_state = _dict_to_aircraft_state(ownship)
            traffic_states = [_dict_to_aircraft_state(t) for t in traffic]
            
            # Validate resolution safety
            if not validate_resolution(resolution, ownship_state, traffic_states):
                logger.error(f"Resolution {resolution.resolution_id} failed safety validation")
                return False
        except Exception as e:
            logger.exception(f"Error converting states for validation: {e}")
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
        ownship: Dict[str, Any], 
        traffic: List[Dict[str, Any]], 
        conflicts: List[ConflictPrediction]
    ) -> None:
        """Update performance metrics.
        
        Args:
            ownship: Current ownship state dict
            traffic: Current traffic state dicts
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
    config = ConfigurationSettings(
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1,
        llm_max_tokens=2048,
        safety_buffer_factor=1.2,
        max_resolution_angle_deg=45.0,
        max_altitude_change_ft=2000.0,
        bluesky_host="localhost",
        bluesky_port=1337,
        bluesky_timeout_sec=5.0
    )
    pipeline = CDRPipeline(config)
    
    try:
        pipeline.run(ownship_id="OWNSHIP")
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
