"""BlueSky ASAS integration for baseline conflict detection and resolution.

This module provides:
- ASAS configuration and control
- Baseline conflict detection using BlueSky's built-in algorithms
- Comparison metrics between ASAS and LLM-based systems
- Standard aviation conflict resolution methods
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .schemas import AircraftState, ConflictPrediction, ConfigurationSettings
from .bluesky_io import BlueSkyClient

logger = logging.getLogger(__name__)

@dataclass
class ASASDetection:
    """ASAS conflict detection result."""
    aircraft_pair: Tuple[str, str]
    time_to_conflict_min: float
    distance_at_cpa_nm: float
    altitude_diff_ft: float
    conflict_severity: float
    detection_method: str = "ASAS"
    timestamp: datetime = None

@dataclass
class ASASResolution:
    """ASAS resolution command."""
    aircraft_id: str
    command_type: str  # "HDG", "ALT", "SPD"
    command_value: float
    reason: str
    timestamp: datetime = None
    success: bool = False

@dataclass
class ASASMetrics:
    """ASAS performance metrics for comparison."""
    total_conflicts_detected: int
    total_resolutions_attempted: int
    successful_resolutions: int
    false_positives: int
    missed_conflicts: int
    average_detection_time_sec: float
    average_resolution_time_sec: float
    resolution_success_rate: float

class BlueSkyASAS:
    """BlueSky ASAS (Airborne Separation Assurance System) integration."""
    
    def __init__(self, bluesky_client: BlueSkyClient, config: ConfigurationSettings):
        self.bluesky_client = bluesky_client
        self.config = config
        self.enabled = False
        self.detection_history: List[ASASDetection] = []
        self.resolution_history: List[ASASResolution] = []
        
    def configure_asas(self) -> bool:
        """Configure BlueSky ASAS with optimal settings."""
        try:
            # Enable ASAS
            if not self.bluesky_client.stack("ASAS ON"):
                logger.error("Failed to enable ASAS")
                return False
            
            # Set conflict detection method to geometric (fastest, most reliable)
            if not self.bluesky_client.stack("CDMETHOD GEOMETRIC"):
                logger.warning("Failed to set CDMETHOD - using default")
            
            # Set lookahead time (convert minutes to seconds)
            lookahead_sec = int(self.config.lookahead_time_min * 60)
            if not self.bluesky_client.stack(f"DTLOOK {lookahead_sec}"):
                logger.warning(f"Failed to set lookahead time to {lookahead_sec}s")
            
            # Set resolution method to vectorial (heading/speed changes)
            if not self.bluesky_client.stack("RESO VECTORIAL"):
                logger.warning("Failed to set resolution method")
            
            # Set horizontal separation zone (convert NM to meters)
            sep_m = int(self.config.min_horizontal_separation_nm * 1852)
            if not self.bluesky_client.stack(f"ZONER {sep_m}"):
                logger.warning(f"Failed to set horizontal separation to {sep_m}m")
            
            # Set vertical separation zone (convert feet to meters)
            alt_m = int(self.config.min_vertical_separation_ft * 0.3048)
            if not self.bluesky_client.stack(f"ZONEDH {alt_m}"):
                logger.warning(f"Failed to set vertical separation to {alt_m}m")
            
            # Set resolution factors for safety margin
            safety_factor = self.config.safety_buffer_factor
            if not self.bluesky_client.stack(f"RFACH {safety_factor}"):
                logger.warning(f"Failed to set horizontal resolution factor to {safety_factor}")
            
            if not self.bluesky_client.stack(f"RFACV {safety_factor}"):
                logger.warning(f"Failed to set vertical resolution factor to {safety_factor}")
            
            # Configure resolution zone (larger than separation zone)
            res_sep_m = int(sep_m * 1.5)  # 50% larger resolution zone
            if not self.bluesky_client.stack(f"RSZONER {res_sep_m}"):
                logger.warning(f"Failed to set resolution zone radius to {res_sep_m}m")
            
            res_alt_m = int(alt_m * 1.5)
            if not self.bluesky_client.stack(f"RSZONEDH {res_alt_m}"):
                logger.warning(f"Failed to set resolution zone altitude to {res_alt_m}m")
            
            # Set maximum resolution angle
            max_angle = int(self.config.max_resolution_angle_deg)
            # Note: BlueSky doesn't have direct command for max resolution angle
            # This would be handled in the resolution logic
            
            self.enabled = True
            logger.info("ASAS configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure ASAS: {e}")
            return False
    
    def get_conflicts(self) -> List[ASASDetection]:
        """Get current conflicts detected by ASAS."""
        if not self.enabled:
            logger.warning("ASAS not enabled")
            return []
        
        try:
            # BlueSky stores conflict information in internal data structures
            # We need to access the ASAS conflict list
            if not hasattr(self.bluesky_client, 'traf'):
                logger.error("BlueSky traffic data not available")
                return []
            
            conflicts = []
            traf = self.bluesky_client.traf
            
            # Access ASAS conflict data
            if hasattr(traf, 'asas') and hasattr(traf.asas, 'inconf'):
                # Get aircraft in conflict
                in_conflict = traf.asas.inconf
                n_aircraft = len(traf.id)
                
                for i in range(n_aircraft):
                    if in_conflict[i]:
                        # Find conflicting pairs
                        for j in range(i + 1, n_aircraft):
                            if in_conflict[j]:
                                # Check if these aircraft are in conflict with each other
                                if hasattr(traf.asas, 'pairwise_conflicts'):
                                    if traf.asas.pairwise_conflicts[i][j]:
                                        conflict = self._create_asas_detection(
                                            traf.id[i], traf.id[j], i, j, traf
                                        )
                                        if conflict:
                                            conflicts.append(conflict)
                                            self.detection_history.append(conflict)
            
            logger.debug(f"ASAS detected {len(conflicts)} conflicts")
            return conflicts
            
        except Exception as e:
            logger.error(f"Error getting ASAS conflicts: {e}")
            return []
    
    def _create_asas_detection(self, callsign1: str, callsign2: str, 
                              idx1: int, idx2: int, traf) -> Optional[ASASDetection]:
        """Create ASAS detection from BlueSky traffic data."""
        try:
            # Get time to conflict and CPA distance from ASAS data
            if hasattr(traf.asas, 'tcpa') and hasattr(traf.asas, 'dcpa'):
                tcpa_sec = traf.asas.tcpa[idx1][idx2] if hasattr(traf.asas.tcpa, '__getitem__') else 0
                dcpa_m = traf.asas.dcpa[idx1][idx2] if hasattr(traf.asas.dcpa, '__getitem__') else 0
                
                # Convert units
                time_to_conflict_min = tcpa_sec / 60.0
                distance_at_cpa_nm = dcpa_m / 1852.0
                
                # Get altitude difference
                alt1_m = traf.alt[idx1]
                alt2_m = traf.alt[idx2]
                altitude_diff_ft = abs(alt1_m - alt2_m) * 3.28084
                
                # Calculate severity (similar to our LLM system)
                severity = self._calculate_asas_severity(
                    distance_at_cpa_nm, altitude_diff_ft, time_to_conflict_min
                )
                
                return ASASDetection(
                    aircraft_pair=(callsign1, callsign2),
                    time_to_conflict_min=time_to_conflict_min,
                    distance_at_cpa_nm=distance_at_cpa_nm,
                    altitude_diff_ft=altitude_diff_ft,
                    conflict_severity=severity,
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error creating ASAS detection: {e}")
        
        return None
    
    def _calculate_asas_severity(self, distance_nm: float, alt_diff_ft: float, 
                                time_min: float) -> float:
        """Calculate conflict severity score for ASAS detection."""
        # Normalize distances to [0-1] where 1 = most severe
        h_severity = max(0.0, 1.0 - distance_nm / self.config.min_horizontal_separation_nm)
        v_severity = max(0.0, 1.0 - alt_diff_ft / self.config.min_vertical_separation_ft)
        
        # Time urgency factor
        time_urgency = max(0.0, 1.0 - time_min / self.config.lookahead_time_min)
        
        # Combined severity
        severity = (h_severity * 0.4 + v_severity * 0.4 + time_urgency * 0.2)
        return min(1.0, severity)
    
    def resolve_conflicts_auto(self) -> List[ASASResolution]:
        """Let ASAS automatically resolve conflicts."""
        if not self.enabled:
            logger.warning("ASAS not enabled for auto-resolution")
            return []
        
        try:
            # Enable automatic conflict resolution
            if not self.bluesky_client.stack("RESO ON"):
                logger.error("Failed to enable ASAS auto-resolution")
                return []
            
            # ASAS will automatically generate and execute resolutions
            # We can monitor the commands it generates
            resolutions = self._monitor_asas_resolutions()
            
            logger.info(f"ASAS auto-resolution generated {len(resolutions)} commands")
            return resolutions
            
        except Exception as e:
            logger.error(f"Error in ASAS auto-resolution: {e}")
            return []
    
    def _monitor_asas_resolutions(self) -> List[ASASResolution]:
        """Monitor ASAS resolution commands."""
        # This would monitor BlueSky's command stack for ASAS-generated commands
        # In practice, this is complex as we'd need to hook into BlueSky's 
        # internal resolution generation
        
        # For now, return empty list - in full implementation this would
        # track actual ASAS commands
        return []
    
    def resolve_conflict_manual(self, aircraft_id: str, conflict: ASASDetection) -> Optional[ASASResolution]:
        """Manually resolve conflict using ASAS-style algorithms."""
        try:
            # Get current aircraft state
            states = self.bluesky_client.get_aircraft_states()
            if aircraft_id not in states:
                logger.error(f"Aircraft {aircraft_id} not found")
                return None
            
            aircraft_state = states[aircraft_id]
            
            # Apply ASAS-style resolution logic
            resolution = self._generate_asas_resolution(aircraft_id, aircraft_state, conflict)
            
            if resolution:
                # Execute the resolution
                success = self._execute_asas_resolution(resolution)
                resolution.success = success
                self.resolution_history.append(resolution)
                
                logger.info(f"ASAS resolution for {aircraft_id}: {resolution.command_type} {resolution.command_value}")
            
            return resolution
            
        except Exception as e:
            logger.error(f"Error in manual ASAS resolution: {e}")
            return None
    
    def _generate_asas_resolution(self, aircraft_id: str, state: Dict[str, Any], 
                                 conflict: ASASDetection) -> Optional[ASASResolution]:
        """Generate ASAS-style resolution command."""
        try:
            # ASAS typically prefers horizontal resolutions (heading changes)
            # with vertical as backup
            
            current_hdg = float(state.get('hdg_deg', 0))
            current_alt = float(state.get('alt_ft', 0))
            
            # Calculate optimal heading change (simple geometric approach)
            # This is a simplified version of ASAS resolution logic
            
            if conflict.distance_at_cpa_nm < self.config.min_horizontal_separation_nm:
                # Horizontal conflict - use heading change
                # Calculate turn direction and magnitude
                turn_angle = min(self.config.max_resolution_angle_deg, 30.0)
                
                # Choose turn direction (right turn as default, left if conflict from right)
                # This is simplified - real ASAS uses more sophisticated geometry
                new_heading = (current_hdg + turn_angle) % 360
                
                return ASASResolution(
                    aircraft_id=aircraft_id,
                    command_type="HDG",
                    command_value=new_heading,
                    reason=f"ASAS horizontal resolution: turn {turn_angle}° right",
                    timestamp=datetime.now()
                )
            
            elif conflict.altitude_diff_ft < self.config.min_vertical_separation_ft:
                # Vertical conflict - use altitude change
                alt_change = max(self.config.min_vertical_separation_ft, 1000.0)
                
                # Choose climb or descend based on current altitude
                if current_alt < 30000:  # Below FL300, prefer climb
                    new_altitude = current_alt + alt_change
                else:  # Above FL300, prefer descend
                    new_altitude = current_alt - alt_change
                
                return ASASResolution(
                    aircraft_id=aircraft_id,
                    command_type="ALT",
                    command_value=new_altitude,
                    reason=f"ASAS vertical resolution: {'climb' if new_altitude > current_alt else 'descend'} {abs(new_altitude - current_alt):.0f}ft",
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error generating ASAS resolution: {e}")
        
        return None
    
    def _execute_asas_resolution(self, resolution: ASASResolution) -> bool:
        """Execute ASAS resolution command."""
        try:
            if resolution.command_type == "HDG":
                cmd = f"{resolution.aircraft_id} HDG {int(resolution.command_value)}"
            elif resolution.command_type == "ALT":
                cmd = f"{resolution.aircraft_id} ALT {int(resolution.command_value)}"
            elif resolution.command_type == "SPD":
                cmd = f"{resolution.aircraft_id} SPD {int(resolution.command_value)}"
            else:
                logger.error(f"Unknown ASAS command type: {resolution.command_type}")
                return False
            
            success = self.bluesky_client.stack(cmd)
            if success:
                logger.debug(f"Executed ASAS command: {cmd}")
            else:
                logger.error(f"Failed to execute ASAS command: {cmd}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing ASAS resolution: {e}")
            return False
    
    def disable_asas(self) -> bool:
        """Disable ASAS for LLM-only mode."""
        try:
            success = self.bluesky_client.stack("ASAS OFF")
            if success:
                self.enabled = False
                logger.info("ASAS disabled")
            return success
        except Exception as e:
            logger.error(f"Error disabling ASAS: {e}")
            return False
    
    def get_metrics(self) -> ASASMetrics:
        """Calculate ASAS performance metrics."""
        total_detections = len(self.detection_history)
        total_resolutions = len(self.resolution_history)
        successful_resolutions = sum(1 for r in self.resolution_history if r.success)
        
        # Calculate average times (simplified)
        avg_detection_time = 0.0  # Would need timing data
        avg_resolution_time = 0.0  # Would need timing data
        
        success_rate = (successful_resolutions / total_resolutions * 100) if total_resolutions > 0 else 0.0
        
        return ASASMetrics(
            total_conflicts_detected=total_detections,
            total_resolutions_attempted=total_resolutions,
            successful_resolutions=successful_resolutions,
            false_positives=0,  # Would need validation data
            missed_conflicts=0,  # Would need validation data
            average_detection_time_sec=avg_detection_time,
            average_resolution_time_sec=avg_resolution_time,
            resolution_success_rate=success_rate
        )
    
    def compare_with_llm(self, llm_conflicts: List[ConflictPrediction]) -> Dict[str, Any]:
        """Compare ASAS performance with LLM-based system."""
        asas_conflicts = self.get_conflicts()
        asas_metrics = self.get_metrics()
        
        # Compare detection capabilities
        asas_aircraft_pairs = {tuple(sorted([c.aircraft_pair[0], c.aircraft_pair[1]])) 
                              for c in asas_conflicts}
        llm_aircraft_pairs = {tuple(sorted([c.ownship_id, c.intruder_id])) 
                             for c in llm_conflicts}
        
        common_detections = asas_aircraft_pairs.intersection(llm_aircraft_pairs)
        asas_only = asas_aircraft_pairs - llm_aircraft_pairs
        llm_only = llm_aircraft_pairs - asas_aircraft_pairs
        
        comparison = {
            "total_asas_conflicts": len(asas_conflicts),
            "total_llm_conflicts": len(llm_conflicts),
            "common_detections": len(common_detections),
            "asas_only_detections": len(asas_only),
            "llm_only_detections": len(llm_only),
            "detection_agreement_rate": len(common_detections) / max(len(asas_aircraft_pairs), len(llm_aircraft_pairs)) * 100 if asas_aircraft_pairs or llm_aircraft_pairs else 0,
            "asas_metrics": asas_metrics,
            "asas_aircraft_pairs": list(asas_aircraft_pairs),
            "llm_aircraft_pairs": list(llm_aircraft_pairs),
            "common_pairs": list(common_detections),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"ASAS vs LLM comparison: {len(common_detections)}/{max(len(asas_aircraft_pairs), len(llm_aircraft_pairs))} agreement")
        
        return comparison
