"""Enhanced Metrics Collection for Wolfgang (2011) Aviation Performance KPIs.

This module implements comprehensive metrics logging for conflict detection and
resolution systems, recording Wolfgang (2011) Key Performance Indicators (KPIs)
on a per-run basis for systematic evaluation and comparison.

Metrics collected:
- TBAS: Time-Based Assessment Score
- LAT: Look-Ahead Time
- DAT: Detection Accuracy Time
- DFA: Detection False Alarm rate
- RE: Resolution Efficiency
- RI: Resolution Intrusion
- RAT: Resolution Action Time
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be recorded."""
    DETECTION = "detection"
    RESOLUTION = "resolution"
    SYSTEM = "system"
    SCENARIO = "scenario"

class ConflictType(Enum):
    """Types of conflicts for metric categorization."""
    TRUE_POSITIVE = "true_positive"      # Correctly detected conflict
    FALSE_POSITIVE = "false_positive"    # False alarm
    TRUE_NEGATIVE = "true_negative"      # Correctly no conflict
    FALSE_NEGATIVE = "false_negative"    # Missed conflict

@dataclass
class DetectionMetrics:
    """Wolfgang (2011) detection-specific metrics."""
    # Core Detection Metrics
    tbas: float = 0.0                    # Time-Based Assessment Score (0-1)
    lat: float = 0.0                     # Look-Ahead Time (minutes)
    dat: float = 0.0                     # Detection Accuracy Time (minutes)
    dfa: float = 0.0                     # Detection False Alarm rate (0-1)
    
    # Additional Detection Context
    detection_time_sec: float = 0.0      # Time to detect conflict
    cpa_time_actual: float = 0.0         # Actual CPA time when detected
    cpa_distance_actual: float = 0.0     # Actual CPA distance when detected
    confidence_score: float = 0.0        # Detection confidence (0-1)
    conflict_type: ConflictType = ConflictType.TRUE_POSITIVE

@dataclass
class ResolutionMetrics:
    """Wolfgang (2011) resolution-specific metrics."""
    # Core Resolution Metrics
    re: float = 0.0                      # Resolution Efficiency (0-1)
    ri: float = 0.0                      # Resolution Intrusion (NM)
    rat: float = 0.0                     # Resolution Action Time (seconds)
    
    # Additional Resolution Context
    resolution_success: bool = False     # Whether resolution was successful
    maneuver_type: str = ""              # Type of maneuver (HDG, ALT, SPD)
    deviation_amount: float = 0.0        # Amount of deviation required
    return_to_route_time: float = 0.0    # Time to return to original route
    fuel_cost_estimate: float = 0.0      # Estimated additional fuel cost

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    # Processing Performance
    processing_time_ms: float = 0.0      # Total processing time
    cpu_usage_percent: float = 0.0       # CPU utilization
    memory_usage_mb: float = 0.0         # Memory usage
    
    # Algorithm Performance
    prediction_accuracy: float = 0.0     # Prediction accuracy (0-1)
    false_alarm_rate: float = 0.0        # False alarm rate (0-1)
    miss_rate: float = 0.0              # Miss rate (0-1)
    
    # Timing Performance
    average_detection_time: float = 0.0  # Average detection time
    max_detection_time: float = 0.0      # Maximum detection time
    min_detection_time: float = 0.0      # Minimum detection time

@dataclass
class ScenarioMetrics:
    """Scenario-specific metrics for reproducible testing."""
    scenario_id: str = ""                # Unique scenario identifier
    scenario_type: str = ""              # Type of scenario (crossing, head_on, etc.)
    scenario_seed: int = 0               # Random seed for reproducibility
    
    # Scenario Parameters
    initial_separation: float = 0.0      # Initial separation distance (NM)
    closure_rate: float = 0.0           # Rate of closure (kt)
    conflict_geometry: str = ""          # Geometric description
    
    # Outcome Metrics
    minimum_separation_achieved: float = 0.0  # Minimum separation achieved
    conflict_resolved: bool = False       # Whether conflict was resolved
    safety_margin_maintained: bool = False  # Whether safety margins were maintained

@dataclass
class RunMetrics:
    """Complete metrics for a single run/encounter."""
    # Run Identification
    run_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    aircraft_id: str = ""
    intruder_id: str = ""
    
    # Component Metrics
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    resolution: ResolutionMetrics = field(default_factory=ResolutionMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    scenario: ScenarioMetrics = field(default_factory=ScenarioMetrics)
    
    # Summary
    overall_success: bool = False         # Overall success of the encounter
    safety_critical: bool = False         # Whether encounter was safety-critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return asdict(self)

class EnhancedMetricsCollector:
    """Enhanced metrics collector for systematic evaluation."""
    
    def __init__(self, output_dir: Path = Path("metrics_output")):
        """Initialize the metrics collector.
        
        Args:
            output_dir: Directory to save metrics files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Active runs tracking
        self.active_runs: Dict[str, RunMetrics] = {}
        self.completed_runs: List[RunMetrics] = []
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Enhanced metrics collector initialized - Session: {self.session_id}")
    
    def start_run(self, aircraft_id: str, intruder_id: str = "", 
                  scenario_params: Optional[Dict[str, Any]] = None) -> str:
        """Start a new metrics run.
        
        Args:
            aircraft_id: Primary aircraft identifier
            intruder_id: Intruder aircraft identifier
            scenario_params: Optional scenario parameters
            
        Returns:
            run_id: Unique identifier for this run
        """
        run_id = f"{aircraft_id}_{intruder_id}_{int(time.time() * 1000)}"
        
        run_metrics = RunMetrics(
            run_id=run_id,
            aircraft_id=aircraft_id,
            intruder_id=intruder_id,
            timestamp=datetime.now()
        )
        
        # Initialize scenario metrics if provided
        if scenario_params:
            run_metrics.scenario.scenario_id = scenario_params.get("scenario_id", "")
            run_metrics.scenario.scenario_type = scenario_params.get("pattern", "")
            run_metrics.scenario.scenario_seed = scenario_params.get("seed", 0)
            run_metrics.scenario.initial_separation = scenario_params.get("initial_separation", 0.0)
            run_metrics.scenario.closure_rate = scenario_params.get("closure_rate", 0.0)
        
        self.active_runs[run_id] = run_metrics
        logger.debug(f"Started metrics run: {run_id}")
        return run_id
    
    def record_detection(self, run_id: str, detection_time: float, 
                        cpa_time: float, cpa_distance: float,
                        confidence: float = 1.0, 
                        conflict_type: ConflictType = ConflictType.TRUE_POSITIVE) -> None:
        """Record detection metrics.
        
        Args:
            run_id: Run identifier
            detection_time: Time taken to detect (seconds)
            cpa_time: CPA time when detected (minutes)
            cpa_distance: CPA distance when detected (NM)
            confidence: Detection confidence (0-1)
            conflict_type: Type of conflict detection
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found for detection recording")
            return
        
        detection = self.active_runs[run_id].detection
        detection.detection_time_sec = detection_time
        detection.cpa_time_actual = cpa_time
        detection.cpa_distance_actual = cpa_distance
        detection.confidence_score = confidence
        detection.conflict_type = conflict_type
        
        # Calculate Wolfgang (2011) metrics
        detection.lat = cpa_time  # Look-Ahead Time
        detection.dat = detection_time / 60.0  # Detection Accuracy Time in minutes
        detection.tbas = self._calculate_tbas(cpa_time, cpa_distance, confidence)
        detection.dfa = 1.0 if conflict_type == ConflictType.FALSE_POSITIVE else 0.0
        
        logger.debug(f"Recorded detection metrics for run {run_id}: "
                    f"LAT={detection.lat:.2f}min, DAT={detection.dat:.2f}min, "
                    f"TBAS={detection.tbas:.3f}")
    
    def record_resolution(self, run_id: str, action_time: float,
                         maneuver_type: str, deviation_amount: float,
                         success: bool = True, intrusion: float = 0.0) -> None:
        """Record resolution metrics.
        
        Args:
            run_id: Run identifier
            action_time: Time to execute resolution action (seconds)
            maneuver_type: Type of maneuver (HDG, ALT, SPD)
            deviation_amount: Amount of deviation
            success: Whether resolution was successful
            intrusion: Resolution intrusion distance (NM)
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found for resolution recording")
            return
        
        resolution = self.active_runs[run_id].resolution
        resolution.rat = action_time  # Resolution Action Time
        resolution.maneuver_type = maneuver_type
        resolution.deviation_amount = deviation_amount
        resolution.resolution_success = success
        resolution.ri = intrusion  # Resolution Intrusion
        
        # Calculate Resolution Efficiency
        resolution.re = self._calculate_resolution_efficiency(
            action_time, deviation_amount, success
        )
        
        logger.debug(f"Recorded resolution metrics for run {run_id}: "
                    f"RAT={resolution.rat:.2f}s, RE={resolution.re:.3f}, "
                    f"RI={resolution.ri:.2f}NM")
    
    def record_system_performance(self, run_id: str, processing_time: float,
                                cpu_usage: float = 0.0, memory_usage: float = 0.0) -> None:
        """Record system performance metrics.
        
        Args:
            run_id: Run identifier
            processing_time: Processing time in milliseconds
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage in MB
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found for system performance recording")
            return
        
        system = self.active_runs[run_id].system
        system.processing_time_ms = processing_time
        system.cpu_usage_percent = cpu_usage
        system.memory_usage_mb = memory_usage
    
    def record_final_outcome(self, run_id: str, min_separation: float,
                           conflict_resolved: bool, safety_maintained: bool) -> None:
        """Record final outcome metrics.
        
        Args:
            run_id: Run identifier
            min_separation: Minimum separation achieved (NM)
            conflict_resolved: Whether conflict was resolved
            safety_maintained: Whether safety margins were maintained
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found for outcome recording")
            return
        
        run_metrics = self.active_runs[run_id]
        run_metrics.scenario.minimum_separation_achieved = min_separation
        run_metrics.scenario.conflict_resolved = conflict_resolved
        run_metrics.scenario.safety_margin_maintained = safety_maintained
        
        # Overall success assessment
        run_metrics.overall_success = (
            conflict_resolved and 
            safety_maintained and 
            min_separation >= 5.0  # Standard separation minimum
        )
        
        # Safety critical assessment
        run_metrics.safety_critical = min_separation < 1.0  # Near miss threshold
    
    def complete_run(self, run_id: str) -> Optional[RunMetrics]:
        """Complete a metrics run and move to completed runs.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Completed run metrics or None if run not found
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found for completion")
            return None
        
        run_metrics = self.active_runs.pop(run_id)
        self.completed_runs.append(run_metrics)
        
        logger.info(f"Completed metrics run {run_id}: "
                   f"Success={run_metrics.overall_success}, "
                   f"Critical={run_metrics.safety_critical}")
        
        return run_metrics
    
    def save_session_metrics(self, filename: Optional[str] = None) -> Path:
        """Save all session metrics to JSON file.
        
        Args:
            filename: Optional filename, defaults to session-based name
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"metrics_session_{self.session_id}.json"
        
        filepath = self.output_dir / filename
        
        session_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_runs": len(self.completed_runs),
            "runs": [run.to_dict() for run in self.completed_runs]
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.completed_runs)} run metrics to {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session.
        
        Returns:
            Dictionary with session summary statistics
        """
        if not self.completed_runs:
            return {"total_runs": 0, "message": "No completed runs"}
        
        # Basic counts
        total_runs = len(self.completed_runs)
        successful_runs = sum(1 for run in self.completed_runs if run.overall_success)
        safety_critical = sum(1 for run in self.completed_runs if run.safety_critical)
        
        # Wolfgang (2011) KPI averages
        avg_tbas = sum(run.detection.tbas for run in self.completed_runs) / total_runs
        avg_lat = sum(run.detection.lat for run in self.completed_runs) / total_runs
        avg_dat = sum(run.detection.dat for run in self.completed_runs) / total_runs
        avg_dfa = sum(run.detection.dfa for run in self.completed_runs) / total_runs
        avg_re = sum(run.resolution.re for run in self.completed_runs) / total_runs
        avg_ri = sum(run.resolution.ri for run in self.completed_runs) / total_runs
        avg_rat = sum(run.resolution.rat for run in self.completed_runs) / total_runs
        
        # Minimum separations
        min_separations = [run.scenario.minimum_separation_achieved 
                          for run in self.completed_runs]
        avg_min_sep = sum(min_separations) / total_runs
        worst_min_sep = min(min_separations)
        
        return {
            "session_id": self.session_id,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": successful_runs / total_runs,
            "safety_critical_runs": safety_critical,
            "safety_critical_rate": safety_critical / total_runs,
            
            # Wolfgang (2011) KPIs
            "wolfgang_2011_kpis": {
                "tbas_avg": avg_tbas,
                "lat_avg_min": avg_lat,
                "dat_avg_min": avg_dat,
                "dfa_avg": avg_dfa,
                "re_avg": avg_re,
                "ri_avg_nm": avg_ri,
                "rat_avg_sec": avg_rat
            },
            
            # Separation Statistics
            "separation_stats": {
                "avg_min_separation_nm": avg_min_sep,
                "worst_min_separation_nm": worst_min_sep,
                "runs_below_5nm": sum(1 for sep in min_separations if sep < 5.0),
                "runs_below_1nm": sum(1 for sep in min_separations if sep < 1.0)
            }
        }
    
    def _calculate_tbas(self, cpa_time: float, cpa_distance: float, 
                       confidence: float) -> float:
        """Calculate Time-Based Assessment Score (Wolfgang 2011).
        
        Args:
            cpa_time: CPA time in minutes
            cpa_distance: CPA distance in NM
            confidence: Detection confidence (0-1)
            
        Returns:
            TBAS score (0-1)
        """
        # TBAS considers time available, separation, and confidence
        # Higher score = better performance
        
        # Time component (more time = better)
        time_factor = min(1.0, cpa_time / 10.0)  # Normalize to 10 minutes
        
        # Distance component (closer conflicts are more challenging)
        distance_factor = 1.0 - min(1.0, cpa_distance / 10.0)  # Inverse relationship
        
        # Confidence component
        confidence_factor = confidence
        
        # Weighted combination
        tbas = (0.4 * time_factor + 0.3 * distance_factor + 0.3 * confidence_factor)
        return max(0.0, min(1.0, tbas))
    
    def _calculate_resolution_efficiency(self, action_time: float, 
                                       deviation: float, success: bool) -> float:
        """Calculate Resolution Efficiency (Wolfgang 2011).
        
        Args:
            action_time: Time to execute action (seconds)
            deviation: Amount of deviation required
            success: Whether resolution was successful
            
        Returns:
            RE score (0-1)
        """
        if not success:
            return 0.0
        
        # Efficiency = successful resolution with minimal time and deviation
        time_factor = max(0.0, 1.0 - (action_time / 60.0))  # Normalize to 1 minute
        deviation_factor = max(0.0, 1.0 - (deviation / 30.0))  # Normalize to 30 degrees/units
        
        return (0.6 * time_factor + 0.4 * deviation_factor)
