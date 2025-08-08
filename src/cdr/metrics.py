"""Wolfgang (2011) KPI metrics collection and calculation.

This module implements comprehensive performance metrics for conflict detection
and resolution systems following aviation research standards:

KPIs implemented:
- TBAS: Time-Based Alerting Score
- LAT: Loss of Alerting Time  
- PA: Predicted Alerts
- PI: Predicted Intrusions
- DAT: Delay in Alert Time
- DFA: Delay in First Alert
- RE: Resolution Efficiency
- RI: Resolution Intrusiveness  
- RAT: Resolution Alert Time
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from .schemas import ConflictPrediction, ResolutionCommand, AircraftState

logger = logging.getLogger(__name__)


@dataclass
class MetricsSummary:
    """Summary of CDR system performance metrics."""
    
    # Timing metrics
    total_simulation_time_min: float
    total_cycles: int
    avg_cycle_time_sec: float
    
    # Detection metrics
    total_conflicts_detected: int
    true_conflicts: int
    false_positives: int
    false_negatives: int
    detection_accuracy: float
    
    # Wolfgang (2011) KPIs
    tbas: float  # Time-Based Alerting Score
    lat: float   # Loss of Alerting Time
    pa: int      # Predicted Alerts
    pi: int      # Predicted Intrusions
    dat: float   # Delay in Alert Time
    dfa: float   # Delay in First Alert
    re: float    # Resolution Efficiency
    ri: float    # Resolution Intrusiveness
    rat: float   # Resolution Alert Time
    
    # Safety metrics
    min_separation_achieved_nm: float
    avg_separation_nm: float
    safety_violations: int
    
    # Resolution metrics
    total_resolutions_issued: int
    successful_resolutions: int
    resolution_success_rate: float
    avg_resolution_time_sec: float


@dataclass
class BaselineMetrics:
    """Metrics for baseline (LLM-disabled) CDR system."""
    
    # Detection metrics
    baseline_conflicts_detected: int
    baseline_false_positives: int
    baseline_detection_latency_sec: float
    
    # Resolution metrics
    baseline_resolutions_issued: int
    baseline_success_rate: float
    baseline_resolution_delay_sec: float
    
    # Safety metrics
    baseline_min_separation_nm: float
    baseline_avg_separation_nm: float
    baseline_safety_violations: int


@dataclass
class ComparisonReport:
    """Comparison report between LLM and baseline systems."""
    
    # Detection comparison
    detection_accuracy_improvement: float  # LLM vs baseline
    false_alert_reduction: float
    detection_latency_change_sec: float
    
    # Resolution comparison
    success_rate_improvement: float
    resolution_delay_improvement_sec: float
    intrusiveness_comparison: float  # Lower is better
    
    # Safety comparison
    separation_margin_improvement_nm: float
    safety_violation_reduction: int
    
    # Overall assessment
    overall_score: float  # Composite score 0-100
    recommendation: str   # Text recommendation


class MetricsCollector:
    """Collects and calculates CDR system performance metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.reset()
        logger.info("Metrics collector initialized")
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.start_time = datetime.now()
        self.cycle_times: List[float] = []
        
        # Detection tracking
        self.conflicts_detected: List[ConflictPrediction] = []
        self.ground_truth_conflicts: List[ConflictPrediction] = []
        self.alert_times: Dict[str, datetime] = {}
        self.intrusion_times: Dict[str, datetime] = {}
        
        # Resolution tracking  
        self.resolutions_issued: List[ResolutionCommand] = []
        self.resolution_outcomes: Dict[str, bool] = {}
        self.resolution_timings: Dict[str, Tuple[datetime, Optional[datetime]]] = {}
        
        # Safety tracking
        self.separation_history: List[Tuple[datetime, str, str, float]] = []
        self.safety_violations: List[Tuple[datetime, str, str, float]] = []
        
        logger.debug("Metrics reset")
    
    def record_cycle_time(self, cycle_duration_sec: float) -> None:
        """Record cycle execution time.
        
        Args:
            cycle_duration_sec: Time taken for cycle execution
        """
        self.cycle_times.append(cycle_duration_sec)
    
    def record_conflict_detection(
        self, 
        conflicts: List[ConflictPrediction],
        detection_time: datetime
    ) -> None:
        """Record conflict detection results.
        
        Args:
            conflicts: List of detected conflicts
            detection_time: When detection was performed
        """
        for conflict in conflicts:
            self.conflicts_detected.append(conflict)
            
            # Record alert time for this conflict pair
            conflict_key = f"{conflict.ownship_id}_{conflict.intruder_id}"
            if conflict_key not in self.alert_times:
                self.alert_times[conflict_key] = detection_time
    
    def record_ground_truth(self, true_conflicts: List[ConflictPrediction]) -> None:
        """Record ground truth conflicts for validation.
        
        Args:
            true_conflicts: List of actual conflicts that occurred
        """
        self.ground_truth_conflicts.extend(true_conflicts)
    
    def record_resolution_issued(
        self, 
        resolution: ResolutionCommand,
        issue_time: datetime
    ) -> None:
        """Record resolution command issuance.
        
        Args:
            resolution: Resolution command issued
            issue_time: When resolution was issued
        """
        self.resolutions_issued.append(resolution)
        self.resolution_timings[resolution.resolution_id] = (issue_time, None)
    
    def record_resolution_outcome(
        self, 
        resolution_id: str, 
        success: bool,
        completion_time: Optional[datetime] = None
    ) -> None:
        """Record resolution execution outcome.
        
        Args:
            resolution_id: Resolution identifier
            success: Whether resolution was successful
            completion_time: When resolution completed
        """
        self.resolution_outcomes[resolution_id] = success
        
        if completion_time and resolution_id in self.resolution_timings:
            start_time = self.resolution_timings[resolution_id][0]
            self.resolution_timings[resolution_id] = (start_time, completion_time)
    
    def record_separation(
        self, 
        time: datetime, 
        aircraft1: str, 
        aircraft2: str, 
        separation_nm: float
    ) -> None:
        """Record aircraft separation measurement.
        
        Args:
            time: Measurement timestamp
            aircraft1: First aircraft identifier
            aircraft2: Second aircraft identifier  
            separation_nm: Separation distance in nautical miles
        """
        self.separation_history.append((time, aircraft1, aircraft2, separation_nm))
        
        # Check for safety violations (< 5 NM)
        if separation_nm < 5.0:
            self.safety_violations.append((time, aircraft1, aircraft2, separation_nm))
            logger.warning(f"Safety violation: {aircraft1}-{aircraft2} separated by {separation_nm:.2f} NM")
    
    def calculate_wolfgang_kpis(self) -> Dict[str, float]:
        """Calculate Wolfgang (2011) KPIs.
        
        Returns:
            Dictionary of KPI values
        """
        kpis = {}
        
        # TBAS: Time-Based Alerting Score
        kpis['tbas'] = self._calculate_tbas()
        
        # LAT: Loss of Alerting Time
        kpis['lat'] = self._calculate_lat()
        
        # PA: Predicted Alerts (count)
        kpis['pa'] = len(self.conflicts_detected)
        
        # PI: Predicted Intrusions (count of conflicts that became actual)
        kpis['pi'] = len([c for c in self.conflicts_detected if c.is_conflict])
        
        # DAT: Delay in Alert Time  
        kpis['dat'] = self._calculate_dat()
        
        # DFA: Delay in First Alert
        kpis['dfa'] = self._calculate_dfa()
        
        # RE: Resolution Efficiency
        kpis['re'] = self._calculate_re()
        
        # RI: Resolution Intrusiveness
        kpis['ri'] = self._calculate_ri()
        
        # RAT: Resolution Alert Time
        kpis['rat'] = self._calculate_rat()
        
        return kpis
    
    def generate_summary(self) -> MetricsSummary:
        """Generate comprehensive metrics summary.
        
        Returns:
            Complete metrics summary
        """
        end_time = datetime.now()
        total_time_min = (end_time - self.start_time).total_seconds() / 60.0
        
        # Calculate detection accuracy
        tp = len([c for c in self.conflicts_detected if c.is_conflict])  # True positives
        fp = len([c for c in self.conflicts_detected if not c.is_conflict])  # False positives
        fn = len(self.ground_truth_conflicts) - tp  # False negatives (approximation)
        
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # Wolfgang KPIs
        wolfgang_kpis = self.calculate_wolfgang_kpis()
        
        # Safety metrics
        separations = [sep for _, _, _, sep in self.separation_history]
        min_sep = min(separations) if separations else float('inf')
        avg_sep = np.mean(separations) if separations else 0.0
        
        # Resolution metrics
        successful_resolutions = sum(self.resolution_outcomes.values())
        success_rate = (successful_resolutions / len(self.resolutions_issued) 
                       if self.resolutions_issued else 0.0)
        
        # Calculate average resolution time
        resolution_durations = []
        for start_time, end_time in self.resolution_timings.values():
            if end_time:
                duration = (end_time - start_time).total_seconds()
                resolution_durations.append(duration)
        
        avg_resolution_time = np.mean(resolution_durations) if resolution_durations else 0.0
        
        return MetricsSummary(
            total_simulation_time_min=total_time_min,
            total_cycles=len(self.cycle_times),
            avg_cycle_time_sec=np.mean(self.cycle_times) if self.cycle_times else 0.0,
            
            total_conflicts_detected=len(self.conflicts_detected),
            true_conflicts=tp,
            false_positives=fp,
            false_negatives=max(0, fn),
            detection_accuracy=accuracy,
            
            tbas=wolfgang_kpis.get('tbas', 0.0),
            lat=wolfgang_kpis.get('lat', 0.0),
            pa=wolfgang_kpis.get('pa', 0),
            pi=wolfgang_kpis.get('pi', 0),
            dat=wolfgang_kpis.get('dat', 0.0),
            dfa=wolfgang_kpis.get('dfa', 0.0),
            re=wolfgang_kpis.get('re', 0.0),
            ri=wolfgang_kpis.get('ri', 0.0),
            rat=wolfgang_kpis.get('rat', 0.0),
            
            min_separation_achieved_nm=min_sep,
            avg_separation_nm=avg_sep,
            safety_violations=len(self.safety_violations),
            
            total_resolutions_issued=len(self.resolutions_issued),
            successful_resolutions=successful_resolutions,
            resolution_success_rate=success_rate,
            avg_resolution_time_sec=avg_resolution_time
        )
    
    def compare_with_baseline(
        self, 
        baseline_metrics: BaselineMetrics
    ) -> ComparisonReport:
        """Compare LLM system performance with baseline.
        
        Args:
            baseline_metrics: Baseline system performance metrics
            
        Returns:
            Detailed comparison report
        """
        llm_summary = self.generate_summary()
        
        # Detection comparison
        detection_accuracy_improvement = (
            llm_summary.detection_accuracy - 
            (1.0 - baseline_metrics.baseline_false_positives / max(1, baseline_metrics.baseline_conflicts_detected))
        )
        
        false_alert_reduction = (
            baseline_metrics.baseline_false_positives - llm_summary.false_positives
        ) / max(1, baseline_metrics.baseline_false_positives)
        
        # Resolution comparison
        success_rate_improvement = (
            llm_summary.resolution_success_rate - baseline_metrics.baseline_success_rate
        )
        
        resolution_delay_improvement = (
            baseline_metrics.baseline_resolution_delay_sec - llm_summary.avg_resolution_time_sec
        )
        
        # Safety comparison
        separation_improvement = (
            llm_summary.min_separation_achieved_nm - baseline_metrics.baseline_min_separation_nm
        )
        
        safety_violation_reduction = (
            baseline_metrics.baseline_safety_violations - llm_summary.safety_violations
        )
        
        # Calculate overall score (0-100)
        score_components = [
            max(0, min(100, detection_accuracy_improvement * 100)),  # 0-100
            max(0, min(100, false_alert_reduction * 100)),           # 0-100
            max(0, min(100, success_rate_improvement * 100)),        # 0-100
            max(0, min(100, separation_improvement * 20)),           # 0-100 (5nm = 100)
            max(0, min(100, safety_violation_reduction * 10))        # 0-100 (10 violations = 100)
        ]
        overall_score = sum(score_components) / len(score_components)
        
        # Generate recommendation
        if overall_score >= 80:
            recommendation = "LLM system significantly outperforms baseline. Recommend deployment."
        elif overall_score >= 60:
            recommendation = "LLM system shows improvement over baseline. Consider deployment with monitoring."
        elif overall_score >= 40:
            recommendation = "LLM system shows mixed results. Requires further optimization."
        else:
            recommendation = "LLM system underperforms baseline. Not recommended for deployment."
            
        return ComparisonReport(
            detection_accuracy_improvement=detection_accuracy_improvement,
            false_alert_reduction=false_alert_reduction,
            detection_latency_change_sec=0.0,  # TODO: Implement latency tracking
            success_rate_improvement=success_rate_improvement,
            resolution_delay_improvement_sec=resolution_delay_improvement,
            intrusiveness_comparison=llm_summary.ri,  # TODO: Compare with baseline RI
            separation_margin_improvement_nm=separation_improvement,
            safety_violation_reduction=safety_violation_reduction,
            overall_score=overall_score,
            recommendation=recommendation
        )

    def save_report(self, filepath: str) -> None:
        """Save metrics report to file.
        
        Args:
            filepath: Path to save report
        """
        summary = self.generate_summary()
        
        with open(filepath, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        logger.info(f"Metrics report saved to {filepath}")
    
    # Wolfgang (2011) KPI calculation methods
    
    def _calculate_tbas(self) -> float:
        """Calculate Time-Based Alerting Score.
        
        TBAS measures the proportion of time the system correctly predicts
        conflicts within the alerting time window (typically 5 minutes).
        """
        if not self.conflicts_detected:
            return 0.0
            
        tbas_window_min = 5.0  # Standard TBAS alerting window
        correct_alerts = 0
        total_opportunities = 0
        
        for conflict in self.conflicts_detected:
            total_opportunities += 1
            if conflict.is_conflict and conflict.time_to_cpa_min <= tbas_window_min:
                correct_alerts += 1
                
        return correct_alerts / total_opportunities if total_opportunities > 0 else 0.0
    
    def _calculate_lat(self) -> float:
        """Calculate Loss of Alerting Time.
        
        LAT measures the average time difference between when an alert
        should have been issued and when it was actually issued.
        """
        lat_window_min = 10.0  # Standard LAT horizon
        total_loss = 0.0
        conflict_count = 0
        
        for conflict in self.conflicts_detected:
            if conflict.is_conflict:
                # Optimal alert time would be at LAT horizon
                optimal_alert_time = lat_window_min
                actual_alert_time = conflict.time_to_cpa_min
                
                if actual_alert_time < optimal_alert_time:
                    loss = optimal_alert_time - actual_alert_time
                    total_loss += loss
                    conflict_count += 1
                    
        return total_loss / conflict_count if conflict_count > 0 else 0.0
    
    def _calculate_dat(self) -> float:
        """Calculate Delay in Alert Time.
        
        DAT measures how long alerts are delayed compared to optimal timing.
        """
        if not self.alert_times:
            return 0.0
            
        delays = []
        for conflict in self.conflicts_detected:
            if conflict.is_conflict:
                # Calculate delay based on when alert should have been issued
                # vs when it was actually detected
                optimal_time = 10.0  # Should alert 10 min before CPA
                actual_time = conflict.time_to_cpa_min
                
                if actual_time < optimal_time:
                    delay = optimal_time - actual_time
                    delays.append(delay)
                    
        return np.mean(delays) if delays else 0.0
    
    def _calculate_dfa(self) -> float:
        """Calculate Delay in First Alert.
        
        DFA measures the delay in issuing the first alert for each conflict.
        """
        if not self.alert_times:
            return 0.0
            
        first_alert_delays = []
        
        # Group conflicts by aircraft pair
        conflict_pairs = {}
        for conflict in self.conflicts_detected:
            if conflict.is_conflict:
                pair_key = f"{conflict.ownship_id}_{conflict.intruder_id}"
                if pair_key not in conflict_pairs:
                    conflict_pairs[pair_key] = []
                conflict_pairs[pair_key].append(conflict)
        
        # Calculate delay for first alert in each pair
        for pair_key, conflicts in conflict_pairs.items():
            if conflicts:
                # Find earliest detection
                earliest_conflict = min(conflicts, key=lambda c: c.time_to_cpa_min)
                optimal_first_alert = 10.0  # Should first alert 10 min before CPA
                actual_first_alert = earliest_conflict.time_to_cpa_min
                
                if actual_first_alert < optimal_first_alert:
                    delay = optimal_first_alert - actual_first_alert
                    first_alert_delays.append(delay)
                    
        return np.mean(first_alert_delays) if first_alert_delays else 0.0
    
    def _calculate_re(self) -> float:
        """Calculate Resolution Efficiency.
        
        RE measures how efficiently resolutions solve conflicts.
        """
        if not self.resolutions_issued:
            return 0.0
            
        successful_count = sum(self.resolution_outcomes.values())
        return successful_count / len(self.resolutions_issued)
    
    def _calculate_ri(self) -> float:
        """Calculate Resolution Intrusiveness.
        
        RI measures how much resolutions deviate from original flight plans.
        """
        if not self.resolutions_issued:
            return 0.0
            
        total_intrusiveness = 0.0
        count = 0
        
        for resolution in self.resolutions_issued:
            # Calculate intrusiveness based on magnitude of changes
            intrusiveness = 0.0
            
            if resolution.new_heading_deg is not None:
                # Heading change intrusiveness (0-1 scale)
                # Assume original heading not available, use moderate weight
                intrusiveness += 0.3
                
            if resolution.new_altitude_ft is not None:
                # Altitude change intrusiveness 
                intrusiveness += 0.4
                
            if resolution.new_speed_kt is not None:
                # Speed change intrusiveness
                intrusiveness += 0.2
                
            total_intrusiveness += intrusiveness
            count += 1
            
        return total_intrusiveness / count if count > 0 else 0.0
    
    def _calculate_rat(self) -> float:
        """Calculate Resolution Alert Time.
        
        RAT measures the average time from alert to resolution issuance.
        """
        if not self.alert_times or not self.resolutions_issued:
            return 0.0
            
        resolution_delays = []
        
        for resolution in self.resolutions_issued:
            # Find corresponding alert time
            # This is simplified - in practice would need better matching
            alert_time = list(self.alert_times.values())[0]  # Simplified
            resolution_time = resolution.issue_time
            
            delay = (resolution_time - alert_time).total_seconds() / 60.0  # Convert to minutes
            if delay >= 0:
                resolution_delays.append(delay)
                
        return np.mean(resolution_delays) if resolution_delays else 0.0
