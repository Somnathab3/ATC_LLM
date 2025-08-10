#!/usr/bin/env python3
"""Wolfgang (2011) Metrics Calculator

This module computes standardized aviation CDR metrics following Wolfgang (2011) research:
- TBAS: Time-Based Alerting Success 
- LAT: Loss of Alert Time
- DAT: Detection Alert Time
- DFA: Detection of First Alert
- RE: Resolution Efficiency
- RI: Resolution Intrusiveness
- RAT: Resolution Action Time

The calculator processes up to 5 CSV inputs and exports comprehensive metrics for
each (scenario_id, conflict_id) pair.

Input CSVs:
- events.csv: Per-conflict timeline events
- baseline_sep.csv: Baseline (no-resolution) separation series  
- resolved_sep.csv: Resolved (with LLM) separation series
- planned_track.csv: Planned ownship path (SCAT)
- resolved_track.csv: Actual flown ownship path (BlueSky)

Output:
- metrics_wolfgang.csv: Complete metrics per conflict
"""

import argparse
import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Parameter constants
DEFAULT_SEP_THRESHOLD_NM = 5.0
DEFAULT_ALT_THRESHOLD_FT = 1000.0  
DEFAULT_MARGIN_MIN = 5.0
DEFAULT_SEP_TARGET_NM = 5.0

# Earth radius for great circle calculations (nautical miles)
EARTH_RADIUS_NM = 3440.065


@dataclass
class ConflictData:
    """Data structure to hold all information for a single conflict."""
    scenario_id: str
    conflict_id: str
    
    # Event timeline data
    t_first_alert_min: Optional[float] = None
    t_last_alert_min: Optional[float] = None 
    t_resolution_action_min: Optional[float] = None
    t_los_min: Optional[float] = None
    
    # Separation data
    baseline_separations: List[Tuple[float, float]] = field(default_factory=list)  # (t_min, sep_nm)
    resolved_separations: List[Tuple[float, float]] = field(default_factory=list)  # (t_min, sep_nm)
    
    # Computed metrics
    TBAS: Optional[float] = None
    LAT_min: Optional[float] = None
    DAT_min: Optional[float] = None
    DFA: Optional[float] = None
    RE: Optional[float] = None
    RI: Optional[float] = None
    RAT_min: Optional[float] = None
    
    # Supporting fields
    min_sep_baseline_nm: Optional[float] = None
    min_sep_resolved_nm: Optional[float] = None


@dataclass
class TrackData:
    """Data structure for flight track information."""
    scenario_id: str
    track_points: List[Tuple[float, float, float]] = field(default_factory=list)  # (t_min, lat, lon)
    
    def get_track_length_nm(self) -> float:
        """Calculate total track length using great circle distance."""
        if len(self.track_points) < 2:
            return 0.0
            
        total_length = 0.0
        for i in range(1, len(self.track_points)):
            _, lat1, lon1 = self.track_points[i-1]
            _, lat2, lon2 = self.track_points[i]
            total_length += great_circle_distance_nm(lat1, lon1, lat2, lon2)
            
        return total_length


def great_circle_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in nautical miles."""
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_NM * c


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range."""
    return max(min_val, min(max_val, value))


class WolfgangMetricsCalculator:
    """Calculator for Wolfgang (2011) aviation CDR metrics."""
    
    def __init__(self, 
                 sep_threshold_nm: float = DEFAULT_SEP_THRESHOLD_NM,
                 alt_threshold_ft: float = DEFAULT_ALT_THRESHOLD_FT,
                 margin_min: float = DEFAULT_MARGIN_MIN,
                 sep_target_nm: float = DEFAULT_SEP_TARGET_NM):
        """Initialize calculator with parameters.
        
        Args:
            sep_threshold_nm: Horizontal LoS threshold
            alt_threshold_ft: Vertical LoS threshold  
            margin_min: Time margin for alerting evaluation
            sep_target_nm: Target minimum separation for RE normalization
        """
        self.sep_threshold_nm = sep_threshold_nm
        self.alt_threshold_ft = alt_threshold_ft
        self.margin_min = margin_min
        self.sep_target_nm = sep_target_nm
        
        # Data storage
        self.conflicts: Dict[Tuple[str, str], ConflictData] = {}
        self.planned_tracks: Dict[str, TrackData] = {}
        self.resolved_tracks: Dict[str, TrackData] = {}
        
        logger.info(f"Wolfgang calculator initialized with parameters:")
        logger.info(f"  sep_threshold_nm: {sep_threshold_nm}")
        logger.info(f"  alt_threshold_ft: {alt_threshold_ft}")
        logger.info(f"  margin_min: {margin_min}")
        logger.info(f"  sep_target_nm: {sep_target_nm}")
    
    def load_events_csv(self, filepath: Path) -> None:
        """Load events.csv and extract timeline data.
        
        Expected columns: scenario_id, conflict_id, event, t_min
        Event types: alert, alert_update, resolution_cmd, los
        """
        if not filepath.exists():
            logger.warning(f"Events CSV not found: {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            # Case-insensitive column handling
            df.columns = df.columns.str.lower()
            
            required_cols = ['scenario_id', 'conflict_id', 'event', 't_min']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in events.csv: {missing_cols}")
                return
            
            # Process each event
            for _, row in df.iterrows():
                scenario_id = str(row['scenario_id'])
                conflict_id = str(row['conflict_id'])
                event_type = str(row['event']).lower()
                t_min = float(row['t_min'])
                
                key = (scenario_id, conflict_id)
                if key not in self.conflicts:
                    self.conflicts[key] = ConflictData(scenario_id=scenario_id, conflict_id=conflict_id)
                
                conflict = self.conflicts[key]
                
                if event_type == 'alert':
                    if conflict.t_first_alert_min is None:
                        conflict.t_first_alert_min = t_min
                    conflict.t_last_alert_min = t_min
                    
                elif event_type == 'alert_update':
                    conflict.t_last_alert_min = t_min
                    
                elif event_type == 'resolution_cmd':
                    if conflict.t_resolution_action_min is None:
                        conflict.t_resolution_action_min = t_min
                        
                elif event_type == 'los':
                    if conflict.t_los_min is None:
                        conflict.t_los_min = t_min
                        
            logger.info(f"Loaded {len(df)} events for {len(self.conflicts)} conflicts")
            
        except Exception as e:
            logger.error(f"Error loading events.csv: {e}")
    
    def load_baseline_separation_csv(self, filepath: Path) -> None:
        """Load baseline_sep.csv with no-resolution separation data.
        
        Expected columns: scenario_id, conflict_id, t_min, sep_nm, vert_ft (optional)
        """
        if not filepath.exists():
            logger.warning(f"Baseline separation CSV not found: {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
            
            required_cols = ['scenario_id', 'conflict_id', 't_min', 'sep_nm']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in baseline_sep.csv: {missing_cols}")
                return
            
            # Group by conflict and store separation data
            for (scenario_id, conflict_id), group in df.groupby(['scenario_id', 'conflict_id']):
                key = (str(scenario_id), str(conflict_id))
                if key not in self.conflicts:
                    self.conflicts[key] = ConflictData(scenario_id=str(scenario_id), conflict_id=str(conflict_id))
                
                conflict = self.conflicts[key]
                
                # Sort by time and store separation points
                group_sorted = group.sort_values('t_min')
                for _, row in group_sorted.iterrows():
                    t_min = float(row['t_min'])
                    sep_nm = float(row['sep_nm'])
                    conflict.baseline_separations.append((t_min, sep_nm))
            
            logger.info(f"Loaded baseline separation data for {len(df.groupby(['scenario_id', 'conflict_id']))} conflicts")
            
        except Exception as e:
            logger.error(f"Error loading baseline_sep.csv: {e}")
    
    def load_resolved_separation_csv(self, filepath: Path) -> None:
        """Load resolved_sep.csv with LLM-resolution separation data.
        
        Expected columns: scenario_id, conflict_id, t_min, sep_nm, vert_ft (optional)
        """
        if not filepath.exists():
            logger.warning(f"Resolved separation CSV not found: {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
            
            required_cols = ['scenario_id', 'conflict_id', 't_min', 'sep_nm']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in resolved_sep.csv: {missing_cols}")
                return
            
            # Group by conflict and store separation data
            for (scenario_id, conflict_id), group in df.groupby(['scenario_id', 'conflict_id']):
                key = (str(scenario_id), str(conflict_id))
                if key not in self.conflicts:
                    self.conflicts[key] = ConflictData(scenario_id=str(scenario_id), conflict_id=str(conflict_id))
                
                conflict = self.conflicts[key]
                
                # Sort by time and store separation points
                group_sorted = group.sort_values('t_min')
                for _, row in group_sorted.iterrows():
                    t_min = float(row['t_min'])
                    sep_nm = float(row['sep_nm'])
                    conflict.resolved_separations.append((t_min, sep_nm))
            
            logger.info(f"Loaded resolved separation data for {len(df.groupby(['scenario_id', 'conflict_id']))} conflicts")
            
        except Exception as e:
            logger.error(f"Error loading resolved_sep.csv: {e}")
    
    def load_planned_track_csv(self, filepath: Path) -> None:
        """Load planned_track.csv with SCAT reference paths.
        
        Expected columns: scenario_id, t_min, lat, lon, conflict_id (optional)
        """
        if not filepath.exists():
            logger.warning(f"Planned track CSV not found: {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
            
            required_cols = ['scenario_id', 't_min', 'lat', 'lon']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in planned_track.csv: {missing_cols}")
                return
            
            # Group by scenario and store track data
            for scenario_id, group in df.groupby('scenario_id'):
                scenario_id = str(scenario_id)
                if scenario_id not in self.planned_tracks:
                    self.planned_tracks[scenario_id] = TrackData(scenario_id=scenario_id)
                
                track = self.planned_tracks[scenario_id]
                
                # Sort by time and store track points
                group_sorted = group.sort_values('t_min')
                for _, row in group_sorted.iterrows():
                    t_min = float(row['t_min'])
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    track.track_points.append((t_min, lat, lon))
            
            logger.info(f"Loaded planned tracks for {len(self.planned_tracks)} scenarios")
            
        except Exception as e:
            logger.error(f"Error loading planned_track.csv: {e}")
    
    def load_resolved_track_csv(self, filepath: Path) -> None:
        """Load resolved_track.csv with actual BlueSky flight paths.
        
        Expected columns: scenario_id, t_min, lat, lon, conflict_id (optional)
        """
        if not filepath.exists():
            logger.warning(f"Resolved track CSV not found: {filepath}")
            return
            
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
            
            required_cols = ['scenario_id', 't_min', 'lat', 'lon']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in resolved_track.csv: {missing_cols}")
                return
            
            # Group by scenario and store track data
            for scenario_id, group in df.groupby('scenario_id'):
                scenario_id = str(scenario_id)
                if scenario_id not in self.resolved_tracks:
                    self.resolved_tracks[scenario_id] = TrackData(scenario_id=scenario_id)
                
                track = self.resolved_tracks[scenario_id]
                
                # Sort by time and store track points
                group_sorted = group.sort_values('t_min')
                for _, row in group_sorted.iterrows():
                    t_min = float(row['t_min'])
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    track.track_points.append((t_min, lat, lon))
            
            logger.info(f"Loaded resolved tracks for {len(self.resolved_tracks)} scenarios")
            
        except Exception as e:
            logger.error(f"Error loading resolved_track.csv: {e}")
    
    def compute_los_time(self, conflict: ConflictData) -> Optional[float]:
        """Compute Loss of Separation time from separation data.
        
        Prioritizes resolved_sep, falls back to baseline_sep, then los event.
        """
        # Try resolved separations first
        separations = conflict.resolved_separations if conflict.resolved_separations else conflict.baseline_separations
        
        if separations:
            for t_min, sep_nm in sorted(separations):
                if sep_nm < self.sep_threshold_nm:
                    return t_min
        
        # Fall back to explicit los event
        if conflict.t_los_min is not None:
            return conflict.t_los_min
            
        return None
    
    def compute_metrics(self) -> None:
        """Compute all Wolfgang (2011) metrics for loaded conflicts."""
        logger.info(f"Computing metrics for {len(self.conflicts)} conflicts")
        
        for key, conflict in self.conflicts.items():
            self._compute_conflict_metrics(conflict)
        
        logger.info("Metrics computation completed")
    
    def _compute_conflict_metrics(self, conflict: ConflictData) -> None:
        """Compute all metrics for a single conflict."""
        # Determine t_los from data
        t_los = self.compute_los_time(conflict)
        if t_los is not None:
            conflict.t_los_min = t_los
        
        # Validate temporal ordering and log warnings
        self._validate_temporal_order(conflict)
        
        # Compute Wolfgang metrics
        self._compute_tbas(conflict)
        self._compute_lat(conflict)
        self._compute_dat(conflict)
        self._compute_dfa(conflict)
        self._compute_re(conflict)
        self._compute_ri(conflict)
        self._compute_rat(conflict)
    
    def _validate_temporal_order(self, conflict: ConflictData) -> None:
        """Validate temporal ordering of events and log warnings if violated."""
        times = []
        labels = []
        
        if conflict.t_first_alert_min is not None:
            times.append(conflict.t_first_alert_min)
            labels.append("t_first_alert")
        if conflict.t_last_alert_min is not None:
            times.append(conflict.t_last_alert_min)
            labels.append("t_last_alert")
        if conflict.t_los_min is not None:
            times.append(conflict.t_los_min)
            labels.append("t_los")
        
        # Check ordering
        if len(times) > 1:
            for i in range(len(times) - 1):
                if times[i] > times[i + 1]:
                    logger.warning(f"Temporal order violation in {conflict.scenario_id}/{conflict.conflict_id}: "
                                 f"{labels[i]} ({times[i]:.2f}) > {labels[i+1]} ({times[i+1]:.2f})")
    
    def _compute_tbas(self, conflict: ConflictData) -> None:
        """Compute TBAS: Time-Based Alerting Success."""
        if conflict.t_first_alert_min is None or conflict.t_los_min is None:
            return
            
        # TBAS = 1 if t_alert ≤ t_LoS − margin_min else 0
        alert_threshold = conflict.t_los_min - self.margin_min
        conflict.TBAS = 1.0 if conflict.t_first_alert_min <= alert_threshold else 0.0
    
    def _compute_lat(self, conflict: ConflictData) -> None:
        """Compute LAT: Loss of Alert Time."""
        if conflict.t_last_alert_min is None or conflict.t_los_min is None:
            return
            
        # LAT = t_LoS − t_last_alert
        conflict.LAT_min = conflict.t_los_min - conflict.t_last_alert_min
    
    def _compute_dat(self, conflict: ConflictData) -> None:
        """Compute DAT: Detection Alert Time."""
        if conflict.t_first_alert_min is None or conflict.t_los_min is None:
            return
            
        # DAT = t_LoS − t_first_alert
        conflict.DAT_min = conflict.t_los_min - conflict.t_first_alert_min
    
    def _compute_dfa(self, conflict: ConflictData) -> None:
        """Compute DFA: Detection of First Alert."""
        if conflict.t_first_alert_min is None or conflict.t_los_min is None:
            return
            
        # DFA = 1 if t_first_alert ≤ t_LoS − margin_min else 0
        alert_threshold = conflict.t_los_min - self.margin_min
        conflict.DFA = 1.0 if conflict.t_first_alert_min <= alert_threshold else 0.0
    
    def _compute_re(self, conflict: ConflictData) -> None:
        """Compute RE: Resolution Efficiency."""
        if not conflict.baseline_separations or not conflict.resolved_separations:
            return
        
        # Calculate minimum separations
        baseline_seps = [sep for _, sep in conflict.baseline_separations]
        resolved_seps = [sep for _, sep in conflict.resolved_separations]
        
        min_sep_baseline = min(baseline_seps)
        min_sep_resolved = min(resolved_seps)
        
        conflict.min_sep_baseline_nm = min_sep_baseline
        conflict.min_sep_resolved_nm = min_sep_resolved
        
        # Compute RE following specified formula
        gain = min_sep_resolved - min_sep_baseline
        
        if min_sep_baseline < self.sep_target_nm:
            max_possible_gain = max(0, self.sep_target_nm - min_sep_baseline)
        else:
            max_possible_gain = max(1e-6, gain)  # Avoid >1; keeps RE in [0,1]
        
        if max_possible_gain > 0:
            conflict.RE = clamp(gain / max_possible_gain, 0, 1)
        else:
            conflict.RE = None
    
    def _compute_ri(self, conflict: ConflictData) -> None:
        """Compute RI: Resolution Intrusiveness (route-length ratio)."""
        scenario_id = conflict.scenario_id
        
        if scenario_id not in self.planned_tracks or scenario_id not in self.resolved_tracks:
            return
        
        planned_track = self.planned_tracks[scenario_id]
        resolved_track = self.resolved_tracks[scenario_id]
        
        if len(planned_track.track_points) < 2 or len(resolved_track.track_points) < 2:
            return
        
        # Calculate track lengths
        planned_length = planned_track.get_track_length_nm()
        resolved_length = resolved_track.get_track_length_nm()
        
        if planned_length > 0:
            # RI = (len(resolved_track)/len(planned_track)) − 1
            conflict.RI = (resolved_length / planned_length) - 1.0
        else:
            conflict.RI = None
    
    def _compute_rat(self, conflict: ConflictData) -> None:
        """Compute RAT: Resolution Action Time."""
        if conflict.t_first_alert_min is None or conflict.t_resolution_action_min is None:
            return
            
        # RAT = t_first_resolution_execution − t_alert
        conflict.RAT_min = conflict.t_resolution_action_min - conflict.t_first_alert_min
    
    def export_to_csv(self, output_path: Path) -> None:
        """Export computed metrics to CSV file."""
        logger.info(f"Exporting metrics to: {output_path}")
        
        # Prepare data for DataFrame
        rows = []
        for key, conflict in self.conflicts.items():
            row = {
                'scenario_id': conflict.scenario_id,
                'conflict_id': conflict.conflict_id,
                't_first_alert_min': conflict.t_first_alert_min,
                't_last_alert_min': conflict.t_last_alert_min,
                't_resolution_action_min': conflict.t_resolution_action_min,
                't_los_min': conflict.t_los_min,
                'TBAS': conflict.TBAS,
                'LAT_min': conflict.LAT_min,
                'DAT_min': conflict.DAT_min,
                'DFA': conflict.DFA,
                'RE': conflict.RE,
                'RI': conflict.RI,
                'RAT_min': conflict.RAT_min,
                'min_sep_baseline_nm': conflict.min_sep_baseline_nm,
                'min_sep_resolved_nm': conflict.min_sep_resolved_nm,
            }
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        logger.info(f"Exported {len(rows)} conflict metrics to {output_path}")
    
    def print_summary(self) -> None:
        """Print summary statistics to console."""
        total_conflicts = len(self.conflicts)
        
        # Count non-null metrics
        tbas_count = sum(1 for c in self.conflicts.values() if c.TBAS is not None)
        dat_count = sum(1 for c in self.conflicts.values() if c.DAT_min is not None)
        re_count = sum(1 for c in self.conflicts.values() if c.RE is not None)
        ri_count = sum(1 for c in self.conflicts.values() if c.RI is not None)
        rat_count = sum(1 for c in self.conflicts.values() if c.RAT_min is not None)
        
        print("\n" + "="*60)
        print("WOLFGANG (2011) METRICS SUMMARY")
        print("="*60)
        print(f"Total conflicts processed: {total_conflicts}")
        print(f"Non-null TBAS: {tbas_count}")
        print(f"Non-null DAT: {dat_count}")
        print(f"Non-null RE: {re_count}")
        print(f"Non-null RI: {ri_count}")
        print(f"Non-null RAT: {rat_count}")
        
        # Calculate averages for non-null values
        if tbas_count > 0:
            avg_tbas = sum(c.TBAS for c in self.conflicts.values() if c.TBAS is not None) / tbas_count
            print(f"Average TBAS: {avg_tbas:.3f}")
        
        if dat_count > 0:
            avg_dat = sum(c.DAT_min for c in self.conflicts.values() if c.DAT_min is not None) / dat_count
            print(f"Average DAT: {avg_dat:.2f} min")
        
        if re_count > 0:
            avg_re = sum(c.RE for c in self.conflicts.values() if c.RE is not None) / re_count
            print(f"Average RE: {avg_re:.3f}")
        
        if ri_count > 0:
            avg_ri = sum(c.RI for c in self.conflicts.values() if c.RI is not None) / ri_count
            print(f"Average RI: {avg_ri:.3f}")
        
        if rat_count > 0:
            avg_rat = sum(c.RAT_min for c in self.conflicts.values() if c.RAT_min is not None) / rat_count
            print(f"Average RAT: {avg_rat:.2f} min")
        
        print("="*60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Calculate Wolfgang (2011) aviation CDR metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python wolfgang_metrics.py \\
    --events events.csv \\
    --baseline-sep baseline_sep.csv \\
    --resolved-sep resolved_sep.csv \\
    --planned-track planned_track.csv \\
    --resolved-track resolved_track.csv \\
    --output metrics_wolfgang.csv
        """
    )
    
    # Input file arguments
    parser.add_argument('--events', type=Path, help='events.csv file path')
    parser.add_argument('--baseline-sep', type=Path, help='baseline_sep.csv file path')
    parser.add_argument('--resolved-sep', type=Path, help='resolved_sep.csv file path')
    parser.add_argument('--planned-track', type=Path, help='planned_track.csv file path')
    parser.add_argument('--resolved-track', type=Path, help='resolved_track.csv file path')
    
    # Output argument
    parser.add_argument('--output', type=Path, default='metrics_wolfgang.csv',
                       help='Output CSV file path (default: metrics_wolfgang.csv)')
    
    # Parameter arguments
    parser.add_argument('--sep-threshold-nm', type=float, default=DEFAULT_SEP_THRESHOLD_NM,
                       help=f'Horizontal LoS threshold in NM (default: {DEFAULT_SEP_THRESHOLD_NM})')
    parser.add_argument('--alt-threshold-ft', type=float, default=DEFAULT_ALT_THRESHOLD_FT,
                       help=f'Vertical LoS threshold in feet (default: {DEFAULT_ALT_THRESHOLD_FT})')
    parser.add_argument('--margin-min', type=float, default=DEFAULT_MARGIN_MIN,
                       help=f'Time margin for alerting in minutes (default: {DEFAULT_MARGIN_MIN})')
    parser.add_argument('--sep-target-nm', type=float, default=DEFAULT_SEP_TARGET_NM,
                       help=f'Target separation for RE normalization in NM (default: {DEFAULT_SEP_TARGET_NM})')
    
    # Logging argument
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize calculator
    calculator = WolfgangMetricsCalculator(
        sep_threshold_nm=args.sep_threshold_nm,
        alt_threshold_ft=args.alt_threshold_ft,
        margin_min=args.margin_min,
        sep_target_nm=args.sep_target_nm
    )
    
    # Load input files
    if args.events:
        calculator.load_events_csv(args.events)
    if args.baseline_sep:
        calculator.load_baseline_separation_csv(args.baseline_sep)
    if args.resolved_sep:
        calculator.load_resolved_separation_csv(args.resolved_sep)
    if args.planned_track:
        calculator.load_planned_track_csv(args.planned_track)
    if args.resolved_track:
        calculator.load_resolved_track_csv(args.resolved_track)
    
    # Compute metrics
    calculator.compute_metrics()
    
    # Export results
    calculator.export_to_csv(args.output)
    
    # Print summary
    calculator.print_summary()
    
    print(f"\nMetrics exported to: {args.output}")


if __name__ == '__main__':
    main()
