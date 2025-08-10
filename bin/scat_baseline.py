#!/usr/bin/env python3
"""
SCAT Baseline Generator - Generate baseline traffic analysis from SCAT data

This module builds a 100 NM/5000 ft baseline from given SCAT files, analyzing
neighbor aircraft and producing comprehensive traffic reports.

Usage:
    python scat_baseline.py --root <dir> --ownship <file> --radius 100nm --altwin 5000
"""

import sys
import json
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Add the parent directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.cdr.scat_adapter import SCATAdapter, SCATFlightRecord
from src.cdr.schemas import AircraftState
from src.cdr.geodesy import haversine_nm, bearing_deg
from src.utils.output_utils import get_output_path, TestTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scat_baseline")


@dataclass
class NeighborAircraft:
    """Information about a neighboring aircraft."""
    callsign: str
    aircraft_type: str
    distance_nm: float
    bearing_deg: float
    altitude_ft: float
    relative_alt_ft: float
    ground_speed_kt: float
    heading_deg: float
    closest_approach_nm: float
    time_to_closest_approach_min: float


@dataclass
class BaselineReport:
    """Complete baseline analysis report."""
    ownship_callsign: str
    analysis_timestamp: str
    radius_filter_nm: float
    altitude_window_ft: float
    total_track_points: int
    total_neighbors_detected: int
    max_simultaneous_neighbors: int
    min_separation_nm: float
    avg_separation_nm: float
    conflict_count: int
    
    # Detailed data
    neighbors: List[NeighborAircraft]
    track_summary: Dict[str, Any]
    time_windows: List[Dict[str, Any]]


class SCATBaselineGenerator:
    """Generate baseline traffic analysis from SCAT data."""
    
    def __init__(self, scat_root: str):
        """Initialize with SCAT data directory."""
        self.scat_root = Path(scat_root)
        self.adapter = SCATAdapter(str(self.scat_root))
        
    def generate_baseline(self, ownship_file: str, radius_nm: float = 100.0, 
                         altitude_window_ft: float = 5000.0) -> BaselineReport:
        """Generate baseline analysis for given ownship."""
        log.info(f"Generating baseline for {ownship_file}")
        log.info(f"Search radius: {radius_nm} NM, altitude window: {altitude_window_ft} ft")
        
        # Load ownship data
        ownship_path = self.scat_root / ownship_file
        if not ownship_path.exists():
            raise FileNotFoundError(f"Ownship file not found: {ownship_path}")
        
        ownship_record = self.adapter.load_flight_record(ownship_path)
        if not ownship_record:
            raise ValueError(f"Failed to load ownship data from {ownship_file}")
        
        ownship_states = self.adapter.extract_aircraft_states(ownship_record)
        if not ownship_states:
            raise ValueError(f"No aircraft states found in {ownship_file}")
        
        log.info(f"Loaded ownship {ownship_record.callsign} with {len(ownship_states)} track points")
        
        # Find all potential neighbor flights
        neighbor_files = list(self.scat_root.glob("*.json"))
        neighbor_files = [f for f in neighbor_files if f.name != ownship_file]
        
        log.info(f"Analyzing {len(neighbor_files)} potential neighbor flights")
        
        # Analyze neighbors for each ownship track point
        all_neighbors = []
        time_windows = []
        conflict_count = 0
        min_separation = float('inf')
        total_separation = 0.0
        separation_count = 0
        max_simultaneous = 0
        
        for i, ownship_state in enumerate(ownship_states):
            if i % 50 == 0:  # Progress logging
                log.info(f"Processing track point {i+1}/{len(ownship_states)}")
            
            # Find neighbors at this time
            neighbors_at_time = self._find_neighbors_at_time(
                ownship_state, neighbor_files, radius_nm, altitude_window_ft
            )
            
            # Update statistics
            max_simultaneous = max(max_simultaneous, len(neighbors_at_time))
            
            for neighbor in neighbors_at_time:
                all_neighbors.append(neighbor)
                min_separation = min(min_separation, neighbor.distance_nm)
                total_separation += neighbor.distance_nm
                separation_count += 1
                
                # Check for conflict (less than 5 NM)
                if neighbor.distance_nm < 5.0:
                    conflict_count += 1
            
            # Time window summary
            time_windows.append({
                "timestamp": ownship_state.timestamp.isoformat(),
                "ownship_lat": ownship_state.latitude,
                "ownship_lon": ownship_state.longitude,
                "ownship_alt_ft": ownship_state.altitude_ft,
                "neighbor_count": len(neighbors_at_time),
                "min_separation_nm": min([n.distance_nm for n in neighbors_at_time]) if neighbors_at_time else None
            })
        
        # Calculate statistics
        avg_separation = total_separation / separation_count if separation_count > 0 else 0.0
        if min_separation == float('inf'):
            min_separation = 0.0
        
        # Track summary
        track_summary = {
            "duration_minutes": (ownship_states[-1].timestamp - ownship_states[0].timestamp).total_seconds() / 60,
            "start_time": ownship_states[0].timestamp.isoformat(),
            "end_time": ownship_states[-1].timestamp.isoformat(),
            "start_position": {"lat": ownship_states[0].latitude, "lon": ownship_states[0].longitude},
            "end_position": {"lat": ownship_states[-1].latitude, "lon": ownship_states[-1].longitude},
            "min_altitude_ft": min(s.altitude_ft for s in ownship_states),
            "max_altitude_ft": max(s.altitude_ft for s in ownship_states),
            "avg_ground_speed_kt": sum(s.ground_speed_kt for s in ownship_states) / len(ownship_states)
        }
        
        # Create report
        report = BaselineReport(
            ownship_callsign=ownship_record.callsign,
            analysis_timestamp=datetime.now().isoformat(),
            radius_filter_nm=radius_nm,
            altitude_window_ft=altitude_window_ft,
            total_track_points=len(ownship_states),
            total_neighbors_detected=len(all_neighbors),
            max_simultaneous_neighbors=max_simultaneous,
            min_separation_nm=min_separation,
            avg_separation_nm=avg_separation,
            conflict_count=conflict_count,
            neighbors=all_neighbors,
            track_summary=track_summary,
            time_windows=time_windows
        )
        
        log.info(f"Baseline analysis complete:")
        log.info(f"  Total neighbors detected: {len(all_neighbors)}")
        log.info(f"  Max simultaneous neighbors: {max_simultaneous}")
        log.info(f"  Min separation: {min_separation:.2f} NM")
        log.info(f"  Conflicts detected: {conflict_count}")
        
        return report
    
    def _find_neighbors_at_time(self, ownship_state: AircraftState, neighbor_files: List[Path],
                               radius_nm: float, altitude_window_ft: float) -> List[NeighborAircraft]:
        """Find neighboring aircraft at a specific time."""
        neighbors = []
        
        for neighbor_file in neighbor_files:
            try:
                # Load neighbor flight record
                neighbor_record = self.adapter.load_flight_record(neighbor_file)
                if not neighbor_record:
                    continue
                
                neighbor_states = self.adapter.extract_aircraft_states(neighbor_record)
                if not neighbor_states:
                    continue
                
                # Find closest time match
                closest_state = self._find_closest_time_state(ownship_state.timestamp, neighbor_states)
                if not closest_state:
                    continue
                
                # Check if within radius
                distance = haversine_nm(
                    (ownship_state.latitude, ownship_state.longitude),
                    (closest_state.latitude, closest_state.longitude)
                )
                
                if distance > radius_nm:
                    continue
                
                # Check altitude window
                relative_altitude = closest_state.altitude_ft - ownship_state.altitude_ft
                if abs(relative_altitude) > altitude_window_ft:
                    continue
                
                # Calculate bearing
                bearing = bearing_deg(
                    ownship_state.latitude, ownship_state.longitude,
                    closest_state.latitude, closest_state.longitude
                )
                
                # Estimate closest approach (simplified)
                closest_approach = self._estimate_closest_approach(ownship_state, closest_state)
                
                neighbor = NeighborAircraft(
                    callsign=neighbor_record.callsign,
                    aircraft_type=neighbor_record.aircraft_type,
                    distance_nm=distance,
                    bearing_deg=bearing,
                    altitude_ft=closest_state.altitude_ft,
                    relative_alt_ft=relative_altitude,
                    ground_speed_kt=closest_state.ground_speed_kt,
                    heading_deg=closest_state.heading_deg,
                    closest_approach_nm=closest_approach[0],
                    time_to_closest_approach_min=closest_approach[1]
                )
                
                neighbors.append(neighbor)
                
            except Exception as e:
                log.debug(f"Error processing neighbor file {neighbor_file}: {e}")
                continue
        
        return neighbors
    
    def _find_closest_time_state(self, target_time: datetime, states: List[AircraftState]) -> Optional[AircraftState]:
        """Find the aircraft state closest in time to the target."""
        if not states:
            return None
        
        min_diff = float('inf')
        closest_state = None
        
        for state in states:
            time_diff = abs((state.timestamp - target_time).total_seconds())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_state = state
        
        # Only return if within 5 minutes
        if min_diff <= 300:  # 5 minutes
            return closest_state
        
        return None
    
    def _estimate_closest_approach(self, ownship: AircraftState, intruder: AircraftState) -> Tuple[float, float]:
        """Estimate closest approach distance and time (simplified calculation)."""
        # This is a simplified linear extrapolation
        # For production, use proper geometric calculation with velocities
        
        current_distance = haversine_nm(
            (ownship.latitude, ownship.longitude),
            (intruder.latitude, intruder.longitude)
        )
        
        # Assume current distance is close to minimum (simplified)
        return current_distance, 0.0
    
    def save_baseline_report(self, report: BaselineReport, output_file: str):
        """Save baseline report to JSON and CSV files."""
        output_path = Path(output_file)
        
        # Save JSON report
        json_file = output_path.with_suffix('.json')
        with open(json_file, 'w') as f:
            # Convert to dict for JSON serialization
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2)
        
        log.info(f"Saved JSON report: {json_file}")
        
        # Save CSV summary
        csv_file = output_path.with_suffix('.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Callsign', 'Aircraft_Type', 'Distance_NM', 'Bearing_Deg', 
                'Altitude_Ft', 'Relative_Alt_Ft', 'Ground_Speed_Kt', 'Heading_Deg',
                'Closest_Approach_NM', 'Time_To_CPA_Min'
            ])
            
            # Data rows
            for neighbor in report.neighbors:
                writer.writerow([
                    neighbor.callsign, neighbor.aircraft_type, neighbor.distance_nm,
                    neighbor.bearing_deg, neighbor.altitude_ft, neighbor.relative_alt_ft,
                    neighbor.ground_speed_kt, neighbor.heading_deg,
                    neighbor.closest_approach_nm, neighbor.time_to_closest_approach_min
                ])
        
        log.info(f"Saved CSV report: {csv_file}")


def main():
    """Main entry point for SCAT baseline generator."""
    parser = argparse.ArgumentParser(description='Generate SCAT baseline traffic analysis')
    parser.add_argument('--root', required=True, help='Root directory containing SCAT files')
    parser.add_argument('--ownship', required=True, help='Ownship SCAT file name (e.g., 100000.json)')
    parser.add_argument('--radius', default='100nm', help='Search radius (default: 100nm)')
    parser.add_argument('--altwin', default='5000', help='Altitude window in feet (default: 5000)')
    parser.add_argument('--output', help='Output file prefix (default: auto-generated)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse radius
    radius_str = args.radius.lower()
    if radius_str.endswith('nm'):
        radius_nm = float(radius_str[:-2])
    else:
        radius_nm = float(radius_str)
    
    # Parse altitude window
    altitude_window_ft = float(args.altwin)
    
    try:
        # Generate baseline
        generator = SCATBaselineGenerator(args.root)
        report = generator.generate_baseline(args.ownship, radius_nm, altitude_window_ft)
        
        # Generate output filename if not provided
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ownship_name = Path(args.ownship).stem
            output_file = f"scat_baseline_{ownship_name}_{timestamp}"
        
        # Save report
        generator.save_baseline_report(report, output_file)
        
        print(f"[OK] Baseline analysis completed successfully!")
        print(f"   Ownship: {report.ownship_callsign}")
        print(f"   Neighbors detected: {report.total_neighbors_detected}")
        print(f"   Max simultaneous: {report.max_simultaneous_neighbors}")
        print(f"   Min separation: {report.min_separation_nm:.2f} NM")
        print(f"   Conflicts: {report.conflict_count}")
        print(f"   Output files: {output_file}.json, {output_file}.csv")
        
    except Exception as e:
        log.error(f"Baseline generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
