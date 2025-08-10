"""SCAT Baseline Builder - Build neighbor baseline for SCAT flights.

This module builds a baseline of neighboring flights within specified proximity
criteria (100 NM horizontal, 5000 ft vertical) for a target SCAT flight.

The baseline includes:
- Neighbor flight identification and proximity analysis
- Path segment extraction for each neighbor during proximity windows
- Serialization to baseline.jsonl and baseline_paths.geojson
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import geojson

try:
    from scipy.spatial import KDTree
    import numpy as np
    scipy_available = True
except ImportError:
    logging.warning("scipy not available. Using fallback spatial search.")
    scipy_available = False
    KDTree = None
    np = None

from .scat_adapter import SCATAdapter, SCATFlightRecord
from .schemas import AircraftState
from .geodesy import haversine_nm

logger = logging.getLogger(__name__)


@dataclass
class NeighborSegment:
    """A path segment where a neighbor is in proximity to the ownship."""
    neighbor_id: str
    start_time: datetime
    end_time: datetime
    min_distance_nm: float
    min_altitude_diff_ft: float
    ownship_positions: List[Tuple[float, float, float]]  # (lat, lon, alt_ft)
    neighbor_positions: List[Tuple[float, float, float]]  # (lat, lon, alt_ft)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'neighbor_id': self.neighbor_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_min': (self.end_time - self.start_time).total_seconds() / 60.0,
            'min_distance_nm': self.min_distance_nm,
            'min_altitude_diff_ft': self.min_altitude_diff_ft,
            'ownship_positions': self.ownship_positions,
            'neighbor_positions': self.neighbor_positions,
            'position_count': len(self.ownship_positions)
        }


@dataclass
class NeighborBaseline:
    """Complete baseline data for a target ownship flight."""
    ownship_id: str
    ownship_flight_record: SCATFlightRecord
    analysis_time: datetime
    proximity_criteria: Dict[str, float]
    neighbor_segments: List[NeighborSegment]
    total_neighbors: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ownship_id': self.ownship_id,
            'analysis_time': self.analysis_time.isoformat(),
            'proximity_criteria': self.proximity_criteria,
            'total_neighbors': self.total_neighbors,
            'neighbor_count': len(self.neighbor_segments),
            'segments': [seg.to_dict() for seg in self.neighbor_segments]
        }


class SCATBaselineBuilder:
    """Build SCAT neighbor baseline for proximity analysis."""
    
    def __init__(self, scat_root: str, 
                 proximity_radius_nm: float = 100.0,
                 altitude_window_ft: float = 5000.0,
                 min_segment_duration_min: float = 2.0):
        """Initialize baseline builder.
        
        Args:
            scat_root: Root directory containing SCAT files
            proximity_radius_nm: Horizontal proximity threshold (default: 100 NM)
            altitude_window_ft: Vertical proximity threshold (default: 5000 ft)
            min_segment_duration_min: Minimum segment duration to include (default: 2 min)
        """
        self.scat_root = Path(scat_root)
        self.adapter = SCATAdapter(str(self.scat_root))
        
        # Proximity criteria
        self.proximity_radius_nm = proximity_radius_nm
        self.altitude_window_ft = altitude_window_ft
        self.min_segment_duration_min = min_segment_duration_min
        
        # Spatial indexing
        self.use_kdtree = scipy_available
        self.flight_index = {}  # flight_id -> flight_data
        self.spatial_index = None  # KDTree for fast spatial queries
        
        logger.info(f"SCATBaselineBuilder initialized with {proximity_radius_nm} NM / {altitude_window_ft} ft criteria")
    
    def build_baseline(self, ownship_file: str, max_neighbors: int = 50) -> NeighborBaseline:
        """Build complete neighbor baseline for an ownship flight.
        
        Args:
            ownship_file: Path to ownship SCAT file (relative to scat_root)
            max_neighbors: Maximum number of neighbor files to process
            
        Returns:
            NeighborBaseline with all proximity segments
        """
        logger.info(f"Building baseline for ownship: {ownship_file}")
        start_time = time.time()
        
        # Load ownship flight
        ownship_path = self.scat_root / ownship_file
        ownship_record = self.adapter.load_flight_record(ownship_path)
        if not ownship_record:
            raise ValueError(f"Failed to load ownship flight: {ownship_path}")
        
        ownship_states = self.adapter.extract_aircraft_states(ownship_record)
        if not ownship_states:
            raise ValueError(f"No aircraft states found for ownship: {ownship_path}")
        
        logger.info(f"Loaded ownship {ownship_record.callsign} with {len(ownship_states)} states")
        
        # Index all available SCAT files
        self._index_scat_files(max_neighbors)
        
        # Find proximity segments for each neighbor
        neighbor_segments = []
        processed_neighbors = 0
        
        for neighbor_id, neighbor_data in self.flight_index.items():
            if neighbor_id == ownship_record.callsign:
                continue  # Skip self
            
            segments = self._find_proximity_segments(ownship_states, neighbor_data['states'])
            neighbor_segments.extend(segments)
            processed_neighbors += 1
            
            if processed_neighbors % 10 == 0:
                logger.info(f"Processed {processed_neighbors} neighbors, found {len(neighbor_segments)} segments")
        
        # Create baseline result
        baseline = NeighborBaseline(
            ownship_id=ownship_record.callsign,
            ownship_flight_record=ownship_record,
            analysis_time=datetime.now(),
            proximity_criteria={
                'horizontal_radius_nm': self.proximity_radius_nm,
                'vertical_window_ft': self.altitude_window_ft,
                'min_segment_duration_min': self.min_segment_duration_min
            },
            neighbor_segments=neighbor_segments,
            total_neighbors=processed_neighbors
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Baseline building completed in {elapsed_time:.1f}s: "
                   f"{len(neighbor_segments)} segments from {processed_neighbors} neighbors")
        
        return baseline
    
    def _index_scat_files(self, max_files: int):
        """Index SCAT files for neighbor analysis."""
        logger.info("Indexing SCAT files for neighbor analysis...")
        
        scat_files = list(self.scat_root.glob("*.json"))
        if len(scat_files) > max_files:
            logger.info(f"Limiting analysis to first {max_files} files (found {len(scat_files)})")
            scat_files = scat_files[:max_files]
        
        self.flight_index = {}
        spatial_points = []
        flight_references = []
        
        for scat_file in scat_files:
            try:
                # Load flight record
                record = self.adapter.load_flight_record(scat_file)
                if not record:
                    continue
                
                # Extract aircraft states
                states = self.adapter.extract_aircraft_states(record)
                if not states or len(states) < 5:  # Skip very short flights
                    continue
                
                # Store flight data
                self.flight_index[record.callsign] = {
                    'record': record,
                    'states': states,
                    'file_path': scat_file
                }
                
                # Build spatial index points if using KDTree
                if self.use_kdtree:
                    for state in states:
                        # Convert lat/lon to approximate cartesian for spatial indexing
                        x = state.longitude * 111.32 * np.cos(np.radians(state.latitude))  # km
                        y = state.latitude * 110.54  # km
                        z = state.altitude_ft * 0.0003048  # km
                        
                        spatial_points.append([x, y, z])
                        flight_references.append(record.callsign)
                
            except Exception as e:
                logger.debug(f"Error processing {scat_file}: {e}")
                continue
        
        logger.info(f"Indexed {len(self.flight_index)} flights")
        
        # Build spatial index if available
        if self.use_kdtree and spatial_points:
            self.spatial_index = KDTree(spatial_points)
            self.flight_references = flight_references
            logger.info(f"Built spatial index with {len(spatial_points)} points")
    
    def _find_proximity_segments(self, ownship_states: List[AircraftState], 
                               neighbor_states: List[AircraftState]) -> List[NeighborSegment]:
        """Find all proximity segments between ownship and neighbor."""
        if not neighbor_states:
            return []
        
        neighbor_id = neighbor_states[0].aircraft_id
        segments = []
        
        # Track proximity windows
        in_proximity = False
        segment_start = None
        segment_ownship_positions = []
        segment_neighbor_positions = []
        segment_min_distance = float('inf')
        segment_min_alt_diff = float('inf')
        
        # Process ownship states chronologically
        ownship_states = sorted(ownship_states, key=lambda s: s.timestamp)
        neighbor_states = sorted(neighbor_states, key=lambda s: s.timestamp)
        
        # For each ownship state, find the closest neighbor state in time
        for ownship_state in ownship_states:
            # Find closest neighbor state in time
            closest_neighbor = self._find_closest_neighbor_in_time(ownship_state, neighbor_states)
            if not closest_neighbor:
                continue
            
            # Calculate proximity
            distance_nm = haversine_nm(
                (ownship_state.latitude, ownship_state.longitude),
                (closest_neighbor.latitude, closest_neighbor.longitude)
            )
            altitude_diff_ft = abs(ownship_state.altitude_ft - closest_neighbor.altitude_ft)
            
            # Check if in proximity
            is_proximate = (distance_nm <= self.proximity_radius_nm and 
                          altitude_diff_ft <= self.altitude_window_ft)
            
            if is_proximate:
                if not in_proximity:
                    # Start new proximity segment
                    in_proximity = True
                    segment_start = ownship_state.timestamp
                    segment_ownship_positions = []
                    segment_neighbor_positions = []
                    segment_min_distance = float('inf')
                    segment_min_alt_diff = float('inf')
                
                # Add to current segment
                segment_ownship_positions.append((
                    ownship_state.latitude, ownship_state.longitude, ownship_state.altitude_ft
                ))
                segment_neighbor_positions.append((
                    closest_neighbor.latitude, closest_neighbor.longitude, closest_neighbor.altitude_ft
                ))
                
                # Update minimums
                segment_min_distance = min(segment_min_distance, distance_nm)
                segment_min_alt_diff = min(segment_min_alt_diff, altitude_diff_ft)
                
            else:
                if in_proximity:
                    # End current proximity segment
                    segment_duration = (ownship_state.timestamp - segment_start).total_seconds() / 60.0
                    
                    if segment_duration >= self.min_segment_duration_min:
                        segment = NeighborSegment(
                            neighbor_id=neighbor_id,
                            start_time=segment_start,
                            end_time=ownship_state.timestamp,
                            min_distance_nm=segment_min_distance,
                            min_altitude_diff_ft=segment_min_alt_diff,
                            ownship_positions=segment_ownship_positions,
                            neighbor_positions=segment_neighbor_positions
                        )
                        segments.append(segment)
                    
                    in_proximity = False
        
        # Handle case where proximity continues to end
        if in_proximity and segment_start:
            segment_duration = (ownship_states[-1].timestamp - segment_start).total_seconds() / 60.0
            if segment_duration >= self.min_segment_duration_min:
                segment = NeighborSegment(
                    neighbor_id=neighbor_id,
                    start_time=segment_start,
                    end_time=ownship_states[-1].timestamp,
                    min_distance_nm=segment_min_distance,
                    min_altitude_diff_ft=segment_min_alt_diff,
                    ownship_positions=segment_ownship_positions,
                    neighbor_positions=segment_neighbor_positions
                )
                segments.append(segment)
        
        return segments
    
    def _find_closest_neighbor_in_time(self, ownship_state: AircraftState, 
                                     neighbor_states: List[AircraftState]) -> Optional[AircraftState]:
        """Find neighbor state closest in time to ownship state."""
        if not neighbor_states:
            return None
        
        min_time_diff = float('inf')
        closest_state = None
        
        for neighbor_state in neighbor_states:
            time_diff = abs((ownship_state.timestamp - neighbor_state.timestamp).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_state = neighbor_state
        
        # Only return if within reasonable time window (e.g., 10 minutes)
        if min_time_diff <= 600:  # 10 minutes
            return closest_state
        
        return None
    
    def save_baseline(self, baseline: NeighborBaseline, output_dir: str):
        """Save baseline to files.
        
        Args:
            baseline: Baseline data to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save baseline.jsonl (one line per segment)
        baseline_file = output_path / f"baseline_{baseline.ownship_id}.jsonl"
        with open(baseline_file, 'w') as f:
            # Write header
            header = {
                'type': 'baseline_header',
                'ownship_id': baseline.ownship_id,
                'analysis_time': baseline.analysis_time.isoformat(),
                'proximity_criteria': baseline.proximity_criteria,
                'total_neighbors': baseline.total_neighbors,
                'segment_count': len(baseline.neighbor_segments)
            }
            f.write(json.dumps(header) + '\n')
            
            # Write segments
            for segment in baseline.neighbor_segments:
                segment_data = {'type': 'proximity_segment', **segment.to_dict()}
                f.write(json.dumps(segment_data) + '\n')
        
        logger.info(f"Saved baseline to: {baseline_file}")
        
        # Save baseline_paths.geojson (geographic visualization)
        geojson_file = output_path / f"baseline_paths_{baseline.ownship_id}.geojson"
        features = []
        
        # Add ownship path (derived from track points or states)
        # Since SCATFlightRecord doesn't have waypoints, we'll create a path from track points
        ownship_coords = []
        if hasattr(baseline.ownship_flight_record, 'track_points') and baseline.ownship_flight_record.track_points:
            # Use track points if available
            for track_point in baseline.ownship_flight_record.track_points:
                if 'lat' in track_point and 'lon' in track_point and 'alt' in track_point:
                    ownship_coords.append((track_point['lon'], track_point['lat'], track_point['alt']))
        
        if ownship_coords:
            ownship_feature = geojson.Feature(
                geometry=geojson.LineString(ownship_coords),
                properties={
                    'type': 'ownship_path',
                    'flight_id': baseline.ownship_id,
                    'waypoint_count': len(ownship_coords)
                }
            )
            features.append(ownship_feature)
        
        # Add neighbor segments
        for segment in baseline.neighbor_segments:
            if segment.neighbor_positions:
                neighbor_coords = [(pos[1], pos[0], pos[2]) for pos in segment.neighbor_positions]  # lon, lat, alt
                neighbor_feature = geojson.Feature(
                    geometry=geojson.LineString(neighbor_coords),
                    properties={
                        'type': 'neighbor_segment',
                        'neighbor_id': segment.neighbor_id,
                        'start_time': segment.start_time.isoformat(),
                        'end_time': segment.end_time.isoformat(),
                        'min_distance_nm': segment.min_distance_nm,
                        'min_altitude_diff_ft': segment.min_altitude_diff_ft
                    }
                )
                features.append(neighbor_feature)
        
        feature_collection = geojson.FeatureCollection(features)
        with open(geojson_file, 'w') as f:
            geojson.dump(feature_collection, f, indent=2)
        
        logger.info(f"Saved GeoJSON to: {geojson_file}")
        
        # Save summary
        summary_file = output_path / f"baseline_summary_{baseline.ownship_id}.json"
        summary = {
            'ownship_id': baseline.ownship_id,
            'analysis_time': baseline.analysis_time.isoformat(),
            'proximity_criteria': baseline.proximity_criteria,
            'total_neighbors_analyzed': baseline.total_neighbors,
            'neighbors_with_proximity': len(set(seg.neighbor_id for seg in baseline.neighbor_segments)),
            'total_proximity_segments': len(baseline.neighbor_segments),
            'total_proximity_duration_min': sum(
                (seg.end_time - seg.start_time).total_seconds() / 60.0 
                for seg in baseline.neighbor_segments
            ),
            'min_distance_encountered_nm': min(
                seg.min_distance_nm for seg in baseline.neighbor_segments
            ) if baseline.neighbor_segments else None,
            'segments_by_neighbor': {
                neighbor_id: len([seg for seg in baseline.neighbor_segments if seg.neighbor_id == neighbor_id])
                for neighbor_id in set(seg.neighbor_id for seg in baseline.neighbor_segments)
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to: {summary_file}")


def main():
    """Example usage of SCAT baseline builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SCAT neighbor baseline')
    parser.add_argument('--root', required=True, help='SCAT root directory')
    parser.add_argument('--ownship', required=True, help='Ownship SCAT file (e.g., 100000.json)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--radius-nm', type=float, default=100.0, help='Proximity radius in NM')
    parser.add_argument('--altitude-ft', type=float, default=5000.0, help='Altitude window in feet')
    parser.add_argument('--max-neighbors', type=int, default=50, help='Maximum neighbors to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create baseline builder
        builder = SCATBaselineBuilder(
            scat_root=args.root,
            proximity_radius_nm=args.radius_nm,
            altitude_window_ft=args.altitude_ft
        )
        
        # Build baseline
        baseline = builder.build_baseline(args.ownship, args.max_neighbors)
        
        # Save results
        builder.save_baseline(baseline, args.output)
        
        # Print summary
        print(f"[OK] SCAT baseline completed for {args.ownship}")
        print(f"   Neighbors analyzed: {baseline.total_neighbors}")
        print(f"   Proximity segments: {len(baseline.neighbor_segments)}")
        print(f"   Neighbors with proximity: {len(set(seg.neighbor_id for seg in baseline.neighbor_segments))}")
        print(f"   Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Baseline building failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
