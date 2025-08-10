"""SCAT (Standard CDR Testing) Dataset Adapter.

This module provides an adapter to load and process SCAT aviation dataset files
for use with the ATC LLM-BlueSky CDR system. SCAT files contain real flight
trajectory data in JSON format with ASTERIX Category 062 surveillance standards.

The adapter converts SCAT data into BlueSky-compatible aircraft states for 
scenario replay and CDR system testing.
"""

import json
import logging
import os
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from scipy.spatial import KDTree
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    logging.warning("scipy not available. Using fallback spatial search.")
    SCIPY_AVAILABLE = False
    KDTree = None
    np = None

from .schemas import AircraftState

logger = logging.getLogger(__name__)


@dataclass
class SCATFlightRecord:
    """Parsed SCAT flight record with extracted key information."""
    
    # Flight identification
    callsign: str
    aircraft_type: str
    flight_rules: str  # I = IFR, V = VFR
    wtc: str          # Wake Turbulence Category
    
    # Route information
    adep: str         # Departure airport
    ades: str         # Destination airport
    
    # Surveillance track points
    track_points: List[Dict[str, Any]]
    
    # Time bounds
    start_time: datetime
    end_time: datetime
    
    # Additional metadata
    centre_id: Optional[int] = None
    rvsm_capable: bool = False


@dataclass
class VicinityQueryPerformance:
    """Performance metrics for vicinity queries."""
    
    query_count: int = 0
    total_query_time_ms: float = 0.0
    min_query_time_ms: float = float('inf')
    max_query_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    
    def add_query_time(self, query_time_ms: float):
        """Add a new query time and update statistics."""
        self.query_count += 1
        self.total_query_time_ms += query_time_ms
        self.min_query_time_ms = min(self.min_query_time_ms, query_time_ms)
        self.max_query_time_ms = max(self.max_query_time_ms, query_time_ms)
        self.avg_query_time_ms = self.total_query_time_ms / self.query_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        return {
            'query_count': self.query_count,
            'total_time_ms': round(self.total_query_time_ms, 2),
            'avg_time_ms': round(self.avg_query_time_ms, 2),
            'min_time_ms': round(self.min_query_time_ms, 2),
            'max_time_ms': round(self.max_query_time_ms, 2),
            'queries_per_second': round(self.query_count / (self.total_query_time_ms / 1000.0), 2) if self.total_query_time_ms > 0 else 0
        }


class VicinityIndex:
    """High-performance vicinity indexing with spatial search and performance tracking."""
    
    def __init__(self, use_kdtree: bool = True):
        """Initialize vicinity index.
        
        Args:
            use_kdtree: Whether to use KDTree for spatial indexing
        """
        self.use_kdtree = use_kdtree and SCIPY_AVAILABLE
        self.spatial_index = None
        self.indexed_positions = []  # AircraftState objects
        self.position_coords = []    # ECEF coordinates for each position
        self.performance = VicinityQueryPerformance()
        self.index_build_time_ms = 0.0
        
        logger.info(f"VicinityIndex initialized with KDTree: {'enabled' if self.use_kdtree else 'disabled'}")
    
    def build_index(self, aircraft_states: List[AircraftState]) -> None:
        """Build spatial index from aircraft states with performance tracking.
        
        Args:
            aircraft_states: List of aircraft states to index
        """
        start_time = time.perf_counter()
        
        if not aircraft_states:
            logger.warning("No aircraft states provided for vicinity indexing")
            return
        
        self.indexed_positions = []
        self.position_coords = []
        
        if self.use_kdtree:
            try:
                # Convert to ECEF coordinates for accurate distance calculations
                points_3d = []
                
                for state in aircraft_states:
                    # Convert altitude from feet to meters for ECEF
                    alt_m = state.altitude_ft * 0.3048
                    x, y, z = self._lat_lon_to_ecef(state.latitude, state.longitude, alt_m)
                    
                    points_3d.append([x, y, z])
                    self.indexed_positions.append(state)
                    self.position_coords.append((x, y, z))
                
                if points_3d:
                    self.spatial_index = KDTree(np.array(points_3d))
                    
                    build_time = (time.perf_counter() - start_time) * 1000.0
                    self.index_build_time_ms = build_time
                    
                    logger.info(f"VicinityIndex: Built KDTree with {len(points_3d)} positions in {build_time:.1f}ms")
                else:
                    logger.warning("No valid positions found for KDTree indexing")
                    
            except Exception as e:
                logger.warning(f"Failed to build KDTree index: {e}, falling back to linear search")
                self.use_kdtree = False
                self.spatial_index = None
        
        if not self.use_kdtree:
            # Simple list-based indexing for fallback
            self.indexed_positions = aircraft_states[:]
            build_time = (time.perf_counter() - start_time) * 1000.0
            self.index_build_time_ms = build_time
            logger.info(f"VicinityIndex: Built linear index with {len(aircraft_states)} positions in {build_time:.1f}ms")
    
    def query_vicinity(self, ownship_state: AircraftState, 
                      radius_nm: float = 100.0, 
                      altitude_window_ft: float = 5000.0) -> List[AircraftState]:
        """Query aircraft in vicinity with performance tracking.
        
        Args:
            ownship_state: Ownship aircraft state
            radius_nm: Search radius in nautical miles
            altitude_window_ft: Altitude window in feet
            
        Returns:
            List of aircraft states within vicinity
        """
        query_start = time.perf_counter()
        
        try:
            if self.use_kdtree and self.spatial_index:
                vicinity_aircraft = self._query_kdtree_vicinity(
                    ownship_state, radius_nm, altitude_window_ft
                )
            else:
                vicinity_aircraft = self._query_linear_vicinity(
                    ownship_state, radius_nm, altitude_window_ft
                )
            
            # Record performance metrics
            query_time_ms = (time.perf_counter() - query_start) * 1000.0
            self.performance.add_query_time(query_time_ms)
            
            logger.debug(f"VicinityIndex query: found {len(vicinity_aircraft)} aircraft in "
                        f"{query_time_ms:.2f}ms (method: {'KDTree' if self.use_kdtree else 'linear'})")
            
            return vicinity_aircraft
            
        except Exception as e:
            query_time_ms = (time.perf_counter() - query_start) * 1000.0
            self.performance.add_query_time(query_time_ms)
            logger.error(f"VicinityIndex query failed in {query_time_ms:.2f}ms: {e}")
            return []
    
    def _query_kdtree_vicinity(self, ownship_state: AircraftState, 
                              radius_nm: float, altitude_window_ft: float) -> List[AircraftState]:
        """Query vicinity using KDTree spatial index."""
        # Convert ownship position to ECEF
        alt_m = ownship_state.altitude_ft * 0.3048
        ox, oy, oz = self._lat_lon_to_ecef(
            ownship_state.latitude, 
            ownship_state.longitude, 
            alt_m
        )
        
        # Convert radius from NM to meters (1 NM = 1852 m)
        radius_m = radius_nm * 1852.0
        
        # Query KDTree for points within radius
        indices = self.spatial_index.query_ball_point([ox, oy, oz], radius_m)
        
        vicinity_aircraft = []
        for idx in indices:
            if idx < len(self.indexed_positions):
                candidate = self.indexed_positions[idx]
                
                # Skip self
                if (candidate.aircraft_id == ownship_state.aircraft_id and 
                    candidate.timestamp == ownship_state.timestamp):
                    continue
                
                # Check altitude window
                alt_diff = abs(candidate.altitude_ft - ownship_state.altitude_ft)
                if alt_diff <= altitude_window_ft:
                    vicinity_aircraft.append(candidate)
        
        return vicinity_aircraft
    
    def _query_linear_vicinity(self, ownship_state: AircraftState, 
                              radius_nm: float, altitude_window_ft: float) -> List[AircraftState]:
        """Query vicinity using linear search fallback."""
        vicinity_aircraft = []
        
        for state in self.indexed_positions:
            if (state.aircraft_id == ownship_state.aircraft_id and 
                state.timestamp == ownship_state.timestamp):
                continue
            
            # Calculate great circle distance
            from .geodesy import haversine_nm
            distance = haversine_nm(
                (ownship_state.latitude, ownship_state.longitude),
                (state.latitude, state.longitude)
            )
            
            alt_diff = abs(state.altitude_ft - ownship_state.altitude_ft)
            
            if distance <= radius_nm and alt_diff <= altitude_window_ft:
                vicinity_aircraft.append(state)
        
        return vicinity_aircraft
    
    def _lat_lon_to_ecef(self, lat: float, lon: float, alt_m: float = 0.0) -> Tuple[float, float, float]:
        """Convert latitude/longitude/altitude to ECEF coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            alt_m: Altitude in meters (default: 0)
            
        Returns:
            (x, y, z) coordinates in ECEF (meters)
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # WGS84 constants
        a = 6378137.0  # Semi-major axis
        e2 = 6.69437999014e-3  # First eccentricity squared
        
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        x = (N + alt_m) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt_m) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - e2) + alt_m) * math.sin(lat_rad)
        
        return x, y, z

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.performance.get_summary()
        summary.update({
            'index_build_time_ms': round(self.index_build_time_ms, 2),
            'index_type': 'KDTree' if self.use_kdtree else 'Linear',
            'indexed_positions': len(self.indexed_positions),
            'spatial_index_available': self.spatial_index is not None
        })
        return summary
    
    def log_performance_summary(self):
        """Log performance summary to logger."""
        summary = self.get_performance_summary()
        logger.info(f"VicinityIndex Performance Summary:")
        logger.info(f"  Index type: {summary['index_type']}")
        logger.info(f"  Build time: {summary['index_build_time_ms']}ms")
        logger.info(f"  Indexed positions: {summary['indexed_positions']}")
        logger.info(f"  Queries executed: {summary['query_count']}")
        logger.info(f"  Avg query time: {summary['avg_time_ms']}ms")
        logger.info(f"  Query throughput: {summary['queries_per_second']} queries/sec")


class SCATAdapter:
    """Adapter for loading and processing SCAT dataset files."""
    
    def __init__(self, dataset_path: str):
        """Initialize SCAT adapter.
        
        Args:
            dataset_path: Path to extracted SCAT dataset directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"SCAT dataset path not found: {dataset_path}")
        
        self.flight_files = list(self.dataset_path.glob("*.json"))
        self.flight_files = [f for f in self.flight_files 
                           if f.name not in ['airspace', 'grib_met']]
        
        # Initialize VicinityIndex for high-performance vicinity queries
        self.vicinity_index = VicinityIndex(use_kdtree=SCIPY_AVAILABLE)
        
        # Legacy attributes for backward compatibility
        self.use_kdtree = SCIPY_AVAILABLE
        self.spatial_index = None  # Deprecated: use vicinity_index instead
        self.flight_positions = []  # Deprecated: use vicinity_index instead
        
        logger.info(f"Found {len(self.flight_files)} SCAT flight records")
        logger.info(f"VicinityIndex initialized with KDTree: {'enabled' if SCIPY_AVAILABLE else 'disabled'}")
    
    def lat_lon_to_ecef(self, lat: float, lon: float, alt_m: float = 0.0) -> Tuple[float, float, float]:
        """Convert latitude/longitude/altitude to ECEF coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            alt_m: Altitude in meters (default: 0)
            
        Returns:
            (x, y, z) coordinates in ECEF (meters)
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # WGS84 constants
        a = 6378137.0  # Semi-major axis
        e2 = 6.69437999014e-3  # First eccentricity squared
        
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        x = (N + alt_m) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt_m) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - e2) + alt_m) * math.sin(lat_rad)
        
        return x, y, z
    
    def build_spatial_index(self, aircraft_states: List[AircraftState]) -> None:
        """Build spatial index for efficient vicinity queries with performance tracking.
        
        Args:
            aircraft_states: List of aircraft states to index
        """
        # Use the new VicinityIndex for improved performance tracking
        self.vicinity_index.build_index(aircraft_states)
        
        # For backward compatibility, also update legacy attributes
        self.flight_positions = self.vicinity_index.indexed_positions[:]
        self.spatial_index = self.vicinity_index.spatial_index
        
        # Log performance summary
        self.vicinity_index.log_performance_summary()
    
    def find_vicinity_aircraft(self, ownship_state: AircraftState, 
                              radius_nm: float = 100.0, 
                              altitude_window_ft: float = 5000.0) -> List[AircraftState]:
        """Find aircraft in vicinity using high-performance VicinityIndex.
        
        Args:
            ownship_state: Ownship aircraft state
            radius_nm: Search radius in nautical miles
            altitude_window_ft: Altitude window in feet
            
        Returns:
            List of aircraft states within vicinity
        """
        return self.vicinity_index.query_vicinity(ownship_state, radius_nm, altitude_window_ft)
    
    def get_vicinity_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for vicinity queries.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.vicinity_index.get_performance_summary()
    
    def log_vicinity_performance(self):
        """Log vicinity query performance summary."""
        self.vicinity_index.log_performance_summary()
    
    def _find_vicinity_linear(self, ownship_state: AircraftState, 
                             radius_nm: float, altitude_window_ft: float) -> List[AircraftState]:
        """Fallback linear search for vicinity aircraft (deprecated - use VicinityIndex)."""
        # This method is kept for backward compatibility
        # but now delegates to the VicinityIndex
        return self.vicinity_index.query_vicinity(ownship_state, radius_nm, altitude_window_ft)
    
    
    def export_to_jsonl(self, ownship_id: str, output_dir: str, 
                        vicinity_radius_nm: float = 100.0,
                        altitude_window_ft: float = 5000.0) -> Tuple[str, str]:
        """Export ownship track and base intruders to normalized JSONL format.
        
        Args:
            ownship_id: Target ownship callsign
            output_dir: Output directory for JSONL files
            vicinity_radius_nm: Vicinity search radius in NM
            altitude_window_ft: Altitude window in feet
            
        Returns:
            Tuple of (ownship_track_path, base_intruders_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all aircraft states
        all_states = self.load_scenario(max_flights=len(self.flight_files), time_window_minutes=0)
        if not all_states:
            raise ValueError("No aircraft states loaded from SCAT data")
        
        # Build spatial index for vicinity searches
        self.build_spatial_index(all_states)
        
        # Separate ownship and other aircraft
        ownship_states = [s for s in all_states if s.aircraft_id == ownship_id]
        if not ownship_states:
            raise ValueError(f"No states found for ownship {ownship_id}")
        
        # Export ownship track
        ownship_file = output_path / f"ownship_track_{ownship_id}.jsonl"
        with open(ownship_file, 'w') as f:
            for state in sorted(ownship_states, key=lambda s: s.timestamp):
                record = {
                    'timestamp': state.timestamp.isoformat(),
                    'aircraft_id': state.aircraft_id,
                    'callsign': getattr(state, 'callsign', state.aircraft_id),
                    'latitude': state.latitude,
                    'longitude': state.longitude,
                    'altitude_ft': state.altitude_ft,
                    'ground_speed_kt': state.ground_speed_kt,
                    'heading_deg': state.heading_deg,
                    'vertical_speed_fpm': state.vertical_speed_fpm,
                    'aircraft_type': getattr(state, 'aircraft_type', 'UNKNOWN')
                }
                f.write(json.dumps(record) + '\n')
        
        # Find and export base intruders (vicinity aircraft)
        intruders_file = output_path / f"base_intruders_{ownship_id}.jsonl"
        intruder_states = []
        intruder_ids = set()  # Track unique aircraft IDs to avoid duplicates
        
        # For each ownship position, find vicinity aircraft
        for ownship_state in ownship_states:
            vicinity_aircraft = self.find_vicinity_aircraft(
                ownship_state, vicinity_radius_nm, altitude_window_ft
            )
            for aircraft in vicinity_aircraft:
                aircraft_key = f"{aircraft.aircraft_id}_{aircraft.timestamp.isoformat()}"
                if aircraft_key not in intruder_ids:
                    intruder_ids.add(aircraft_key)
                    intruder_states.append(aircraft)
        
        # Export intruder states
        with open(intruders_file, 'w') as f:
            for state in sorted(intruder_states, key=lambda s: (s.aircraft_id, s.timestamp)):
                record = {
                    'timestamp': state.timestamp.isoformat(),
                    'aircraft_id': state.aircraft_id,
                    'callsign': getattr(state, 'callsign', state.aircraft_id),
                    'latitude': state.latitude,
                    'longitude': state.longitude,
                    'altitude_ft': state.altitude_ft,
                    'ground_speed_kt': state.ground_speed_kt,
                    'heading_deg': state.heading_deg,
                    'vertical_speed_fpm': state.vertical_speed_fpm,
                    'aircraft_type': getattr(state, 'aircraft_type', 'UNKNOWN'),
                    'proximity_to_ownship': True
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Exported SCAT data to JSONL:")
        logger.info(f"  Ownship track: {ownship_file} ({len(ownship_states)} records)")
        logger.info(f"  Base intruders: {intruders_file} ({len(intruder_states)} records)")
        
        return str(ownship_file), str(intruders_file)

    def get_flight_summary(self) -> Dict[str, Any]:
        """Get summary of flights in the dataset.
        
        Returns:
            Dictionary with flight summary information
        """
        summary = {
            'total_flights': len(self.flight_files),
            'dataset_path': str(self.dataset_path),
            'use_kdtree': self.use_kdtree,
            'spatial_index_available': self.spatial_index is not None
        }
        
        # Add vicinity performance if available
        if hasattr(self, 'vicinity_index'):
            summary.update(self.vicinity_index.get_performance_summary())
        
        return summary

    def load_flight_record(self, file_path: Path) -> Optional[SCATFlightRecord]:
        """Load a single SCAT flight record from JSON file.
        
        Args:
            file_path: Path to SCAT JSON file
            
        Returns:
            Parsed flight record or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract flight plan base information
            fpl_base = data.get('fpl', {}).get('fpl_base', [])
            if not fpl_base:
                logger.warning(f"No flight plan base data in {file_path.name}")
                return None
            
            base_info = fpl_base[0]  # Take first entry
            
            # Extract surveillance track (try both 'plots' and 'surveillance_track')
            track = data.get('plots', data.get('surveillance_track', []))
            if not track:
                logger.warning(f"No surveillance track/plots data in {file_path.name}")
                return None
            
            # Extract centre control info
            centre_ctrl = data.get('centre_ctrl', [])
            centre_id = centre_ctrl[0]['centre_id'] if centre_ctrl else None
            
            # Parse time bounds
            start_time = self._parse_timestamp(track[0]['time_of_track'])
            end_time = self._parse_timestamp(track[-1]['time_of_track'])
            
            return SCATFlightRecord(
                callsign=base_info.get('callsign', 'UNKNOWN'),
                aircraft_type=base_info.get('aircraft_type', 'UNKN'),
                flight_rules=base_info.get('flight_rules', 'I'),
                wtc=base_info.get('wtc', 'M'),
                adep=base_info.get('adep', 'UNKN'),
                ades=base_info.get('ades', 'UNKN'),
                track_points=track,
                start_time=start_time,
                end_time=end_time,
                centre_id=centre_id,
                rvsm_capable=base_info.get('equip_status_rvsm', False)
            )
            
        except Exception as e:
            logger.error(f"Error loading SCAT file {file_path.name}: {e}")
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse SCAT timestamp string to datetime object.
        
        Args:
            timestamp_str: Timestamp in format "2016-10-18T19:19:03.898437"
            
        Returns:
            Parsed datetime object in UTC
        """
        # Handle microseconds properly
        if '.' in timestamp_str:
            base_time, microseconds = timestamp_str.split('.')
            # Truncate microseconds to 6 digits if longer
            microseconds = microseconds[:6].ljust(6, '0')
            timestamp_str = f"{base_time}.{microseconds}"
        
        try:
            return datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback parsing
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return datetime.now(timezone.utc)
    
    def extract_aircraft_states(self, flight_record: SCATFlightRecord, 
                              time_filter: Optional[Tuple[datetime, datetime]] = None) -> List[AircraftState]:
        """Extract aircraft states from SCAT flight record.
        
        Args:
            flight_record: Parsed SCAT flight record
            time_filter: Optional (start_time, end_time) filter
            
        Returns:
            List of aircraft states suitable for BlueSky
        """
        states = []
        
        for track_point in flight_record.track_points:
            try:
                # Parse timestamp
                timestamp = self._parse_timestamp(track_point['time_of_track'])
                
                # Apply time filter if specified
                if time_filter:
                    start_time, end_time = time_filter
                    if timestamp < start_time or timestamp > end_time:
                        continue
                
                # Extract position (I062/105)
                position = track_point.get('I062/105', {})
                lat = position.get('lat')
                lon = position.get('lon')
                
                if lat is None or lon is None:
                    continue
                
                # Extract flight level (I062/136) - convert to feet
                fl_data = track_point.get('I062/136', {})
                measured_fl = fl_data.get('measured_flight_level')
                altitude_ft = measured_fl * 100 if measured_fl else 0  # FL350 = 35000 ft
                
                # Extract velocity (I062/185) - in knots
                velocity = track_point.get('I062/185', {})
                vx = velocity.get('vx', 0)  # East velocity component
                vy = velocity.get('vy', 0)  # North velocity component
                
                # Calculate ground speed and heading
                import math
                ground_speed_kt = math.sqrt(vx**2 + vy**2)
                heading_deg = math.degrees(math.atan2(vx, vy)) % 360
                
                # Extract rate of climb/descent (I062/220)
                rocd_data = track_point.get('I062/220', {})
                vertical_rate_fpm = rocd_data.get('rocd', 0) * 60  # Convert ft/s to ft/min
                
                # Create aircraft state
                state = AircraftState(
                    aircraft_id=flight_record.callsign,
                    latitude=lat,
                    longitude=lon,
                    altitude_ft=altitude_ft,
                    ground_speed_kt=ground_speed_kt,
                    heading_deg=heading_deg,
                    vertical_speed_fpm=vertical_rate_fpm,
                    timestamp=timestamp
                )
                
                states.append(state)
                
            except Exception as e:
                logger.warning(f"Error processing track point for {flight_record.callsign}: {e}")
                continue
        
        logger.info(f"Extracted {len(states)} states for {flight_record.callsign}")
        return states
    
    def load_scenario(self, max_flights: int = 50, 
                     time_window_minutes: int = 60) -> List[AircraftState]:
        """Load a complete traffic scenario from SCAT dataset.
        
        Args:
            max_flights: Maximum number of flights to load
            time_window_minutes: Time window duration in minutes
            
        Returns:
            List of aircraft states for all flights in scenario
        """
        all_states = []
        flight_count = 0
        
        # Find time bounds across all flights
        earliest_time = None
        latest_time = None
        
        logger.info(f"Loading scenario with up to {max_flights} flights...")
        
        for flight_file in self.flight_files[:max_flights]:
            flight_record = self.load_flight_record(flight_file)
            if not flight_record:
                continue
            
            # Track time bounds
            if earliest_time is None or flight_record.start_time < earliest_time:
                earliest_time = flight_record.start_time
            if latest_time is None or flight_record.end_time > latest_time:
                latest_time = flight_record.end_time
            
            # Extract states for this flight
            states = self.extract_aircraft_states(flight_record)
            all_states.extend(states)
            
            flight_count += 1
            if flight_count >= max_flights:
                break
        
        # Apply time window filter if needed
        if time_window_minutes > 0 and earliest_time:
            from datetime import timedelta
            window_end = earliest_time + timedelta(minutes=time_window_minutes)
            
            filtered_states = []
            for state in all_states:
                if earliest_time <= state.timestamp <= window_end:
                    filtered_states.append(state)
            
            all_states = filtered_states
        
        logger.info(f"Loaded scenario with {flight_count} flights, {len(all_states)} total states")
        logger.info(f"Time range: {earliest_time} to {latest_time}")
        
        return sorted(all_states, key=lambda s: s.timestamp)


def load_scat_scenario(dataset_path: str, max_flights: int = 50, 
                      time_window_minutes: int = 60) -> List[AircraftState]:
    """Convenience function to load SCAT scenario.
    
    Args:
        dataset_path: Path to extracted SCAT dataset directory
        max_flights: Maximum number of flights to load
        time_window_minutes: Time window duration in minutes
        
    Returns:
        List of aircraft states for BlueSky simulation
    """
    adapter = SCATAdapter(dataset_path)
    return adapter.load_scenario(max_flights, time_window_minutes)
