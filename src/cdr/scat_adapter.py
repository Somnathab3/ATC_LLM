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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

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
        
        logger.info(f"Found {len(self.flight_files)} SCAT flight records")
    
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
    
    def get_flight_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the SCAT dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        summary = {
            'total_files': len(self.flight_files),
            'aircraft_types': set(),
            'airports': set(),
            'callsigns': set(),
            'time_range': {'earliest': None, 'latest': None}
        }
        
        # Sample first 100 files for statistics
        sample_files = self.flight_files[:100]
        
        for flight_file in sample_files:
            flight_record = self.load_flight_record(flight_file)
            if not flight_record:
                continue
            
            summary['aircraft_types'].add(flight_record.aircraft_type)
            summary['airports'].add(flight_record.adep)
            summary['airports'].add(flight_record.ades)
            summary['callsigns'].add(flight_record.callsign)
            
            # Track time bounds
            if (summary['time_range']['earliest'] is None or 
                flight_record.start_time < summary['time_range']['earliest']):
                summary['time_range']['earliest'] = flight_record.start_time
            
            if (summary['time_range']['latest'] is None or 
                flight_record.end_time > summary['time_range']['latest']):
                summary['time_range']['latest'] = flight_record.end_time
        
        # Convert sets to sorted lists for JSON serialization
        summary['aircraft_types'] = sorted(summary['aircraft_types'])
        summary['airports'] = sorted(summary['airports'])
        summary['callsigns'] = sorted(summary['callsigns'])
        
        return summary


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
