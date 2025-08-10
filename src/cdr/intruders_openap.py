"""
Monte-Carlo Intruder Generator with OpenAP Performance Constraints

This module generates realistic intruder aircraft with performance limits derived
from OpenAP aircraft database. It provides:
- Aircraft type sampling from OpenAP fleet data
- Type-specific performance envelopes (speed, climb/descend, turn rates)
- Dynamic intruder injection during simulation
- Collision geometry patterns (head-on, crossing, overtaking)

Based on OpenAP: Open Aircraft Performance Model and Toolkit
"""

import json
import logging
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import OpenAP with fallback to mock data
try:
    import openap
    from openap import prop, FuelFlow, Emission, WRAP
    OPENAP_AVAILABLE = True
    logger.info("OpenAP library loaded successfully")
except ImportError:
    openap = None
    OPENAP_AVAILABLE = False
    logger.warning("OpenAP not available, using fallback performance data")


class GeometryType(Enum):
    """Conflict geometry patterns."""
    HEAD_ON = "head_on"
    CROSSING = "crossing"  
    OVERTAKING = "overtaking"
    LEVEL = "level"
    SPLIT_LEVEL = "split_level"


class FlightPhase(Enum):
    """Flight phase for performance envelope selection."""
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"


@dataclass
class PerformanceEnvelope:
    """Aircraft performance limits for a specific type and flight level."""
    aircraft_type: str
    min_speed_kt: float
    max_speed_kt: float
    max_climb_rate_fpm: float
    max_descent_rate_fpm: float
    max_bank_angle_deg: float
    max_turn_rate_deg_sec: float
    service_ceiling_ft: float
    typical_cruise_fl: int


@dataclass
class IntruderDefinition:
    """Complete intruder aircraft definition."""
    callsign: str
    aircraft_type: str
    initial_lat: float
    initial_lon: float
    initial_alt_ft: float
    initial_heading_deg: float
    initial_speed_kt: float
    spawn_time_sec: float
    geometry_type: GeometryType
    flight_phase: FlightPhase
    performance: PerformanceEnvelope
    waypoints: List[Tuple[float, float, float]]  # (lat, lon, alt_ft)


class OpenAPIntruderGenerator:
    """Generates Monte-Carlo intruders with OpenAP performance constraints."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize intruder generator.
        
        Args:
            seed: Random seed for reproducible scenarios
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.performance_cache: Dict[str, PerformanceEnvelope] = {}
        self.aircraft_types = self._load_aircraft_types()
        
        logger.info(f"Initialized OpenAP intruder generator with {len(self.aircraft_types)} aircraft types")
    
    def _load_aircraft_types(self) -> List[str]:
        """Load available aircraft types from OpenAP or fallback data."""
        if OPENAP_AVAILABLE:
            try:
                # Get all available aircraft from OpenAP
                aircraft_db = openap.prop.available_aircraft()
                # Filter for common commercial aircraft
                common_types = [ac for ac in aircraft_db if ac in [
                    'a320', 'a321', 'a330', 'a350', 'a380',
                    'b737', 'b747', 'b757', 'b767', 'b777', 'b787',
                    'crj2', 'e145', 'e170', 'e190'
                ]]
                return common_types if common_types else aircraft_db[:15]  # Top 15 if filter fails
            except Exception as e:
                logger.warning(f"Failed to load OpenAP aircraft types: {e}")
        
        # Fallback aircraft types
        return [
            'a320', 'a321', 'a330', 'b737', 'b747', 'b757', 'b767', 'b777',
            'crj2', 'e145', 'e170', 'e190'
        ]
    
    def get_performance_envelope(self, aircraft_type: str, altitude_ft: float) -> PerformanceEnvelope:
        """
        Get performance envelope for aircraft type at given altitude.
        
        Args:
            aircraft_type: ICAO aircraft type code
            altitude_ft: Operating altitude in feet
            
        Returns:
            Performance envelope with speed, climb, turn limits
        """
        cache_key = f"{aircraft_type}_{int(altitude_ft // 1000)}"
        
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        if OPENAP_AVAILABLE:
            envelope = self._get_openap_performance(aircraft_type, altitude_ft)
        else:
            envelope = self._get_fallback_performance(aircraft_type, altitude_ft)
        
        self.performance_cache[cache_key] = envelope
        return envelope
    
    def _get_openap_performance(self, aircraft_type: str, altitude_ft: float) -> PerformanceEnvelope:
        """Get performance data from OpenAP."""
        try:
            # Get aircraft properties
            ac_props = openap.prop.aircraft(aircraft_type)
            
            # Operating speeds (convert from m/s to kt as needed)
            cruise_mach = ac_props.get('cruise', {}).get('mach', 0.78)
            max_mach = ac_props.get('limits', {}).get('mach', 0.82)
            
            # Convert Mach to TAS at altitude (simplified)
            # Speed of sound varies with altitude, roughly 661 kt at sea level, 573 kt at FL350
            altitude_factor = max(0.8, 1.0 - altitude_ft / 100000)  # Rough approximation
            speed_of_sound_kt = 661 * altitude_factor
            
            cruise_speed_kt = cruise_mach * speed_of_sound_kt
            max_speed_kt = max_mach * speed_of_sound_kt
            min_speed_kt = cruise_speed_kt * 0.75  # Approximate minimum speed
            
            # Performance limits
            max_ceiling_ft = ac_props.get('limits', {}).get('ceiling', 41000)
            
            # Engine-based climb performance (simplified)
            try:
                # Use simplified climb rate based on aircraft size
                mtow_kg = ac_props.get('limits', {}).get('MTOW', 70000)
                if mtow_kg > 200000:  # Heavy aircraft
                    max_climb_fpm = 1500
                    max_turn_rate = 1.5
                elif mtow_kg > 100000:  # Medium aircraft  
                    max_climb_fpm = 2000
                    max_turn_rate = 2.0
                else:  # Light aircraft
                    max_climb_fpm = 2500
                    max_turn_rate = 2.5
                
                # Adjust for altitude (lower performance at high altitude)
                altitude_factor = max(0.3, 1.0 - altitude_ft / 45000)
                max_climb_fpm *= altitude_factor
                
            except Exception:
                max_climb_fpm = 2000  # Default
                max_turn_rate = 2.0
            
            return PerformanceEnvelope(
                aircraft_type=aircraft_type,
                min_speed_kt=min_speed_kt,
                max_speed_kt=max_speed_kt,
                max_climb_rate_fpm=max_climb_fpm,
                max_descent_rate_fpm=3000,  # Typical maximum descent rate
                max_bank_angle_deg=30,      # Standard maximum bank angle
                max_turn_rate_deg_sec=max_turn_rate,
                service_ceiling_ft=max_ceiling_ft,
                typical_cruise_fl=int(ac_props.get('cruise', {}).get('height', 35000) / 100)
            )
            
        except Exception as e:
            logger.warning(f"OpenAP performance lookup failed for {aircraft_type}: {e}")
            return self._get_fallback_performance(aircraft_type, altitude_ft)
    
    def _get_fallback_performance(self, aircraft_type: str, altitude_ft: float) -> PerformanceEnvelope:
        """Get fallback performance data when OpenAP is not available."""
        # Performance data based on aircraft type patterns
        if aircraft_type.startswith('a3'):  # A320 family
            base_speed = 450
            climb_rate = 2000
            ceiling = 39000
        elif aircraft_type.startswith('b77') or aircraft_type.startswith('a35'):  # Large wide-body
            base_speed = 500
            climb_rate = 1800
            ceiling = 43000
        elif aircraft_type.startswith('b74') or aircraft_type.startswith('a38'):  # Very large
            base_speed = 480
            climb_rate = 1500
            ceiling = 45000
        elif aircraft_type.startswith('b73'):  # B737 family
            base_speed = 440
            climb_rate = 2200
            ceiling = 41000
        elif 'crj' in aircraft_type or 'e1' in aircraft_type:  # Regional jets
            base_speed = 400
            climb_rate = 2500
            ceiling = 37000
        else:  # Default medium aircraft
            base_speed = 450
            climb_rate = 2000
            ceiling = 39000
        
        # Adjust for altitude
        altitude_factor = max(0.4, 1.0 - altitude_ft / 50000)
        
        return PerformanceEnvelope(
            aircraft_type=aircraft_type,
            min_speed_kt=base_speed * 0.75,
            max_speed_kt=base_speed * 1.15,
            max_climb_rate_fpm=climb_rate * altitude_factor,
            max_descent_rate_fpm=3000,
            max_bank_angle_deg=30,
            max_turn_rate_deg_sec=2.0,
            service_ceiling_ft=ceiling,
            typical_cruise_fl=int(ceiling / 100) - 2
        )
    
    def generate_intruders(self,
                          ownship_state: Dict[str, Any],
                          count: int,
                          vicinity_radius_nm: float = 100,
                          simulation_duration_sec: float = 3600,
                          geometry_weights: Optional[Dict[GeometryType, float]] = None) -> List[IntruderDefinition]:
        """
        Generate Monte-Carlo intruders around ownship.
        
        Args:
            ownship_state: Current ownship position and velocity
            count: Number of intruders to generate
            vicinity_radius_nm: Maximum distance from ownship for spawning
            simulation_duration_sec: Total simulation time for spawn scheduling
            geometry_weights: Probability weights for different geometry types
            
        Returns:
            List of intruder definitions
        """
        if geometry_weights is None:
            geometry_weights = {
                GeometryType.HEAD_ON: 0.3,
                GeometryType.CROSSING: 0.4,
                GeometryType.OVERTAKING: 0.2,
                GeometryType.LEVEL: 0.1
            }
        
        intruders = []
        
        # Extract ownship parameters
        own_lat = ownship_state.get('lat', ownship_state.get('latitude', 0.0))
        own_lon = ownship_state.get('lon', ownship_state.get('longitude', 0.0))
        own_alt = ownship_state.get('alt_ft', ownship_state.get('altitude_ft', 35000.0))
        own_hdg = ownship_state.get('hdg', ownship_state.get('heading', 0.0))
        own_spd = ownship_state.get('spd_kt', ownship_state.get('ground_speed_kt', 450.0))
        
        for i in range(count):
            # Select aircraft type
            aircraft_type = random.choice(self.aircraft_types)
            
            # Select geometry type
            geometry_type = self._weighted_choice(geometry_weights)
            
            # Generate initial position and heading based on geometry
            initial_lat, initial_lon, initial_heading = self._generate_initial_state(
                own_lat, own_lon, own_hdg, vicinity_radius_nm, geometry_type
            )
            
            # Select altitude (within ±5000 ft of ownship for potential conflicts)
            if geometry_type == GeometryType.SPLIT_LEVEL:
                alt_offset = random.choice([-2000, -1000, 1000, 2000])
            else:
                alt_offset = random.randint(-3000, 3000)
            
            initial_alt = max(10000, min(45000, own_alt + alt_offset))
            
            # Get performance envelope
            performance = self.get_performance_envelope(aircraft_type, initial_alt)
            
            # Select speed within performance limits
            speed_range = performance.max_speed_kt - performance.min_speed_kt
            initial_speed = performance.min_speed_kt + random.random() * speed_range
            
            # Generate spawn time (spread throughout simulation)
            spawn_time = random.uniform(0, simulation_duration_sec * 0.8)  # Don't spawn too late
            
            # Determine flight phase
            if initial_alt < 25000:
                phase = FlightPhase.CLIMB
            elif initial_alt > 38000:
                phase = FlightPhase.DESCENT  
            else:
                phase = FlightPhase.CRUISE
            
            # Generate waypoints for basic flight plan
            waypoints = self._generate_waypoints(
                initial_lat, initial_lon, initial_alt, initial_heading, performance
            )
            
            # Create intruder
            intruder = IntruderDefinition(
                callsign=f"INT{i+1:03d}",
                aircraft_type=aircraft_type,
                initial_lat=initial_lat,
                initial_lon=initial_lon,
                initial_alt_ft=initial_alt,
                initial_heading_deg=initial_heading,
                initial_speed_kt=initial_speed,
                spawn_time_sec=spawn_time,
                geometry_type=geometry_type,
                flight_phase=phase,
                performance=performance,
                waypoints=waypoints
            )
            
            intruders.append(intruder)
            
            logger.debug(f"Generated {aircraft_type} intruder {intruder.callsign} "
                        f"at {initial_lat:.3f},{initial_lon:.3f} FL{int(initial_alt/100)}")
        
        # Sort by spawn time
        intruders.sort(key=lambda x: x.spawn_time_sec)
        
        logger.info(f"Generated {len(intruders)} intruders with OpenAP performance constraints")
        return intruders
    
    def _weighted_choice(self, weights: Dict[GeometryType, float]) -> GeometryType:
        """Select geometry type based on weights."""
        choices = list(weights.keys())
        probabilities = list(weights.values())
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return np.random.choice(choices, p=probabilities)
    
    def _generate_initial_state(self,
                               own_lat: float,
                               own_lon: float, 
                               own_hdg: float,
                               radius_nm: float,
                               geometry: GeometryType) -> Tuple[float, float, float]:
        """Generate initial position and heading for intruder based on geometry type."""
        
        # Convert NM to degrees (rough approximation)
        radius_deg = radius_nm / 60.0
        
        if geometry == GeometryType.HEAD_ON:
            # Place intruder ahead on opposite heading
            angle_offset = random.uniform(-30, 30)  # Some variation
            bearing = (own_hdg + 180 + angle_offset) % 360
            distance = random.uniform(radius_nm * 0.6, radius_nm)
            intruder_heading = (own_hdg + 180 + random.uniform(-15, 15)) % 360
            
        elif geometry == GeometryType.CROSSING:
            # Place intruder to the side with crossing angle
            side_angle = random.choice([90, -90]) + random.uniform(-30, 30)
            bearing = (own_hdg + side_angle) % 360
            distance = random.uniform(radius_nm * 0.4, radius_nm * 0.8)
            # Crossing angle typically 30-90 degrees
            cross_angle = random.uniform(30, 90) * random.choice([1, -1])
            intruder_heading = (own_hdg + cross_angle) % 360
            
        elif geometry == GeometryType.OVERTAKING:
            # Place intruder behind with similar heading
            angle_offset = random.uniform(-45, 45)
            bearing = (own_hdg + angle_offset) % 360
            distance = random.uniform(radius_nm * 0.3, radius_nm * 0.7)
            intruder_heading = (own_hdg + random.uniform(-20, 20)) % 360
            
        else:  # LEVEL or default
            # Random placement
            bearing = random.uniform(0, 360)
            distance = random.uniform(radius_nm * 0.4, radius_nm)
            intruder_heading = random.uniform(0, 360)
        
        # Convert to lat/lon
        distance_deg = distance / 60.0
        bearing_rad = np.radians(bearing)
        
        new_lat = own_lat + distance_deg * np.cos(bearing_rad)
        new_lon = own_lon + distance_deg * np.sin(bearing_rad) / np.cos(np.radians(own_lat))
        
        return new_lat, new_lon, intruder_heading
    
    def _generate_waypoints(self,
                           lat: float,
                           lon: float,
                           alt: float,
                           heading: float,
                           performance: PerformanceEnvelope) -> List[Tuple[float, float, float]]:
        """Generate basic waypoints for intruder flight plan."""
        waypoints = [(lat, lon, alt)]
        
        # Generate 2-3 waypoints ahead
        current_lat, current_lon, current_alt = lat, lon, alt
        
        for i in range(random.randint(2, 4)):
            # Distance to next waypoint (50-150 NM)
            distance_nm = random.uniform(50, 150)
            distance_deg = distance_nm / 60.0
            
            # Small heading variation
            heading += random.uniform(-15, 15)
            heading = heading % 360
            
            # Altitude change based on flight phase
            if current_alt < 30000:  # Climbing
                alt_change = random.uniform(0, 5000)
            elif current_alt > 40000:  # Descending
                alt_change = random.uniform(-5000, 0)
            else:  # Cruise
                alt_change = random.uniform(-2000, 2000)
            
            current_alt = max(10000, min(performance.service_ceiling_ft, current_alt + alt_change))
            
            # Calculate new position
            heading_rad = np.radians(heading)
            current_lat += distance_deg * np.cos(heading_rad)
            current_lon += distance_deg * np.sin(heading_rad) / np.cos(np.radians(current_lat))
            
            waypoints.append((current_lat, current_lon, current_alt))
        
        return waypoints
    
    def export_scenarios(self, intruders: List[IntruderDefinition], output_file: Path):
        """Export intruder scenarios to JSON file."""
        scenarios_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'OpenAPIntruderGenerator',
                'count': len(intruders)
            },
            'intruders': []
        }
        
        for intruder in intruders:
            intruder_data = {
                'callsign': intruder.callsign,
                'aircraft_type': intruder.aircraft_type,
                'initial_position': {
                    'lat': intruder.initial_lat,
                    'lon': intruder.initial_lon,
                    'alt_ft': intruder.initial_alt_ft
                },
                'initial_velocity': {
                    'heading_deg': intruder.initial_heading_deg,
                    'speed_kt': intruder.initial_speed_kt
                },
                'spawn_time_sec': intruder.spawn_time_sec,
                'geometry_type': intruder.geometry_type.value,
                'flight_phase': intruder.flight_phase.value,
                'performance_envelope': {
                    'min_speed_kt': intruder.performance.min_speed_kt,
                    'max_speed_kt': intruder.performance.max_speed_kt,
                    'max_climb_rate_fpm': intruder.performance.max_climb_rate_fpm,
                    'max_descent_rate_fpm': intruder.performance.max_descent_rate_fpm,
                    'max_bank_angle_deg': intruder.performance.max_bank_angle_deg,
                    'max_turn_rate_deg_sec': intruder.performance.max_turn_rate_deg_sec,
                    'service_ceiling_ft': intruder.performance.service_ceiling_ft
                },
                'waypoints': [
                    {'lat': wp[0], 'lon': wp[1], 'alt_ft': wp[2]} 
                    for wp in intruder.waypoints
                ]
            }
            scenarios_data['intruders'].append(intruder_data)
        
        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(scenarios_data, f, indent=2)
        
        logger.info(f"Exported {len(intruders)} intruder scenarios to {output_file}")
    
    def get_performance_hints_for_llm(self, aircraft_type: str, altitude_ft: float) -> Dict[str, Any]:
        """Get performance hints formatted for LLM prompts."""
        performance = self.get_performance_envelope(aircraft_type, altitude_ft)
        
        return {
            'aircraft_type': aircraft_type,
            'speed_range_kt': {
                'min': performance.min_speed_kt,
                'max': performance.max_speed_kt
            },
            'climb_limits': {
                'max_rate_fpm': performance.max_climb_rate_fpm,
                'max_descent_fpm': performance.max_descent_rate_fpm
            },
            'maneuvering': {
                'max_bank_deg': performance.max_bank_angle_deg,
                'max_turn_rate_deg_sec': performance.max_turn_rate_deg_sec
            },
            'altitude_limits': {
                'service_ceiling_ft': performance.service_ceiling_ft,
                'current_alt_ft': altitude_ft
            }
        }