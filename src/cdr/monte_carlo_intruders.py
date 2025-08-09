"""Monte Carlo Intruder Generation for ATC Simulation.

This module generates diverse intruder scenarios using Monte Carlo methods,
integrating with KDTree-based spatial search for efficient intrusion detection.
"""

import logging
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2, degrees, asin

try:
    from scipy.spatial import KDTree
    scipy_available = True
except ImportError:
    logging.warning("scipy not available. Using fallback spatial search.")
    scipy_available = False
    # Create a dummy KDTree for type hints
    class KDTree:
        def __init__(self, data): pass
        def query(self, x, k=1): return 0.0, 0

from .schemas import (
    FlightRecord, IntruderScenario, AircraftState, MonteCarloParameters
)

logger = logging.getLogger(__name__)


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in nautical miles."""
    R = 6371.0  # Earth radius in km
    φ1, λ1, φ2, λ2 = map(radians, (lat1, lon1, lat2, lon2))
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    km = R * 2 * atan2(sqrt(a), sqrt(1 - a))
    return km / 1.852


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2 in degrees."""
    φ1, λ1, φ2, λ2 = map(radians, (lat1, lon1, lat2, lon2))
    y = sin(λ2-λ1)*cos(φ2)
    x = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(λ2-λ1)
    bearing = atan2(y, x)
    return (degrees(bearing) + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> Tuple[float, float]:
    """Calculate destination point given start point, bearing and distance."""
    R = 6371.0 / 1.852  # Earth radius in nautical miles
    
    φ1 = radians(lat)
    λ1 = radians(lon)
    θ = radians(bearing_deg)
    
    φ2 = asin(sin(φ1)*cos(distance_nm/R) + cos(φ1)*sin(distance_nm/R)*cos(θ))
    λ2 = λ1 + atan2(sin(θ)*sin(distance_nm/R)*cos(φ1), cos(distance_nm/R) - sin(φ1)*sin(φ2))
    
    return degrees(φ2), degrees(λ2)


@dataclass
class FlightPathPoint:
    """Point along a flight path with temporal information."""
    latitude: float
    longitude: float
    altitude_ft: float
    timestamp: datetime
    heading_deg: float
    speed_kt: float


class FlightPathAnalyzer:
    """Analyzes flight paths for intrusion detection using KDTree."""
    
    def __init__(self, flight_record: FlightRecord):
        self.flight_record = flight_record
        self.path_points: List[FlightPathPoint] = []
        self.kdtree: Optional[KDTree] = None
        self._build_path_points()
        self._build_spatial_index()
    
    def _build_path_points(self):
        """Build detailed path points from flight record."""
        waypoints = self.flight_record.waypoints
        altitudes = self.flight_record.altitudes_ft
        timestamps = self.flight_record.timestamps
        
        for i in range(len(waypoints)):
            if i < len(waypoints) - 1:
                # Calculate heading to next waypoint
                heading = bearing_deg(
                    waypoints[i][0], waypoints[i][1],
                    waypoints[i+1][0], waypoints[i+1][1]
                )
            else:
                # Use previous heading for last point
                heading = self.path_points[-1].heading_deg if self.path_points else 0.0
            
            point = FlightPathPoint(
                latitude=waypoints[i][0],
                longitude=waypoints[i][1],
                altitude_ft=altitudes[i],
                timestamp=timestamps[i],
                heading_deg=heading,
                speed_kt=self.flight_record.cruise_speed_kt
            )
            self.path_points.append(point)
    
    def _build_spatial_index(self):
        """Build KDTree spatial index for efficient nearest neighbor queries."""
        if not scipy_available or not self.path_points:
            return
        
        # Convert lat/lon to 3D Cartesian coordinates for better distance calculations
        points_3d = []
        for point in self.path_points:
            # Convert to ECEF coordinates
            lat_rad = radians(point.latitude)
            lon_rad = radians(point.longitude)
            alt_km = point.altitude_ft * 0.0003048  # ft to km
            
            # Earth radius + altitude
            r = 6371.0 + alt_km
            
            x = r * cos(lat_rad) * cos(lon_rad)
            y = r * cos(lat_rad) * sin(lon_rad)
            z = r * sin(lat_rad)
            
            points_3d.append([x, y, z])
        
        self.kdtree = KDTree(points_3d)
        logger.debug(f"Built KDTree with {len(points_3d)} points for flight {self.flight_record.flight_id}")
    
    def find_closest_path_point(self, lat: float, lon: float, alt_ft: float) -> Tuple[int, float]:
        """Find closest point on flight path to given position.
        
        Returns:
            Tuple of (index, distance_nm)
        """
        if not self.kdtree:
            # Fallback to linear search
            min_dist = float('inf')
            closest_idx = 0
            for i, point in enumerate(self.path_points):
                dist = haversine_distance_nm(lat, lon, point.latitude, point.longitude)
                # Add vertical component (roughly)
                alt_diff_nm = abs(alt_ft - point.altitude_ft) / 6076.0  # ft to nm
                total_dist = sqrt(dist**2 + alt_diff_nm**2)
                
                if total_dist < min_dist:
                    min_dist = total_dist
                    closest_idx = i
            
            return closest_idx, min_dist
        
        # Use KDTree for efficient search
        lat_rad = radians(lat)
        lon_rad = radians(lon)
        alt_km = alt_ft * 0.0003048
        r = 6371.0 + alt_km
        
        query_point = [
            r * cos(lat_rad) * cos(radians(lon)),
            r * cos(lat_rad) * sin(radians(lon)),
            r * sin(lat_rad)
        ]
        
        distance_3d, index = self.kdtree.query(query_point)
        
        # Convert 3D distance back to approximate nautical miles
        distance_nm = distance_3d * 0.539957  # rough conversion
        
        return index, distance_nm
    
    def detect_intrusions_along_path(self, intruder_states: List[AircraftState], 
                                   threshold_nm: float = 5.0) -> List[Dict[str, Any]]:
        """Detect intrusions along the flight path using spatial indexing."""
        intrusions = []
        
        for intruder in intruder_states:
            # Find closest point on our path
            closest_idx, distance_nm = self.find_closest_path_point(
                intruder.latitude, intruder.longitude, intruder.altitude_ft
            )
            
            if distance_nm <= threshold_nm:
                path_point = self.path_points[closest_idx]
                
                # Calculate vertical separation
                vertical_sep_ft = abs(intruder.altitude_ft - path_point.altitude_ft)
                
                intrusion = {
                    'intruder_id': intruder.aircraft_id,
                    'intrusion_time': path_point.timestamp,
                    'ownship_position': (path_point.latitude, path_point.longitude),
                    'intruder_position': (intruder.latitude, intruder.longitude),
                    'horizontal_separation_nm': distance_nm,
                    'vertical_separation_ft': vertical_sep_ft,
                    'path_point_index': closest_idx,
                    'is_conflict': distance_nm <= 5.0 and vertical_sep_ft <= 1000.0
                }
                intrusions.append(intrusion)
                
                logger.debug(f"Detected intrusion: {intruder.aircraft_id} at {distance_nm:.2f} NM")
        
        return intrusions


class MonteCarloIntruderGenerator:
    """Generates diverse intruder scenarios using Monte Carlo methods."""
    
    def __init__(self, parameters: MonteCarloParameters):
        self.params = parameters
        self.rng = random.Random()
        
    def generate_scenarios_for_flight(self, flight_record: FlightRecord) -> List[IntruderScenario]:
        """Generate multiple scenarios for a single flight."""
        scenarios = []
        
        # Analyze flight path
        path_analyzer = FlightPathAnalyzer(flight_record)
        
        for scenario_idx in range(self.params.scenarios_per_flight):
            # Set seed for reproducibility
            seed = hash(f"{flight_record.flight_id}_{scenario_idx}") % (2**32)
            self.rng.seed(seed)
            
            scenario = self._generate_single_scenario(
                flight_record, path_analyzer, scenario_idx, seed
            )
            scenarios.append(scenario)
            
            logger.debug(f"Generated scenario {scenario_idx} for flight {flight_record.flight_id}")
        
        return scenarios
    
    def _generate_single_scenario(self, flight_record: FlightRecord, 
                                path_analyzer: FlightPathAnalyzer,
                                scenario_idx: int, seed: int) -> IntruderScenario:
        """Generate a single intruder scenario."""
        # Determine number of intruders
        min_intruders, max_intruders = self.params.intruder_count_range
        num_intruders = self.rng.randint(min_intruders, max_intruders)
        
        # Generate intruder aircraft
        intruder_states = []
        airspace_bounds = self._calculate_airspace_bounds(flight_record)
        
        for i in range(num_intruders):
            # Decide if this intruder should create a conflict
            should_conflict = self.rng.random() < self.params.conflict_probability
            
            if should_conflict:
                intruder = self._generate_conflicting_intruder(
                    flight_record, path_analyzer, i
                )
            else:
                intruder = self._generate_non_conflicting_intruder(
                    flight_record, airspace_bounds, i
                )
            
            intruder_states.append(intruder)
        
        # Detect actual intrusions
        intrusions = path_analyzer.detect_intrusions_along_path(intruder_states)
        has_conflicts = any(intrusion['is_conflict'] for intrusion in intrusions)
        expected_conflicts = [
            f"{flight_record.flight_id}_{intrusion['intruder_id']}"
            for intrusion in intrusions if intrusion['is_conflict']
        ]
        
        # Calculate geometric complexity
        complexity = self._calculate_geometric_complexity(intruder_states, flight_record)
        
        scenario = IntruderScenario(
            scenario_id=f"{flight_record.flight_id}_scenario_{scenario_idx}",
            intruder_states=intruder_states,
            intruder_count=num_intruders,
            conflict_probability=self.params.conflict_probability,
            geometric_complexity=complexity,
            generation_seed=seed,
            airspace_bounds=airspace_bounds,
            has_conflicts=has_conflicts,
            expected_conflicts=expected_conflicts
        )
        
        return scenario
    
    def _generate_conflicting_intruder(self, flight_record: FlightRecord,
                                     path_analyzer: FlightPathAnalyzer,
                                     intruder_idx: int) -> AircraftState:
        """Generate an intruder that will conflict with the flight path."""
        # Choose a random point along the flight path
        path_point = self.rng.choice(path_analyzer.path_points)
        
        # Generate position near the path point
        conflict_distance = self.rng.uniform(1.0, self.params.conflict_zone_radius_nm)
        conflict_bearing = self.rng.uniform(0, 360)
        
        intruder_lat, intruder_lon = destination_point(
            path_point.latitude, path_point.longitude,
            conflict_bearing, conflict_distance
        )
        
        # Generate altitude with some variance
        altitude_variance = self.rng.uniform(-self.params.altitude_spread_ft/2, 
                                           self.params.altitude_spread_ft/2)
        intruder_alt = path_point.altitude_ft + altitude_variance
        intruder_alt = max(1000, min(60000, intruder_alt))  # Keep within bounds
        
        # Generate realistic speed and heading
        speed_variance = self.rng.uniform(-self.params.speed_variance_kt,
                                        self.params.speed_variance_kt)
        intruder_speed = flight_record.cruise_speed_kt + speed_variance
        intruder_speed = max(150, min(600, intruder_speed))  # Realistic bounds
        
        heading_variance = self.rng.uniform(-self.params.heading_variance_deg,
                                          self.params.heading_variance_deg)
        intruder_heading = (path_point.heading_deg + heading_variance) % 360
        
        # Generate timestamp around the path point time
        time_variance = timedelta(
            minutes=self.rng.uniform(-self.params.conflict_timing_variance_min,
                                   self.params.conflict_timing_variance_min)
        )
        intruder_time = path_point.timestamp + time_variance
        
        return AircraftState(
            aircraft_id=f"INTRUDER_{flight_record.flight_id}_{intruder_idx:02d}",
            timestamp=intruder_time,
            latitude=intruder_lat,
            longitude=intruder_lon,
            altitude_ft=intruder_alt,
            ground_speed_kt=intruder_speed,
            heading_deg=intruder_heading,
            vertical_speed_fpm=self.rng.uniform(-1000, 1000),
            callsign=f"INT{intruder_idx:02d}",
            aircraft_type=self.rng.choice(["B737", "A320", "B777", "A350"])
        )
    
    def _generate_non_conflicting_intruder(self, flight_record: FlightRecord,
                                         airspace_bounds: Dict[str, float],
                                         intruder_idx: int) -> AircraftState:
        """Generate an intruder that will not conflict with the flight path."""
        # Generate position in the broader airspace
        lat = self.rng.uniform(airspace_bounds['min_lat'], airspace_bounds['max_lat'])
        lon = self.rng.uniform(airspace_bounds['min_lon'], airspace_bounds['max_lon'])
        
        # Generate altitude
        alt = self.rng.uniform(
            min(flight_record.altitudes_ft) - self.params.altitude_spread_ft,
            max(flight_record.altitudes_ft) + self.params.altitude_spread_ft
        )
        alt = max(1000, min(60000, alt))
        
        # Generate realistic speed and heading
        speed = self.rng.uniform(200, 500)
        heading = self.rng.uniform(0, 360)
        
        # Use first timestamp as reference
        timestamp = flight_record.timestamps[0] + timedelta(
            minutes=self.rng.uniform(-30, 30)
        )
        
        return AircraftState(
            aircraft_id=f"TRAFFIC_{flight_record.flight_id}_{intruder_idx:02d}",
            timestamp=timestamp,
            latitude=lat,
            longitude=lon,
            altitude_ft=alt,
            ground_speed_kt=speed,
            heading_deg=heading,
            vertical_speed_fpm=self.rng.uniform(-1000, 1000),
            callsign=f"TFC{intruder_idx:02d}",
            aircraft_type=self.rng.choice(["B737", "A320", "B777", "A350"])
        )
    
    def _calculate_airspace_bounds(self, flight_record: FlightRecord) -> Dict[str, float]:
        """Calculate airspace bounds around the flight path."""
        lats = [wp[0] for wp in flight_record.waypoints]
        lons = [wp[1] for wp in flight_record.waypoints]
        
        # Add buffer around the flight path
        buffer_deg = self.params.non_conflict_zone_radius_nm / 60.0  # rough conversion
        
        return {
            'min_lat': min(lats) - buffer_deg,
            'max_lat': max(lats) + buffer_deg,
            'min_lon': min(lons) - buffer_deg,
            'max_lon': max(lons) + buffer_deg
        }
    
    def _calculate_geometric_complexity(self, intruder_states: List[AircraftState],
                                      flight_record: FlightRecord) -> float:
        """Calculate geometric complexity score for the scenario."""
        if not intruder_states:
            return 0.0
        
        # Factors contributing to complexity:
        # 1. Number of intruders
        # 2. Spatial distribution
        # 3. Speed/altitude variations
        
        num_factor = min(len(intruder_states) / 10.0, 1.0)  # Normalize to 0-1
        
        # Calculate spatial spread
        positions = [(state.latitude, state.longitude) for state in intruder_states]
        if len(positions) > 1:
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = haversine_distance_nm(
                        positions[i][0], positions[i][1],
                        positions[j][0], positions[j][1]
                    )
                    distances.append(dist)
            avg_separation = sum(distances) / len(distances)
            spatial_factor = min(avg_separation / 100.0, 1.0)  # Normalize
        else:
            spatial_factor = 0.0
        
        # Calculate speed/altitude variations
        speeds = [state.ground_speed_kt for state in intruder_states]
        altitudes = [state.altitude_ft for state in intruder_states]
        
        speed_var = np.std(speeds) / np.mean(speeds) if speeds else 0.0
        alt_var = np.std(altitudes) / np.mean(altitudes) if altitudes else 0.0
        
        variation_factor = min((speed_var + alt_var) / 2.0, 1.0)
        
        # Combine factors
        complexity = (num_factor + spatial_factor + variation_factor) / 3.0
        return min(complexity, 1.0)


class BatchIntruderGenerator:
    """Coordinates Monte Carlo generation across multiple flights."""
    
    def __init__(self, parameters: MonteCarloParameters):
        self.params = parameters
        self.generator = MonteCarloIntruderGenerator(parameters)
    
    def generate_scenarios_for_flights(self, flight_records: List[FlightRecord]) -> Dict[str, List[IntruderScenario]]:
        """Generate scenarios for multiple flights.
        
        Returns:
            Dictionary mapping flight_id to list of scenarios
        """
        all_scenarios = {}
        
        for flight_record in flight_records:
            logger.info(f"Generating {self.params.scenarios_per_flight} scenarios for flight {flight_record.flight_id}")
            
            scenarios = self.generator.generate_scenarios_for_flight(flight_record)
            all_scenarios[flight_record.flight_id] = scenarios
            
            # Log statistics
            conflict_scenarios = sum(1 for s in scenarios if s.has_conflicts)
            logger.info(f"Flight {flight_record.flight_id}: {conflict_scenarios}/{len(scenarios)} scenarios have conflicts")
        
        total_scenarios = sum(len(scenarios) for scenarios in all_scenarios.values())
        logger.info(f"Generated {total_scenarios} total scenarios for {len(flight_records)} flights")
        
        return all_scenarios
