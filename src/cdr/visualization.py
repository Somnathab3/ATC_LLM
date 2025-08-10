"""
Visualization module for ATC LLM conflict scenarios.

This module provides visualization capabilities including:
- BlueSky plugin for GUI overlay (preferred)
- Pygame-based viewer as fallback
- Real-time display of aircraft, waypoints, conflicts, and Direct-To legs
- Conflict cylinder visualization (5 NM, 1000 ft separation standards)
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

# Import pygame with fallback
try:
    import pygame
    PYGAME_AVAILABLE = True
    logger.info("Pygame available for visualization")
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not available - visualization disabled")


@dataclass
class VisualizationConfig:
    """Configuration for visualization display."""
    
    # Display settings
    width: int = 1200
    height: int = 800
    fps: int = 30
    
    # Map projection
    center_lat: float = 0.0
    center_lon: float = 0.0
    zoom_nm: float = 200.0  # Nautical miles visible on screen
    
    # Colors (RGB)
    background_color: Tuple[int, int, int] = (20, 30, 40)
    ownship_color: Tuple[int, int, int] = (0, 255, 0)
    intruder_color: Tuple[int, int, int] = (255, 255, 0)
    conflict_color: Tuple[int, int, int] = (255, 0, 0)
    waypoint_color: Tuple[int, int, int] = (0, 150, 255)
    route_color: Tuple[int, int, int] = (100, 100, 100)
    direct_to_color: Tuple[int, int, int] = (255, 150, 0)
    
    # Visualization options
    show_conflict_cylinders: bool = True
    show_waypoints: bool = True
    show_routes: bool = True
    show_direct_to: bool = True
    show_labels: bool = True
    show_separation_circles: bool = True


class PygameViewer:
    """Pygame-based aircraft conflict visualization."""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize pygame viewer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.screen = None
        self.clock = None
        self.font = None
        self.running = False
        
        # Data storage
        self.aircraft_states: Dict[str, Dict[str, Any]] = {}
        self.conflicts: List[Dict[str, Any]] = []
        self.waypoints: List[Dict[str, Any]] = []
        self.direct_to_commands: List[Dict[str, Any]] = []
        
        if PYGAME_AVAILABLE:
            self._initialize_pygame()
        else:
            logger.error("Cannot initialize pygame viewer - pygame not available")
    
    def _initialize_pygame(self):
        """Initialize pygame components."""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("ATC LLM Conflict Visualization")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.running = True
            logger.info("Pygame viewer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            self.running = False
    
    def update_aircraft_states(self, states: List[Dict[str, Any]]):
        """Update aircraft positions and states."""
        self.aircraft_states.clear()
        for state in states:
            callsign = state.get('callsign', state.get('aircraft_id', 'UNKNOWN'))
            self.aircraft_states[callsign] = state
        
        # Auto-center map on first aircraft if not set
        if states and self.config.center_lat == 0.0 and self.config.center_lon == 0.0:
            first_aircraft = states[0]
            self.config.center_lat = first_aircraft.get('lat', first_aircraft.get('latitude', 0.0))
            self.config.center_lon = first_aircraft.get('lon', first_aircraft.get('longitude', 0.0))
    
    def update_conflicts(self, conflicts: List[Dict[str, Any]]):
        """Update conflict information."""
        self.conflicts = conflicts.copy()
    
    def update_waypoints(self, waypoints: List[Dict[str, Any]]):
        """Update waypoint locations."""
        self.waypoints = waypoints.copy()
    
    def add_direct_to_command(self, aircraft_id: str, waypoint_name: str, 
                             waypoint_lat: float, waypoint_lon: float):
        """Add Direct-To command for visualization."""
        aircraft_state = self.aircraft_states.get(aircraft_id)
        if aircraft_state:
            self.direct_to_commands.append({
                'aircraft_id': aircraft_id,
                'waypoint_name': waypoint_name,
                'start_lat': aircraft_state.get('lat', aircraft_state.get('latitude', 0.0)),
                'start_lon': aircraft_state.get('lon', aircraft_state.get('longitude', 0.0)),
                'end_lat': waypoint_lat,
                'end_lon': waypoint_lon,
                'timestamp': datetime.now().isoformat()
            })
    
    def _lat_lon_to_screen(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert latitude/longitude to screen coordinates."""
        # Simple equirectangular projection centered on map center
        lat_diff = lat - self.config.center_lat
        lon_diff = lon - self.config.center_lon
        
        # Convert to nautical miles from center
        lat_nm = lat_diff * 60.0  # 1 degree = 60 NM
        lon_nm = lon_diff * 60.0 * math.cos(math.radians(self.config.center_lat))
        
        # Convert to screen pixels
        scale = min(self.config.width, self.config.height) / (2 * self.config.zoom_nm)
        screen_x = int(self.config.width / 2 + lon_nm * scale)
        screen_y = int(self.config.height / 2 - lat_nm * scale)  # Y is inverted
        
        return screen_x, screen_y
    
    def _nm_to_pixels(self, distance_nm: float) -> float:
        """Convert nautical miles to screen pixels."""
        scale = min(self.config.width, self.config.height) / (2 * self.config.zoom_nm)
        return distance_nm * scale
    
    def _draw_aircraft(self, aircraft_id: str, state: Dict[str, Any], is_ownship: bool = False):
        """Draw aircraft symbol on screen."""
        lat = state.get('lat', state.get('latitude', 0.0))
        lon = state.get('lon', state.get('longitude', 0.0))
        heading = state.get('hdg', state.get('heading', 0.0))
        
        screen_x, screen_y = self._lat_lon_to_screen(lat, lon)
        
        # Skip if off-screen
        if screen_x < 0 or screen_x >= self.config.width or screen_y < 0 or screen_y >= self.config.height:
            return
        
        # Choose color
        color = self.config.ownship_color if is_ownship else self.config.intruder_color
        
        # Check if in conflict
        in_conflict = any(aircraft_id in conf.get('aircraft_ids', []) for conf in self.conflicts)
        if in_conflict:
            color = self.config.conflict_color
        
        # Draw aircraft symbol (triangle pointing in heading direction)
        size = 8
        heading_rad = math.radians(heading - 90)  # Adjust for screen coordinates
        
        # Calculate triangle points
        nose_x = screen_x + size * math.cos(heading_rad)
        nose_y = screen_y + size * math.sin(heading_rad)
        
        left_wing_x = screen_x + size * 0.6 * math.cos(heading_rad + 2.5)
        left_wing_y = screen_y + size * 0.6 * math.sin(heading_rad + 2.5)
        
        right_wing_x = screen_x + size * 0.6 * math.cos(heading_rad - 2.5)
        right_wing_y = screen_y + size * 0.6 * math.sin(heading_rad - 2.5)
        
        pygame.draw.polygon(self.screen, color, [
            (nose_x, nose_y), (left_wing_x, left_wing_y), (right_wing_x, right_wing_y)
        ])
        
        # Draw separation circle if enabled
        if self.config.show_separation_circles:
            circle_radius = self._nm_to_pixels(5.0)  # 5 NM separation
            pygame.draw.circle(self.screen, (50, 50, 50), (screen_x, screen_y), int(circle_radius), 1)
        
        # Draw label if enabled
        if self.config.show_labels:
            label_text = f"{aircraft_id}"
            altitude = state.get('alt_ft', state.get('altitude_ft', 0))
            if altitude > 0:
                label_text += f" FL{int(altitude/100):03d}"
            
            label_surface = self.font.render(label_text, True, (255, 255, 255))
            self.screen.blit(label_surface, (screen_x + 10, screen_y - 10))
    
    def _draw_waypoints(self):
        """Draw waypoints on screen."""
        if not self.config.show_waypoints:
            return
        
        for waypoint in self.waypoints:
            lat = waypoint.get('lat', waypoint.get('latitude', 0.0))
            lon = waypoint.get('lon', waypoint.get('longitude', 0.0))
            name = waypoint.get('name', 'WPT')
            
            screen_x, screen_y = self._lat_lon_to_screen(lat, lon)
            
            # Skip if off-screen
            if screen_x < 0 or screen_x >= self.config.width or screen_y < 0 or screen_y >= self.config.height:
                continue
            
            # Draw waypoint symbol (diamond)
            size = 5
            points = [
                (screen_x, screen_y - size),
                (screen_x + size, screen_y),
                (screen_x, screen_y + size),
                (screen_x - size, screen_y)
            ]
            pygame.draw.polygon(self.screen, self.config.waypoint_color, points)
            
            # Draw label
            if self.config.show_labels:
                label_surface = self.font.render(name, True, self.config.waypoint_color)
                self.screen.blit(label_surface, (screen_x + 8, screen_y + 8))
    
    def _draw_direct_to_commands(self):
        """Draw Direct-To command lines."""
        if not self.config.show_direct_to:
            return
        
        for command in self.direct_to_commands:
            start_x, start_y = self._lat_lon_to_screen(command['start_lat'], command['start_lon'])
            end_x, end_y = self._lat_lon_to_screen(command['end_lat'], command['end_lon'])
            
            # Draw dashed line
            pygame.draw.line(self.screen, self.config.direct_to_color, 
                           (start_x, start_y), (end_x, end_y), 2)
            
            # Draw arrow at end
            arrow_size = 8
            angle = math.atan2(end_y - start_y, end_x - start_x)
            arrow_points = [
                (end_x, end_y),
                (end_x - arrow_size * math.cos(angle - 0.5), end_y - arrow_size * math.sin(angle - 0.5)),
                (end_x - arrow_size * math.cos(angle + 0.5), end_y - arrow_size * math.sin(angle + 0.5))
            ]
            pygame.draw.polygon(self.screen, self.config.direct_to_color, arrow_points)
    
    def _draw_conflict_cylinders(self):
        """Draw conflict prediction cylinders."""
        if not self.config.show_conflict_cylinders:
            return
        
        for conflict in self.conflicts:
            # Draw conflict areas as circles
            aircraft_ids = conflict.get('aircraft_ids', [])
            if len(aircraft_ids) >= 2:
                # Get positions of conflicting aircraft
                for aircraft_id in aircraft_ids[:2]:  # Take first two
                    if aircraft_id in self.aircraft_states:
                        state = self.aircraft_states[aircraft_id]
                        lat = state.get('lat', state.get('latitude', 0.0))
                        lon = state.get('lon', state.get('longitude', 0.0))
                        
                        screen_x, screen_y = self._lat_lon_to_screen(lat, lon)
                        
                        # Draw 5 NM conflict circle
                        radius = self._nm_to_pixels(5.0)
                        pygame.draw.circle(self.screen, self.config.conflict_color, 
                                         (screen_x, screen_y), int(radius), 2)
    
    def _draw_info_panel(self):
        """Draw information panel."""
        panel_height = 120
        panel_rect = pygame.Rect(10, 10, 300, panel_height)
        pygame.draw.rect(self.screen, (0, 0, 0), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), panel_rect, 2)
        
        # Display information
        info_lines = [
            f"Aircraft: {len(self.aircraft_states)}",
            f"Conflicts: {len(self.conflicts)}",
            f"Waypoints: {len(self.waypoints)}",
            f"Direct-To: {len(self.direct_to_commands)}",
            f"Zoom: {self.config.zoom_nm:.0f} NM"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (20, 20 + i * 20))
    
    def render_frame(self):
        """Render one frame of the visualization."""
        if not self.running:
            return False
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.config.zoom_nm *= 0.8  # Zoom in
                elif event.key == pygame.K_MINUS:
                    self.config.zoom_nm *= 1.25  # Zoom out
        
        # Clear screen
        self.screen.fill(self.config.background_color)
        
        # Draw components
        self._draw_waypoints()
        self._draw_conflict_cylinders()
        self._draw_direct_to_commands()
        
        # Draw aircraft (ownship first, then intruders)
        ownship_drawn = False
        for aircraft_id, state in self.aircraft_states.items():
            is_ownship = aircraft_id == "OWNSHIP" or not ownship_drawn
            self._draw_aircraft(aircraft_id, state, is_ownship)
            if is_ownship:
                ownship_drawn = True
        
        # Draw info panel
        self._draw_info_panel()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        
        return True
    
    def run_visualization_loop(self):
        """Run the main visualization loop."""
        logger.info("Starting pygame visualization loop")
        while self.running:
            if not self.render_frame():
                break
        
        pygame.quit()
        logger.info("Pygame visualization loop ended")
    
    def export_screenshot(self, filename: str):
        """Export current view as screenshot."""
        if self.screen and PYGAME_AVAILABLE:
            pygame.image.save(self.screen, filename)
            logger.info(f"Screenshot saved to {filename}")


class VisualizationManager:
    """Manager for different visualization modes."""
    
    def __init__(self, config: VisualizationConfig, enable_pygame: bool = True):
        """
        Initialize visualization manager.
        
        Args:
            config: Visualization configuration
            enable_pygame: Whether to enable pygame viewer
        """
        self.config = config
        self.pygame_viewer = None
        
        if enable_pygame and PYGAME_AVAILABLE:
            self.pygame_viewer = PygameViewer(config)
        
        # TODO: Add BlueSky plugin initialization here
        self.bluesky_plugin = None
        
        logger.info("Visualization manager initialized")
    
    def update_scenario_data(self, aircraft_states: List[Dict[str, Any]], 
                           conflicts: List[Dict[str, Any]], 
                           waypoints: List[Dict[str, Any]]):
        """Update scenario data for all visualization modes."""
        if self.pygame_viewer:
            self.pygame_viewer.update_aircraft_states(aircraft_states)
            self.pygame_viewer.update_conflicts(conflicts)
            self.pygame_viewer.update_waypoints(waypoints)
    
    def add_direct_to_command(self, aircraft_id: str, waypoint_name: str, 
                             waypoint_lat: float, waypoint_lon: float):
        """Add Direct-To command visualization."""
        if self.pygame_viewer:
            self.pygame_viewer.add_direct_to_command(
                aircraft_id, waypoint_name, waypoint_lat, waypoint_lon
            )
    
    def render_frame(self) -> bool:
        """Render one frame across all visualization modes."""
        if self.pygame_viewer:
            return self.pygame_viewer.render_frame()
        return True
    
    def export_visualization(self, output_dir: Path, scenario_id: str):
        """Export visualization artifacts."""
        if self.pygame_viewer:
            screenshot_file = output_dir / f"visualization_{scenario_id}.png"
            self.pygame_viewer.export_screenshot(str(screenshot_file))


def create_visualization_from_jsonl(jsonl_file: Path, output_file: Path):
    """Create visualization from JSONL scenario data."""
    if not PYGAME_AVAILABLE:
        logger.warning("Cannot create visualization - pygame not available")
        return
    
    config = VisualizationConfig()
    viewer = PygameViewer(config)
    
    # Load data from JSONL
    aircraft_states = []
    conflicts = []
    waypoints = []
    
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get('type') == 'aircraft_state':
                    aircraft_states.append(data)
                elif data.get('type') == 'conflict':
                    conflicts.append(data)
                elif data.get('type') == 'waypoint':
                    waypoints.append(data)
        
        # Update viewer
        viewer.update_aircraft_states(aircraft_states)
        viewer.update_conflicts(conflicts)
        viewer.update_waypoints(waypoints)
        
        # Render and save
        viewer.render_frame()
        viewer.export_screenshot(str(output_file))
        
        logger.info(f"Created visualization from {jsonl_file} -> {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to create visualization from JSONL: {e}")