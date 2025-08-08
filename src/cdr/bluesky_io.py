"""BlueSky simulator interface for aircraft state and command execution.

This module provides a clean interface to BlueSky that:
- Connects to BlueSky TCP socket interface
- Fetches real-time aircraft states (position, velocity, flight plan)
- Executes ATC commands (HDG, ALT, SPD, DCT)
- Handles connection errors and retries gracefully
"""

import socket
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .schemas import AircraftState, ResolutionCommand, ConfigurationSettings

logger = logging.getLogger(__name__)


class BlueSkyClient:
    """Client for BlueSky air traffic simulator."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize BlueSky client.
        
        Args:
            config: System configuration including BlueSky connection details
        """
        self.config = config
        self.host = config.bluesky_host
        self.port = config.bluesky_port
        self.timeout = config.bluesky_timeout_sec
        
        self.socket: Optional[socket.socket] = None
        self.connected = False
        
        logger.info(f"BlueSky client configured for {self.host}:{self.port}")
    
    def connect(self) -> bool:
        """Establish connection to BlueSky simulator.
        
        Returns:
            True if connection successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            logger.info(f"Connected to BlueSky at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to BlueSky: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to BlueSky simulator."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        logger.info("Disconnected from BlueSky")
    
    def send_command(self, command: str) -> bool:
        """Send command to BlueSky simulator.
        
        Args:
            command: BlueSky command string (e.g., "HDG KLM123 090")
            
        Returns:
            True if command sent successfully
        """
        if not self.connected and not self.connect():
            return False
        
        try:
            # BlueSky expects commands terminated with newline
            message = command.strip() + '\n'
            self.socket.send(message.encode('utf-8'))
            
            logger.debug(f"Sent command: {command}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command '{command}': {e}")
            self.connected = False
            return False
    
    def get_aircraft_states(self) -> List[AircraftState]:
        """Fetch current aircraft states from BlueSky.
        
        Returns:
            List of current aircraft states
        """
        # TODO: Implement actual BlueSky state fetching in Sprint 1
        # For now, return empty list
        logger.debug("Fetching aircraft states from BlueSky")
        return []
    
    def execute_command(self, resolution: ResolutionCommand) -> bool:
        """Execute resolution command via BlueSky.
        
        Args:
            resolution: Resolution command to execute
            
        Returns:
            True if command executed successfully
        """
        command_str = self._resolution_to_bluesky_command(resolution)
        if not command_str:
            logger.error(f"Failed to convert resolution {resolution.resolution_id} to BlueSky command")
            return False
        
        success = self.send_command(command_str)
        if success:
            logger.info(f"Executed BlueSky command: {command_str}")
        
        return success
    
    def create_aircraft(self, aircraft_id: str, aircraft_type: str, lat: float, lon: float, 
                       hdg: float, alt: float, spd: float) -> bool:
        """Create new aircraft in BlueSky simulation.
        
        Args:
            aircraft_id: Unique aircraft identifier
            aircraft_type: Aircraft type code (e.g., "A320")
            lat: Initial latitude in degrees
            lon: Initial longitude in degrees  
            hdg: Initial heading in degrees
            alt: Initial altitude in feet
            spd: Initial speed in knots
            
        Returns:
            True if aircraft created successfully
        """
        command = f"CRE {aircraft_id} {aircraft_type} {lat} {lon} {hdg} {alt} {spd}"
        return self.send_command(command)
    
    def delete_aircraft(self, aircraft_id: str) -> bool:
        """Delete aircraft from BlueSky simulation.
        
        Args:
            aircraft_id: Aircraft identifier to delete
            
        Returns:
            True if aircraft deleted successfully
        """
        command = f"DEL {aircraft_id}"
        return self.send_command(command)
    
    def set_heading(self, aircraft_id: str, heading_deg: float) -> bool:
        """Set aircraft heading.
        
        Args:
            aircraft_id: Target aircraft identifier
            heading_deg: New heading in degrees
            
        Returns:
            True if command successful
        """
        command = f"HDG {aircraft_id} {heading_deg:03.0f}"
        return self.send_command(command)
    
    def set_altitude(self, aircraft_id: str, altitude_ft: float) -> bool:
        """Set aircraft altitude.
        
        Args:
            aircraft_id: Target aircraft identifier
            altitude_ft: New altitude in feet
            
        Returns:
            True if command successful
        """
        command = f"ALT {aircraft_id} {altitude_ft:.0f}"
        return self.send_command(command)
    
    def set_speed(self, aircraft_id: str, speed_kt: float) -> bool:
        """Set aircraft speed.
        
        Args:
            aircraft_id: Target aircraft identifier
            speed_kt: New speed in knots
            
        Returns:
            True if command successful
        """
        command = f"SPD {aircraft_id} {speed_kt:.0f}"
        return self.send_command(command)
    
    def direct_to_waypoint(self, aircraft_id: str, waypoint: str) -> bool:
        """Direct aircraft to waypoint.
        
        Args:
            aircraft_id: Target aircraft identifier
            waypoint: Waypoint identifier
            
        Returns:
            True if command successful
        """
        command = f"DCT {aircraft_id} {waypoint}"
        return self.send_command(command)
    
    def _resolution_to_bluesky_command(self, resolution: ResolutionCommand) -> Optional[str]:
        """Convert resolution command to BlueSky command string.
        
        Args:
            resolution: Resolution command to convert
            
        Returns:
            BlueSky command string or None if conversion failed
        """
        aircraft_id = resolution.target_aircraft
        
        if resolution.resolution_type.value == "heading_change" and resolution.new_heading_deg is not None:
            return f"HDG {aircraft_id} {resolution.new_heading_deg:03.0f}"
        
        elif resolution.resolution_type.value == "altitude_change" and resolution.new_altitude_ft is not None:
            return f"ALT {aircraft_id} {resolution.new_altitude_ft:.0f}"
        
        elif resolution.resolution_type.value == "speed_change" and resolution.new_speed_kt is not None:
            return f"SPD {aircraft_id} {resolution.new_speed_kt:.0f}"
        
        elif resolution.resolution_type.value == "combined":
            # For combined resolutions, prioritize the most critical maneuver
            # TODO: Implement multi-command execution
            if resolution.new_heading_deg is not None:
                return f"HDG {aircraft_id} {resolution.new_heading_deg:03.0f}"
            elif resolution.new_altitude_ft is not None:
                return f"ALT {aircraft_id} {resolution.new_altitude_ft:.0f}"
        
        logger.error(f"Unknown resolution type: {resolution.resolution_type}")
        return None
    
    def close(self) -> None:
        """Close BlueSky client and clean up resources."""
        self.disconnect()
        logger.info("BlueSky client closed")
