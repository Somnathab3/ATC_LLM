"""BlueSky simulator interface for aircraft state and command execution.

This module provides a clean interface to BlueSky that:
- Embeds BlueSky simulator directly for better performance
- Fetches real-time aircraft states (position, velocity, flight plan)
- Executes ATC commands (HDG, ALT, SPD, DCT)
- Handles connection errors and retries gracefully
"""

from __future__ import annotations
import logging
import time
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from .schemas import AircraftState, ResolutionCommand, ConfigurationSettings

logger = logging.getLogger(__name__)


@dataclass
class BSConfig:
    """BlueSky configuration parameters."""
    headless: bool = True
    # Add any IPC/embedding knobs your BlueSky runner needs


class BlueSkyClient:
    """Client for BlueSky air traffic simulator with embedded integration."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize BlueSky client.
        
        Args:
            config: System configuration including BlueSky connection details
        """
        self.config = config
        self.bs = None  # handle to embedded bluesky or API wrapper
        self.connected = False
        
        logger.info(f"BlueSky client configured for embedded mode")
    
    def _ensure_cache_dir(self):
        """Prevent cache error on Windows by pre-creating cache directory."""
        try:
            import importlib.resources as ir
            res_dir = str(ir.files('bluesky.resources'))
            os.makedirs(os.path.join(res_dir, 'cache'), exist_ok=True)
        except Exception as e:
            logger.warning("Could not pre-create BlueSky cache: %s", e)
    
    def connect(self) -> bool:
        """Start/attach BlueSky (embedded) and return True on success."""
        self._ensure_cache_dir()
        try:
            # Embedded import/run
            import bluesky as bs
            from bluesky import settings
            from bluesky.tools import stack
            # If you have a local runner helper, wire it here instead.
            self.bs = stack  # use stack.stack(cmd) for commands
            self.connected = True
            logger.info("BlueSky stack ready.")
            return True
        except Exception as e:
            logger.exception("BlueSky connect failed: %s", e)
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to BlueSky simulator."""
        self.bs = None
        self.connected = False
        logger.info("Disconnected from BlueSky")
    
    def stack(self, cmd: str) -> bool:
        """Send command to BlueSky stack.
        
        Args:
            cmd: BlueSky command string
            
        Returns:
            True if command executed successfully
        """
        try:
            ok = self.bs.stack(cmd)
            if not ok:
                logger.error("BS cmd failed: %s", cmd)
            else:
                logger.debug("BS cmd OK: %s", cmd)
            return bool(ok)
        except Exception as e:
            logger.exception("BS stack error: %s", e)
            return False
    
    def send_command(self, command: str) -> bool:
        """Send command to BlueSky simulator.
        
        Args:
            command: BlueSky command string
            
        Returns:
            True if command sent successfully
        """
        return self.stack(command)
    
    def get_aircraft_states(self) -> List[AircraftState]:
        """Fetch current aircraft states from BlueSky.
        
        Returns:
            List of current aircraft states
        """
        states: List[AircraftState] = []
        try:
            # Direct arrays (embedded mode)
            import bluesky as bs
            from bluesky import traf
            n = traf.ntraf
            for i in range(n):
                cs = traf.id[i]
                lat = float(traf.lat[i])
                lon = float(traf.lon[i])
                alt_ft = float(traf.alt[i])  # typically in meters in some builds; if so, convert: * 3.28084
                hdg = float(traf.hdg[i])    # deg
                tas = float(traf.tas[i])    # m/s or kt depending build; convert if needed
                
                # Best-effort conversions (guard):
                spd_kt = tas if tas > 50 and tas < 1000 else tas * 1.94384
                roc_fpm = float(getattr(traf, "vs", [0]*n)[i]) * 196.8504 if hasattr(traf, "vs") else 0.0
                
                # Convert to AircraftState
                aircraft_state = AircraftState(
                    aircraft_id=cs,
                    timestamp=datetime.now(),
                    latitude=lat,
                    longitude=lon,
                    altitude_ft=alt_ft,
                    ground_speed_kt=spd_kt,
                    heading_deg=hdg,
                    vertical_speed_fpm=roc_fpm,
                    callsign=cs,
                    aircraft_type="UNKNOWN",
                    destination=None
                )
                states.append(aircraft_state)
            return states
        except Exception as e:
            logger.exception("Direct traf arrays failed, fallback not implemented: %s", e)
            return states
    
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
        # CRE <acid> <type> <lat> <lon> <alt> <tas> <hdg>
        return self.stack(f"CRE {aircraft_id} {aircraft_type} {lat:.6f} {lon:.6f} {alt:.0f} {spd:.0f} {hdg:.0f}")
    
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
        return self.stack(f"{aircraft_id} HDG {int(round(heading_deg))}")
    
    def set_altitude(self, aircraft_id: str, altitude_ft: float) -> bool:
        """Set aircraft altitude.
        
        Args:
            aircraft_id: Target aircraft identifier
            altitude_ft: New altitude in feet
            
        Returns:
            True if command successful
        """
        return self.stack(f"{aircraft_id} ALT {int(round(altitude_ft))}")
    
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
        return self.stack(f"{aircraft_id} DCT {waypoint}")
    
    def step_minutes(self, minutes: float) -> bool:
        """Step simulation forward by specified minutes.
        
        Args:
            minutes: Number of minutes to advance simulation
            
        Returns:
            True if step successful
        """
        # Run sim forward N seconds; adjust to your runner if needed
        secs = int(minutes * 60)
        return self.stack(f"FF {secs}")  # fast-forward; if unsupported, replace with your own tick loop
    
    def _resolution_to_bluesky_command(self, resolution: ResolutionCommand) -> Optional[str]:
        """Convert ResolutionCommand to BlueSky command format.
        
        Args:
            resolution: Resolution command to convert
            
        Returns:
            BlueSky command string or None if conversion failed
        """
        try:
            aircraft_id = resolution.target_aircraft
            
            if resolution.resolution_type.value == "heading_change" and resolution.new_heading_deg is not None:
                return f"{aircraft_id} HDG {int(round(resolution.new_heading_deg))}"
            
            elif resolution.resolution_type.value == "altitude_change" and resolution.new_altitude_ft is not None:
                return f"{aircraft_id} ALT {int(round(resolution.new_altitude_ft))}"
            
            elif resolution.resolution_type.value == "speed_change" and resolution.new_speed_kt is not None:
                return f"{aircraft_id} SPD {int(round(resolution.new_speed_kt))}"
            
            else:
                logger.error(f"Unsupported resolution type: {resolution.resolution_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to convert resolution to BlueSky command: {e}")
            return None
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
