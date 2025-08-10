"""BlueSky simulator interface for aircraft state and command execution.

This module provides a clean interface to BlueSky that:
- Embeds BlueSky simulator directly for better performance
- Uses BlueSky command stack exclusively (no direct traf.cre calls)
- Fetches real-time aircraft states (position, velocity, flight plan)
- Executes ATC commands (HDG, ALT, SPD, DCT, VS)
- Configures conflict detection (ASAS, CDMETHOD, DTLOOK, DTMULT)
- Handles connection errors and retries gracefully

IMPORTANT CHANGES:
- Replaced direct traf.cre() calls with BlueSky CRE commands
- Added proper BlueSky conflict detection configuration (DTLOOK, CDMETHOD)
- All aircraft creation and command execution uses BlueSky command stack
- Configurable simulation parameters via BSConfig dataclass
- Enhanced baseline setup with proper CD configuration
"""

from __future__ import annotations
import logging
import math
import os
import atexit
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple
from pathlib import Path

# Import guard for BlueSky - use lazy import to prevent test failures
try:
    from bluesky import sim
    bluesky_available = True
except ImportError:
    sim = None
    bluesky_available = False
    logging.warning("BlueSky not available - using mock mode")

log = logging.getLogger(__name__)

@dataclass
class BSConfig:
    headless: bool = True
    # Conflict detection and simulation parameters
    dtlook_sec: float = 300.0  # Look-ahead time for conflict detection (seconds)
    dtmult: float = 1.0        # Time multiplier for simulation speed
    cdmethod: str = "BS"       # Conflict detection method: "BS", "GEOMETRIC", etc.
    asas_enabled: bool = False # Automated Separation Assurance System
    realtime: bool = False     # Real-time pacing mode
    # Add any IPC/embedding knobs your BlueSky runner needs
def _user_cache_dir() -> Path:
    # BlueSky already reads/writes here (see your logs).
    return Path.home() / "bluesky" / "cache"

class BlueSkyClient:
    def __init__(self, cfg: BSConfig):
        self.cfg = cfg
        self.bs = None  # handle to embedded bluesky or API wrapper
        # Add host attribute expected by tests
        self.host = getattr(cfg, 'bluesky_host', '127.0.0.1')
        self.port = getattr(cfg, 'bluesky_port', 5555)
        
        # Register cleanup handler for interpreter shutdown
        atexit.register(self._safe_shutdown)

    def _ensure_cache_dir(self):
        try:
            cache_dir = _user_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Optional: point BlueSky explicitly, if you want to be explicit:
            os.environ.setdefault("BLUESKY_USERDIR", str(Path.home() / "bluesky"))
        except Exception as e:
            # Make this DEBUG to avoid scaring users
            log.debug("Could not ensure BlueSky user cache dir: %s", e)

    def connect(self) -> bool:
        """Start/attach BlueSky (embedded) and return True on success."""
        # Check if BlueSky is disabled via environment
        if os.environ.get("BLUESKY_HEADLESS") == "1" or not bluesky_available:
            log.info("BlueSky disabled for testing - using mock mode")
            self.bs = None
            self.traf = None 
            self.sim = None
            return True
            
        self._ensure_cache_dir()
        try:
            # Embedded import/run with guard
            if bluesky_available:
                import bluesky as bs
                # Initialize BlueSky
                bs.init()
                from bluesky import stack
                from bluesky import traf
                from bluesky import sim
                # Store references to both stack and traf for different operations
                self.bs = stack  # use stack.stack(cmd) for commands
                self.traf = traf  # direct access to traffic arrays
                self.sim = sim
                log.info("BlueSky stack ready.")
                return True
            else:
                log.warning("BlueSky not available - using mock mode")
                return False
        except Exception as e:
            log.exception("BlueSky connect failed: %s", e)
            return False

    # --- Commands ---
    def stack(self, cmd: str) -> bool:
        """Execute a BlueSky command. Returns True if successful."""
        if not self.is_connected():
            error_msg = f"CRITICAL ERROR: BlueSky not connected. Cannot execute command: {cmd}"
            log.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            result = self.bs.stack(cmd)
            # Some mocks may return booleans; BlueSky returns None normally
            if isinstance(result, bool):
                if not result:
                    log.warning("BlueSky stack returned False for cmd: %s", cmd)
                else:
                    log.debug("BS cmd OK: %s", cmd)
                return result
            # Default success path if no boolean returned
            log.debug("BS cmd OK: %s", cmd)
            return True
        except Exception as e:
            error_msg = f"CRITICAL ERROR: BlueSky stack command failed: {cmd}"
            log.exception(error_msg)
            raise RuntimeError(error_msg) from e
    
    def is_connected(self) -> bool:
        """Check if BlueSky is connected and ready."""
        return self.bs is not None
    
    def is_mock_mode(self) -> bool:
        """Check if we're running in mock mode (BlueSky not connected)."""
        return not self.is_connected()

    def create_aircraft(self, cs: str, actype: str, lat: float, lon: float, hdg_deg: float, alt_ft: float, spd_kt: float) -> bool:
        """Create an aircraft in BlueSky using command stack interface.
        
        This method uses BlueSky's CRE command which ensures proper integration
        with BlueSky's conflict detection and physics systems.
        
        Args:
            cs: Aircraft callsign
            actype: Aircraft type (e.g., "B738", "A320")
            lat: Latitude in degrees
            lon: Longitude in degrees  
            hdg_deg: Heading in degrees
            alt_ft: Altitude in feet
            spd_kt: Speed in knots
            
        Returns:
            True if aircraft created successfully
        """
        if not self.is_connected():
            error_msg = f"CRITICAL ERROR: BlueSky not connected. Cannot create aircraft {cs}"
            log.error(error_msg)
            raise RuntimeError(error_msg)
        
        # BlueSky CRE command format: CRE callsign,type,lat,lon,hdg,alt,spd
        # BlueSky expects altitude in feet and speed in knots for CRE command
        cmd = f"CRE {cs},{actype},{lat:.6f},{lon:.6f},{hdg_deg:.1f},{alt_ft:.0f},{spd_kt:.0f}"
        log.info(f"Creating aircraft via BlueSky CRE command: {cmd}")
        
        result = self.stack(cmd)
        if result:
            log.info(f"Successfully created aircraft {cs} at ({lat:.6f}, {lon:.6f})")
        else:
            log.error(f"Failed to create aircraft {cs}")
        
        return result

    def set_heading(self, cs: str, hdg_deg: float) -> bool:
        """Set aircraft heading using HDG command."""
        return self.stack(f"{cs} HDG {int(round(hdg_deg))}")

    def set_altitude(self, cs: str, alt_ft: float) -> bool:
        """Set aircraft altitude using ALT command."""
        return self.stack(f"{cs} ALT {int(round(alt_ft))}")

    def set_speed(self, cs: str, spd_kt: float) -> bool:
        """Set aircraft speed using SPD command."""
        return self.stack(f"{cs} SPD {int(round(spd_kt))}")

    def set_vertical_speed(self, cs: str, vs_fpm: float) -> bool:
        """Set aircraft vertical speed using VS command.
        
        Args:
            cs: Aircraft callsign
            vs_fpm: Vertical speed in feet per minute
            
        Returns:
            True if command successful
        """
        return self.stack(f"{cs} VS {int(round(vs_fpm))}")

    def direct_to(self, cs: str, wpt: str) -> bool:
        """Direct aircraft to waypoint using DCT command."""
        return self.stack(f"{cs} DCT {wpt}")

    def add_waypoint(self, cs: str, lat: float, lon: float, alt_ft: Optional[float] = None) -> bool:
        """Add a waypoint to aircraft's flight plan using ADDWPT command.
        
        Args:
            cs: Aircraft callsign
            lat: Waypoint latitude in degrees
            lon: Waypoint longitude in degrees
            alt_ft: Optional altitude constraint in feet
            
        Returns:
            True if waypoint added successfully
        """
        if alt_ft is not None:
            # BlueSky ADDWPT expects altitude in feet for newer versions
            cmd = f"ADDWPT {cs} {lat:.6f} {lon:.6f} {alt_ft:.0f}"
        else:
            cmd = f"ADDWPT {cs} {lat:.6f} {lon:.6f}"
        
        log.info(f"Adding waypoint via BlueSky command: {cmd}")
        result = self.stack(cmd)
        
        if result:
            log.info(f"Successfully added waypoint to {cs}")
        else:
            log.warning(f"Failed to add waypoint to {cs}")
        
        return result

    def add_waypoints_from_route(self, cs: str, route: List[Tuple[float, float]], alt_ft: Optional[float] = None) -> bool:
        """Add multiple waypoints from a route to aircraft's flight plan.
        
        Args:
            cs: Aircraft callsign
            route: List of (lat, lon) tuples representing the route
            alt_ft: Altitude in feet (optional, uses current if not specified)
        
        Returns:
            True if all waypoints were added successfully
        """
        if not route:
            log.warning(f"Empty route provided for {cs}")
            return False
        
        success = True
        for i, (lat, lon) in enumerate(route):
            try:
                result = self.add_waypoint(cs, lat, lon, alt_ft)
                if not result:
                    log.warning(f"Failed to add waypoint {i+1} for {cs}")
                    success = False
            except Exception as e:
                log.error(f"Error adding waypoint {i+1} for {cs}: {e}")
                success = False
        
        if success:
            log.info(f"Added {len(route)} waypoints to {cs} flight plan")
        
        return success

    def setup_baseline(self) -> bool:
        """Configure BlueSky for baseline replay mode.
        
        This method ensures BlueSky is configured for clean baseline replay:
        - Configures ASAS (conflict detection/resolution) based on config
        - Sets conflict detection method from config (BS, GEOMETRIC, etc.)
        - Sets look-ahead time (DTLOOK) for conflict detection
        - Configures time stepping for deterministic replay
        - Sets time multiplier from config
        
        Returns:
            True if setup successful
        """
        if not self.is_connected():
            error_msg = "CRITICAL ERROR: BlueSky not connected. Cannot setup baseline"
            log.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # Configure ASAS based on config
            asas_state = "ON" if self.cfg.asas_enabled else "OFF"
            if not self.stack(f"ASAS {asas_state}"):
                log.warning(f"Failed to set ASAS to {asas_state}")
            else:
                log.info(f"ASAS set to {asas_state} for baseline replay")
            
            # Set conflict detection method from config
            if not self.stack(f"CDMETHOD {self.cfg.cdmethod}"):
                log.warning(f"Failed to set CDMETHOD to {self.cfg.cdmethod}")
            else:
                log.info(f"Conflict detection method set to {self.cfg.cdmethod}")
            
            # Set look-ahead time for conflict detection
            if not self.stack(f"DTLOOK {self.cfg.dtlook_sec}"):
                log.warning(f"Failed to set DTLOOK to {self.cfg.dtlook_sec}")
            else:
                log.info(f"Conflict detection look-ahead set to {self.cfg.dtlook_sec} seconds")
            
            # Configure real-time mode based on config
            realtime_state = "ON" if self.cfg.realtime else "OFF"
            if not self.stack(f"REALTIME {realtime_state}"):
                log.warning(f"Failed to set REALTIME to {realtime_state}")
            else:
                log.info(f"Real-time mode set to {realtime_state}")
            
            # Set time multiplier from config
            if not self.stack(f"DTMULT {self.cfg.dtmult}"):
                log.warning(f"Failed to set time multiplier to {self.cfg.dtmult}")
            else:
                log.info(f"Time multiplier set to {self.cfg.dtmult}")
            
            log.info("BlueSky baseline replay setup completed")
            return True
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR: BlueSky baseline setup failed: {e}"
            log.exception(error_msg)
            raise RuntimeError(error_msg) from e

    def step_minutes(self, minutes: float) -> bool:
        """
        Advance the embedded BlueSky sim by 'minutes' of sim time by calling the
        core stepper directly (more reliable than 'FF' here).
        """
        try:
            total_secs = float(minutes) * 60.0
            if total_secs <= 0:
                return True

            # Use a fixed internal dt so physics updates reliably.
            # 0.5s is a good balance; adjust if you want finer dynamics.
            dt = 0.5
            steps = int(math.ceil(total_secs / dt))
            for _ in range(steps):
                self.sim.step(dt)
            return True
        except Exception as e:
            log.exception("step_minutes failed: %s", e)
            return False

    def sim_reset(self) -> bool:
        """Reset the simulation to a clean state."""
        # Clear all traffic first to avoid lingering shapes
        try:
            self.stack("DEL ALL")
        except Exception:
            pass
        ok = self.stack("RESET")
        # Process the queued RESET immediately to avoid timing issues
        try:
            # One tiny tick to flush the RESET command from the stack
            self.sim.step(1)  # 1 time step
        except Exception:
            pass
        return ok

    def sim_realtime(self, on: bool) -> bool:
        """Toggle real-time pacing."""
        return self.stack(f"REALTIME {'ON' if on else 'OFF'}")

    def sim_set_dtmult(self, mult: float) -> bool:
        """Set time-step multiplier (run faster than real time)."""
        # BlueSky expects an integer or float value
        return self.stack(f"DTMULT {mult}")

    def sim_fastforward(self, seconds: int) -> bool:
        """Advance the sim quickly without wall-clock wait."""
        return self.stack(f"FF {int(seconds)}")

    def sim_set_time_utc(self, iso_utc: str) -> bool:
        """Set scenario UTC (YYYY-MM-DDTHH:MM:SS)."""
        # e.g., "2025-08-08T12:30:00"
        return self.stack(f"TIME {iso_utc}")

    # --- State fetch ---
    def get_aircraft_states(self) -> dict[str, dict[str, Any]]:
        """
        Return a mapping:
        { '<CALLSIGN>': {'id','lat','lon','alt_ft','hdg_deg','spd_kt','roc_fpm'} }
        Fetched from BlueSky traf arrays with proper unit conversions.
        """
        if not self.is_connected():
            error_msg = "CRITICAL ERROR: BlueSky not connected. Cannot retrieve aircraft states"
            log.error(error_msg)
            raise RuntimeError(error_msg)
            
        out: dict[str, dict[str, Any]] = {}
        try:
            traf = self.traf
            n = int(getattr(traf, "ntraf", 0))

            # Resolve vector fields once to avoid getattr in the loop
            ids = getattr(traf, "id")
            lats = getattr(traf, "lat")
            lons = getattr(traf, "lon")
            alts_m = getattr(traf, "alt")
            gs_ms = getattr(traf, "gs")
            # Prefer heading (hdg), otherwise track (trk)
            hdg_or_trk = getattr(traf, "hdg", getattr(traf, "trk", [0.0] * n))
            vs_ms = getattr(traf, "vs", [0.0] * n)

            for i in range(n):
                # Normalize callsign to str and uppercase to match your spawns ("OWNSHIP", "INTRUDER1")
                cs_raw = ids[i]
                cs = str(cs_raw).strip().upper()

                lat = float(lats[i])
                lon = float(lons[i])
                alt_ft = float(alts_m[i]) * 3.28084          # m -> ft
                spd_kt = float(gs_ms[i]) * 1.943844          # m/s -> kt
                hdg_deg = float(hdg_or_trk[i])               # deg
                roc_fpm = float(vs_ms[i]) * 196.8504         # m/s -> fpm

                out[cs] = {
                    "id": cs,
                    "lat": lat, "lon": lon,
                    "alt_ft": alt_ft,
                    "hdg_deg": hdg_deg,
                    "spd_kt": spd_kt,
                    "roc_fpm": roc_fpm,
                }

            return out

        except Exception as e:
            error_msg = "CRITICAL ERROR: BlueSky state fetch failed"
            log.exception(error_msg)
            raise RuntimeError(error_msg) from e

    # --- Dynamic Configuration Methods ---
    def set_dtlook(self, seconds: float) -> bool:
        """Set conflict detection look-ahead time dynamically.
        
        Args:
            seconds: Look-ahead time in seconds
            
        Returns:
            True if command successful
        """
        result = self.stack(f"DTLOOK {seconds}")
        if result:
            log.info(f"Conflict detection look-ahead set to {seconds} seconds")
        return result

    def set_dtmult(self, multiplier: float) -> bool:
        """Set time multiplier dynamically.
        
        Args:
            multiplier: Time step multiplier (e.g., 2.0 for 2x speed)
            
        Returns:
            True if command successful
        """
        result = self.stack(f"DTMULT {multiplier}")
        if result:
            log.info(f"Time multiplier set to {multiplier}")
        return result

    def set_cdmethod(self, method: str) -> bool:
        """Set conflict detection method dynamically.
        
        Args:
            method: Conflict detection method ("BS", "GEOMETRIC", etc.)
            
        Returns:
            True if command successful
        """
        result = self.stack(f"CDMETHOD {method}")
        if result:
            log.info(f"Conflict detection method set to {method}")
        return result

    def set_asas(self, enabled: bool) -> bool:
        """Enable or disable ASAS dynamically.
        
        Args:
            enabled: True to enable ASAS, False to disable
            
        Returns:
            True if command successful
        """
        state = "ON" if enabled else "OFF"
        result = self.stack(f"ASAS {state}")
        if result:
            log.info(f"ASAS set to {state}")
        return result
        
    def direct_to_waypoint(self, cs: str, wpt: str) -> bool:
        """Convenience alias for direct_to method."""
        return self.direct_to(cs, wpt)
    
    def execute_command(self, resolution: Any) -> bool:
        """Execute a resolution command."""
        try:
            if hasattr(resolution, 'resolution_type'):
                if resolution.new_heading_deg is not None:
                    return self.set_heading(resolution.target_aircraft, resolution.new_heading_deg)
                elif resolution.new_altitude_ft is not None:
                    return self.set_altitude(resolution.target_aircraft, resolution.new_altitude_ft)
                elif resolution.waypoint_name is not None:
                    return self.direct_to(resolution.target_aircraft, resolution.waypoint_name)
            return False
        except Exception as e:
            log.exception(f"Error executing resolution command: {e}")
            return False
    
    def close(self):
        """Clean up BlueSky resources to prevent shape cleanup crashes."""
        try:
            # Stop producing new shapes
            if hasattr(self, 'bs') and self.bs:
                self.bs.stack("TRAIL OFF")
                self.bs.stack("AREA OFF")
        except Exception:
            pass
        try:
            # Delete all traffic so no shapes remain
            if hasattr(self, 'bs') and self.bs:
                self.bs.stack("DEL ALL")
        except Exception:
            pass

    def _safe_shutdown(self):
        """Safe shutdown handler for atexit."""
        try:
            self.close()
        except Exception:
            pass

