"""BlueSky simulator interface for aircraft state and command execution.

This module provides a clean interface to BlueSky that:
- Embeds BlueSky simulator directly for better performance
- Fetches real-time aircraft states (position, velocity, flight plan)
- Executes ATC commands (HDG, ALT, SPD, DCT)
- Handles connection errors and retries gracefully
"""

from __future__ import annotations
import logging, time, math, os, atexit
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple
from pathlib import Path
import os, logging

# Import guard for BlueSky - use lazy import to prevent test failures
try:
    from bluesky import sim
    BLUESKY_AVAILABLE = True
except ImportError:
    sim = None
    BLUESKY_AVAILABLE = False
    logging.warning("BlueSky not available - using mock mode")

log = logging.getLogger(__name__)

@dataclass
class BSConfig:
    headless: bool = True
    # Add any IPC/embedding knobs your BlueSky runner needs
def _user_cache_dir() -> Path:
    # BlueSky already reads/writes here (see your logs).
    return Path.home() / "bluesky" / "cache"

class BlueSkyClient:
    def __init__(self, cfg):
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
        if os.environ.get("BLUESKY_HEADLESS") == "1" or not BLUESKY_AVAILABLE:
            log.info("BlueSky disabled for testing - using mock mode")
            self.bs = None
            self.traf = None 
            self.sim = None
            return True
            
        self._ensure_cache_dir()
        try:
            # Embedded import/run with guard
            if BLUESKY_AVAILABLE:
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
        """Create an aircraft in BlueSky. Returns True if successful."""
        if not self.is_connected():
            error_msg = f"CRITICAL ERROR: BlueSky not connected. Cannot create aircraft {cs}"
            log.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Use stack-based creation for better consistency
        return self._create_aircraft_via_stack(cs, actype, lat, lon, hdg_deg, alt_ft, spd_kt)
    
    def _create_aircraft_via_stack(self, cs: str, actype: str, lat: float, lon: float, hdg_deg: float, alt_ft: float, spd_kt: float) -> bool:
        """Create aircraft using BlueSky command stack (safer approach)."""
        # BlueSky CRE expects ft and kt; let it convert internally
        cmd = f"CRE {cs},{actype},{lat:.6f},{lon:.6f},{hdg_deg:.1f},{alt_ft:.0f},{spd_kt:.0f}"
        log.info(f"DEBUG: Creating aircraft via stack: {cmd}")
        result = self.stack(cmd)
        log.info(f"DEBUG: Stack command returned: {result}")
        return result
    
    def _create_aircraft_via_traf(self, cs: str, actype: str, lat: float, lon: float, hdg_deg: float, alt_ft: float, spd_kt: float) -> bool:
        """Create aircraft using direct traf.cre (legacy approach)."""            
        # Convert altitude from feet to meters (BlueSky uses meters)
        alt_m = alt_ft * 0.3048
        # Convert speed from knots to m/s (BlueSky uses m/s)
        spd_ms = spd_kt * 0.514444
        
        log.info(f"DEBUG: Creating aircraft {cs} type={actype} pos=({lat:.6f},{lon:.6f}) hdg={hdg_deg:.1f} alt={alt_ft:.0f}ft spd={spd_kt:.0f}kt")
        
        try:
            result = self.traf.cre(cs, actype, lat, lon, hdg_deg, alt_m, spd_ms)
            log.info(f"DEBUG: BlueSky traf.cre({cs}, {actype}, {lat:.6f}, {lon:.6f}, {hdg_deg:.1f}, {alt_m:.1f}, {spd_ms:.1f}) returned: {result}")
            
            # Also check if aircraft exists immediately after creation
            if hasattr(self.traf, 'id') and cs in self.traf.id:
                log.info(f"DEBUG: Aircraft {cs} found in traf.id after creation")
            else:
                log.warning(f"DEBUG: Aircraft {cs} NOT found in traf.id after creation")
            
            if result:
                log.info(f"Successfully created aircraft {cs} at ({lat:.6f}, {lon:.6f})")
                return True
            else:
                error_msg = f"CRITICAL ERROR: BlueSky failed to create aircraft {cs} (returned {result})"
                log.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"CRITICAL ERROR: Exception creating aircraft {cs}: {e}"
            log.exception(error_msg)
            raise RuntimeError(error_msg) from e

    def set_heading(self, cs: str, hdg_deg: float) -> bool:
        return self.stack(f"{cs} HDG {int(round(hdg_deg))}")

    def set_altitude(self, cs: str, alt_ft: float) -> bool:
        return self.stack(f"{cs} ALT {int(round(alt_ft))}")

    def direct_to(self, cs: str, wpt: str) -> bool:
        return self.stack(f"{cs} DCT {wpt}")

    def add_waypoint(self, cs: str, lat: float, lon: float, alt_ft: Optional[float] = None) -> bool:
        """Add a waypoint to aircraft's flight plan."""
        if alt_ft is not None:
            alt_m = alt_ft * 0.3048  # Convert feet to meters
            cmd = f"ADDWPT {cs} {lat:.6f} {lon:.6f} {alt_m:.1f}"
        else:
            cmd = f"ADDWPT {cs} {lat:.6f} {lon:.6f}"
        
        log.info(f"DEBUG: Sending waypoint command: {cmd}")
        result = self.stack(cmd)
        log.info(f"DEBUG: Waypoint command result: {result}")
        
        # Check if aircraft still exists after the command (guard if traf missing)
        try:
            if hasattr(self, 'traf') and hasattr(self.traf, 'id') and cs in getattr(self.traf, 'id', []):
                log.info(f"DEBUG: Aircraft {cs} still exists after waypoint command")
            else:
                log.warning(f"DEBUG: Aircraft {cs} NO LONGER EXISTS after waypoint command!")
        except Exception:
            # In mocked contexts, traf may be absent; ignore
            pass
        
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
        
    def set_speed(self, cs: str, spd_kt: float) -> bool:
         return self.stack(f"{cs} SPD {int(round(spd_kt))}")
    
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

