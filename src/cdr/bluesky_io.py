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
from typing import Any

log = logging.getLogger(__name__)

@dataclass
class BSConfig:
    headless: bool = True
    # Add any IPC/embedding knobs your BlueSky runner needs


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
        # Prevent earlier cache error on Windows
        try:
            import importlib.resources as ir
            res_dir = str(ir.files('bluesky.resources'))
            os.makedirs(os.path.join(res_dir, 'cache'), exist_ok=True)
        except Exception as e:
            log.warning("Could not pre-create BlueSky cache: %s", e)

    def connect(self) -> bool:
        """Start/attach BlueSky (embedded) and return True on success."""
        self._ensure_cache_dir()
        try:
            # Embedded import/run
            import bluesky as bs
            # Initialize BlueSky
            bs.init()
            from bluesky import stack
            from bluesky import traf
            # Store references to both stack and traf for different operations
            self.bs = stack  # use stack.stack(cmd) for commands
            self.traf = traf  # direct access to traffic arrays
            log.info("BlueSky stack ready.")
            return True
        except Exception as e:
            log.exception("BlueSky connect failed: %s", e)
            return False

    # --- Commands ---
    def stack(self, cmd: str) -> bool:
        try:
            self.bs.stack(cmd)  # BlueSky stack returns None, not boolean
            log.debug("BS cmd OK: %s", cmd)
            return True
        except Exception as e:
            log.exception("BS stack error: %s", e)
            return False

    def create_aircraft(self, cs: str, actype: str, lat: float, lon: float, hdg_deg: float, alt_ft: float, spd_kt: float) -> bool:
        # Use BlueSky's direct traffic creation method
        # Convert altitude from feet to meters (BlueSky uses meters)
        alt_m = alt_ft * 0.3048
        # Convert speed from knots to m/s (BlueSky uses m/s)
        spd_ms = spd_kt * 0.514444
        
        try:
            result = self.traf.cre(cs, actype, lat, lon, hdg_deg, alt_m, spd_ms)
            if result:
                log.debug(f"Created aircraft {cs} at ({lat:.6f}, {lon:.6f})")
            else:
                log.error(f"Failed to create aircraft {cs}")
            return bool(result)
        except Exception as e:
            log.exception(f"Error creating aircraft {cs}: %s", e)
            return False

    def set_heading(self, cs: str, hdg_deg: float) -> bool:
        return self.stack(f"{cs} HDG {int(round(hdg_deg))}")

    def set_altitude(self, cs: str, alt_ft: float) -> bool:
        return self.stack(f"{cs} ALT {int(round(alt_ft))}")

    def direct_to(self, cs: str, wpt: str) -> bool:
        return self.stack(f"{cs} DCT {wpt}")

    def step_minutes(self, minutes: float) -> bool:
        # Run sim forward N seconds; adjust to your runner if needed
        secs = int(minutes * 60)
        return self.stack(f"FF {secs}")  # fast-forward; if unsupported, replace with your own tick loop
    def sim_reset(self) -> bool:
        """Reset the simulation to a clean state."""
        # Clear all traffic first to avoid lingering shapes
        try:
            self.stack("DEL ALL")
        except Exception:
            pass
        return self.stack("RESET")

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
    def get_aircraft_states(self) -> list[dict[str, Any]]:
        """
        Return [{'id','lat','lon','alt_ft','hdg_deg','spd_kt','roc_fpm'}].
        Fetches states from BlueSky traf arrays with proper unit conversions.
        """
        states: list[dict[str, Any]] = []
        try:
            if not hasattr(self, "traf"):
                log.warning("BlueSkyClient not connected. Attempting to connect()…")
                if not self.connect():
                    log.error("BlueSky connect() failed; returning empty state list.")
                    return states
            traf = self.traf
            n = traf.ntraf
            for i in range(n):
                # Common BlueSky arrays: id, lat, lon, alt [m], gs [m/s], hdg/trk [deg]
                cs = traf.id[i]
                lat = float(traf.lat[i])
                lon = float(traf.lon[i])
                alt_ft = float(traf.alt[i]) * 3.28084  # m -> ft
                spd_kt = float(traf.gs[i]) * 1.943844  # m/s -> kt
                # Prefer heading if available, else use track
                hdg_deg = float(getattr(traf, "hdg", getattr(traf, "trk", [0]*n))[i])
                roc_fpm = float(getattr(traf, "vs", [0]*n)[i]) * 196.8504  # m/s -> fpm

                states.append({
                    "id": cs, "lat": lat, "lon": lon,
                    "alt_ft": alt_ft, "hdg_deg": hdg_deg,
                    "spd_kt": spd_kt, "roc_fpm": roc_fpm,
                })
            return states
        except Exception as e:
            log.exception("BS state fetch failed: %s", e)
            return states
    
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

