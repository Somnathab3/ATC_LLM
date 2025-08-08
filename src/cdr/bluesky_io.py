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
from pathlib import Path
import os, logging
from bluesky import sim
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
        self._ensure_cache_dir()
        try:
            # Embedded import/run
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
    def get_aircraft_states(self) -> dict[str, dict[str, Any]]:
        """
        Return a mapping:
        { '<CALLSIGN>': {'id','lat','lon','alt_ft','hdg_deg','spd_kt','roc_fpm'} }
        Fetched from BlueSky traf arrays with proper unit conversions.
        """
        out: dict[str, dict[str, Any]] = {}
        try:
            if not hasattr(self, "traf"):
                log.warning("BlueSkyClient not connected. Attempting to connect()…")
                if not self.connect():
                    log.error("BlueSky connect() failed; returning empty state map.")
                    return out

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
            log.exception("BS state fetch failed: %s", e)
            return out
        
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

