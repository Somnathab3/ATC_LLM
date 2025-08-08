"""BlueSky simulator interface for aircraft state and command execution.

This module provides a clean interface to BlueSky that:
- Embeds BlueSky simulator directly for better performance
- Fetches real-time aircraft states (position, velocity, flight plan)
- Executes ATC commands (HDG, ALT, SPD, DCT)
- Handles connection errors and retries gracefully
"""

from __future__ import annotations
import logging, time, math, os
from dataclasses import dataclass
from typing import Dict, List

log = logging.getLogger(__name__)

@dataclass
class BSConfig:
    headless: bool = True
    # Add any IPC/embedding knobs your BlueSky runner needs


class BlueSkyClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = None  # handle to embedded bluesky or API wrapper

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

    # --- State fetch ---
    def get_aircraft_states(self) -> List[Dict]:
        """
        Return [{'id','lat','lon','alt_ft','hdg_deg','spd_kt','roc_fpm'}].
        Try direct access to traf arrays if embedded; otherwise provide a fallback.
        """
        states: List[Dict] = []
        try:
            # Use the stored traffic reference
            traf = self.traf
            n = traf.ntraf
            for i in range(n):
                cs = traf.id[i]
                lat = float(traf.lat[i])
                lon = float(traf.lon[i])
                # Convert altitude from meters to feet (BlueSky uses meters)
                alt_ft = float(traf.alt[i]) * 3.28084
                # Use track (trk) instead of hdg if hdg not available
                hdg = float(getattr(traf, 'hdg', getattr(traf, 'trk', [0]*n))[i])
                # Convert speed from m/s to knots (BlueSky uses m/s)
                tas_ms = float(getattr(traf, 'tas', [0]*n)[i])
                spd_kt = tas_ms * 1.94384  # m/s to knots
                # Convert vertical speed from m/s to fpm
                vs_ms = float(getattr(traf, 'vs', [0]*n)[i])
                roc_fpm = vs_ms * 196.8504  # m/s to ft/min
                
                states.append({
                    "id": cs, "lat": lat, "lon": lon, "alt_ft": alt_ft,
                    "hdg_deg": hdg, "spd_kt": spd_kt, "roc_fpm": roc_fpm
                })
            return states
        except Exception as e:
            log.exception("Direct traf arrays failed, fallback not implemented: %s", e)
            return states

