"""
Fast route follower over an entire SCAT track:
- Loads one SCAT flight.
- Builds a waypoint list from the SCAT states (first -> last).
- Flies ownship via heading guidance to each waypoint (no wall-clock sleep).
- Spawns an intruder to cross the track, forcing a predicted loss-of-sep.
- Detects conflict with a simple lookahead and resolves horizontally.
- Reaches the final SCAT waypoint and exits.
"""

import logging, math
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import src
from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState, ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.geodesy import (
    haversine_nm,  # already in your file
    bearing_deg,   # we just added
    destination_point_nm,  # we just added
    normalize_heading_deg, # optional helper if you added it
)

log = logging.getLogger("route_full")
logging.basicConfig(level=logging.INFO)

SCAT_DIR = r"F:\SCAT_extracted"
SCAT_FILE = "100000.json"   # the file you tested earlier

# --- tune here
CRUISE_SPD_KT = 420.0
CRUISE_ALT_FT = 34000.0
WPT_HIT_RADIUS_NM = 1.0
STEP_MIN = 0.2            # sim step (minutes) per inner loop tick
DTMULT = 60               # speed up the sim 60x
LOOKAHEAD_MIN = 10.0
SEP_NM = 5.0
SEP_FT = 1000.0
MAX_TURN_PER_TICK = 15.0  # limit heading change per tick

@dataclass
class MovingPoint:
    lat: float
    lon: float
    hdg_deg: float
    gs_kt: float

def build_route_from_scat(states: List[AircraftState], spacing_nm: float = 10.0) -> List[Tuple[float,float]]:
    """Downsample SCAT track to a clean waypoint list first->last."""
    if not states:
        return []
    # Ensure time order
    states = sorted(states, key=lambda s: s.timestamp)
    route = [(states[0].latitude, states[0].longitude)]
    last = route[0]
    for s in states[1:]:
        d = haversine_nm(last, (s.latitude, s.longitude))
        if d >= spacing_nm:
            route.append((s.latitude, s.longitude))
            last = route[-1]
    # Always end with the final SCAT point
    final_ll = (states[-1].latitude, states[-1].longitude)
    if haversine_nm(route[-1], final_ll) > 0.5:
        route.append(final_ll)
    return route

def project_cpa_minutes(a: MovingPoint, b: MovingPoint) -> Tuple[float, float, float]:
    """
    Very simple flat-earth relative motion CPA in NM/min.
    Returns (t_cpa_min, h_sep_at_cpa_nm, v_sep_ft) positive minima.
    Good enough for short look-ahead conflict screening.
    """
    # Convert headings to unit velocity vectors (NM/min)
    va = a.gs_kt / 60.0
    vb = b.gs_kt / 60.0
    ax = va * math.sin(math.radians(a.hdg_deg))
    ay = va * math.cos(math.radians(a.hdg_deg))
    bx = vb * math.sin(math.radians(b.hdg_deg))
    by = vb * math.cos(math.radians(b.hdg_deg))

    # crude local xy using equirectangular approx around a
    def to_xy(ref_lat, ref_lon, lat, lon):
        kx = 60.0 * math.cos(math.radians(ref_lat))  # NM per deg lon
        ky = 60.0                                   # NM per deg lat
        return ((lon - ref_lon) * kx, (lat - ref_lat) * ky)

    ax0, ay0 = to_xy(a.lat, a.lon, a.lat, a.lon)
    bx0, by0 = to_xy(a.lat, a.lon, b.lat, b.lon)

    rvx = bx - ax
    rvy = by - ay
    rx0 = bx0 - ax0
    ry0 = by0 - ay0

    rv2 = rvx*rvx + rvy*rvy
    t_cpa = 0.0 if rv2 == 0.0 else -(rx0*rvx + ry0*rvy)/rv2
    t_cpa = max(0.0, t_cpa)

    # positions at CPA (simple linear)
    ax_c = ax0 + ax * t_cpa
    ay_c = ay0 + ay * t_cpa
    bx_c = bx0 + bx * t_cpa
    by_c = by0 + by * t_cpa

    dx = bx_c - ax_c
    dy = by_c - ay_c
    h_sep = math.hypot(dx, dy)
    v_sep = abs(0.0)  # caller gives alt sep if needed elsewhere; here horizontal only
    return t_cpa, h_sep, v_sep

def main():
    # --- fast configuration
    cfg = ConfigurationSettings(
        polling_interval_min=STEP_MIN,       # we won't sleep; STEP_MIN drives sim stepping
        lookahead_time_min=LOOKAHEAD_MIN,
        min_horizontal_separation_nm=SEP_NM,
        min_vertical_separation_ft=SEP_FT,
        model_name="llama3.1:8b",
        llm_host="http://127.0.0.1:11434",
        llm_temperature=0.1,
        llm_max_tokens=512,
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        safety_buffer_factor=1.10,
        max_resolution_angle_deg=30,
    )

    # --- load SCAT track
    adapter = SCATAdapter(SCAT_DIR)
    rec = adapter.load_flight_record(Path(SCAT_DIR)/SCAT_FILE)
    states = adapter.extract_aircraft_states(rec)
    if not states:
        raise SystemExit("No states in SCAT record.")
    route = build_route_from_scat(states, spacing_nm=10.0)
    log.info("Route has %d waypoints; final waypoint = %s", len(route), route[-1])

    # --- BlueSky
    bs = BlueSkyClient(cfg)
    assert bs.connect(), "BlueSky connect failed"
    bs.sim_reset()
    bs.sim_realtime(False)
    bs.sim_set_dtmult(DTMULT)
    bs.sim_fastforward(60)

    # create ownship at route[0]
    (lat0, lon0) = route[0]
    if len(route) > 1:
        hdg0 = bearing_deg(lat0, lon0, route[1][0], route[1][1])
    else:
        hdg0 = 90.0
    bs.create_aircraft("OWNSHIP", "A320", lat0, lon0, hdg0, CRUISE_ALT_FT, CRUISE_SPD_KT)

    # --- spawn an intruder to cross near the middle of the route
    mid = route[len(route)//2]
    # place intruder start ~40 NM west of mid, flying eastbound (90°)
    intr_lat, intr_lon = destination_point_nm(mid[0], mid[1], 270.0, 40.0)
    bs.create_aircraft("INTRUDER1", "B738", intr_lat, intr_lon, 90.0, CRUISE_ALT_FT, CRUISE_SPD_KT)

    # --- run loop: fly to each waypoint, conflict screen, resolve if needed
    wpt_idx = 1
    done = False
    last_cmd_hdg = hdg0

    # give the sim one small step to settle
    bs.step_minutes(0.1)

    while not done:
        states_now = bs.get_aircraft_states()
        me = states_now.get("OWNSHIP")
        it = states_now.get("INTRUDER1")
        if me is None or wpt_idx >= len(route):
            break

        tgt_lat, tgt_lon = route[wpt_idx]
        dist_to_wpt = haversine_nm((me.latitude, me.longitude), (tgt_lat, tgt_lon))

        # advance waypoint
        if dist_to_wpt <= WPT_HIT_RADIUS_NM:
            wpt_idx += 1
            if wpt_idx >= len(route):
                done = True
                log.info("Final waypoint reached.")
                break
            tgt_lat, tgt_lon = route[wpt_idx]

        # steer to next waypoint (bounded heading change per tick)
        desired_hdg = bearing_deg(me.latitude, me.longitude, tgt_lat, tgt_lon)
        # small turn limiter to avoid over-commanding
        def ang_diff(a, b):
            d = (a - b + 540.0) % 360.0 - 180.0
            return d
        delta = ang_diff(desired_hdg, me.heading_deg)
        cmd = me.heading_deg + max(-MAX_TURN_PER_TICK, min(MAX_TURN_PER_TICK, delta))
        cmd = normalize_heading_deg(cmd)
        if abs(delta) > 1.0:
            bs.set_heading("OWNSHIP", cmd)
            last_cmd_hdg = cmd

        # simple conflict screen with 10-min lookahead (horizontal only)
        if it is not None:
            a = MovingPoint(me.latitude, me.longitude, me.heading_deg, max(me.ground_speed_kt, 1.0))
            b = MovingPoint(it.latitude, it.longitude, it.heading_deg, max(it.ground_speed_kt, 1.0))
            t_cpa, h_sep, _ = project_cpa_minutes(a, b)
            if 0.0 <= t_cpa <= LOOKAHEAD_MIN and h_sep < SEP_NM:
                # Horizontal resolution: turn right 30 deg for a short while
                new_hdg = normalize_heading_deg(me.heading_deg + 30.0)
                log.info("Conflict predicted in %.1f min at %.1f NM. Turning to %3.0f°",
                         t_cpa, h_sep, new_hdg)
                bs.set_heading("OWNSHIP", new_hdg)

        # advance sim quickly
        bs.step_minutes(STEP_MIN)

    log.info("Route follower finished.")

if __name__ == "__main__":
    main()
