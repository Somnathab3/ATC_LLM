"""
Fast route follower over an entire SCAT track:
- Loads one SCAT flight.
- Builds a waypoint list from the SCAT states (first -> last).
- Flies ownship via heading guidance to each waypoint (no wall-clock sleep).
- Spawns an intruder to cross the track, forcing a predicted loss-of-sep.
- Detects conflict with a simple lookahead and resolves horizontally.
- Reaches the final SCAT waypoint and exits.
"""
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import src modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
import logging, math, time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import src
from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState, ConfigurationSettings, ConflictPrediction, ResolutionCommand, ResolutionType
from src.cdr.detect import predict_conflicts
from src.cdr.llm_client import LlamaClient
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.geodesy import (
    haversine_nm,  # already in your file
    bearing_deg,   # we just added
    destination_point_nm,  # we just added
    normalize_heading_deg, # optional helper if you added it
)
from src.cdr.pipeline import CDRPipeline
from src.cdr.llm_client import LlamaClient
from src.cdr.resolve import execute_resolution

log = logging.getLogger("route_full")
logging.basicConfig(level=logging.INFO)

SCAT_DIR = r"F:\SCAT_extracted"
SCAT_FILE = "100000.json"   # the file you tested earlier

# --- tune here
CRUISE_SPD_KT = 420.0
CRUISE_ALT_FT = 34000.0
WPT_HIT_RADIUS_NM = 1.0
STEP_MIN = 2            # sim step (minutes) per inner loop tick
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

def _states_by_callsign(raw):
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    out = {}
    for s in raw:
        cs = str(s.get("id", "")).strip().upper()
        if cs:
            out[cs] = s
    return out
def _convert_to_aircraft_state(aircraft_dict: dict, aircraft_id: str) -> AircraftState:
    """Convert BlueSky aircraft state dict to AircraftState object."""
    from datetime import datetime
    return AircraftState(
        aircraft_id=aircraft_id,
        timestamp=datetime.now(),
        latitude=float(aircraft_dict.get("lat", 0)),
        longitude=float(aircraft_dict.get("lon", 0)),
        altitude_ft=float(aircraft_dict.get("alt_ft", 0)),
        ground_speed_kt=float(aircraft_dict.get("spd_kt", 0)),
        heading_deg=float(aircraft_dict.get("hdg_deg", 0)),
        vertical_speed_fpm=float(aircraft_dict.get("roc_fpm", 0))
    )

def _generate_llm_resolution(conflict: ConflictPrediction, ownship: AircraftState, traffic: List[AircraftState], llm_client: LlamaClient) -> Optional[ResolutionCommand]:
    """Generate LLM-based resolution for a conflict."""
    try:
        # Build detect output for LLM resolution input
        detect_out = {
            "conflict": True,
            "intruders": [{
                "id": traffic[0].callsign if traffic else "INTRUDER1",
                "cpa_time_min": conflict.time_to_cpa_min,
                "cpa_distance_nm": conflict.distance_at_cpa_nm
            }],
            "assessment": f"Conflict detected with separation {conflict.distance_at_cpa_nm:.1f} NM in {conflict.time_to_cpa_min:.1f} min"
        }
        
        # Use LLM client to generate resolution
        response = llm_client.generate_resolution(detect_out)
        if not response:
            return None
            
        # Parse the resolution response
        parsed = llm_client.parse_resolve_response(response)
        action = parsed.get("action", "").lower()
        value = float(parsed.get("value", 0))
        
        # Map action to resolution type and create command
        if action in ["turn_left", "turn_right"]:
            new_heading = ownship.heading_deg
            if action == "turn_left":
                new_heading = normalize_heading_deg(ownship.heading_deg - value)
            else:
                new_heading = normalize_heading_deg(ownship.heading_deg + value)
                
            return ResolutionCommand(
                resolution_id=f"llm_hdg_{conflict.ownship_id}_{datetime.now().strftime('%H%M%S')}",
                target_aircraft=conflict.ownship_id,
                resolution_type=ResolutionType.HEADING_CHANGE,
                new_heading_deg=new_heading,
                new_speed_kt=None,
                new_altitude_ft=None,
                issue_time=datetime.now(),
                is_validated=True,
                safety_margin_nm=5.0
            )
        elif action in ["climb", "descend"]:
            new_altitude = ownship.altitude_ft
            if action == "climb":
                new_altitude += value
            else:
                new_altitude -= value
                
            return ResolutionCommand(
                resolution_id=f"llm_alt_{conflict.ownship_id}_{datetime.now().strftime('%H%M%S')}",
                target_aircraft=conflict.ownship_id,
                resolution_type=ResolutionType.ALTITUDE_CHANGE,
                new_heading_deg=None,
                new_speed_kt=None,
                new_altitude_ft=new_altitude,
                issue_time=datetime.now(),
                is_validated=True,
                safety_margin_nm=5.0
            )
        elif action in ["speed_up", "slow_down"]:
            new_speed = ownship.ground_speed_kt
            if action == "speed_up":
                new_speed += value
            else:
                new_speed -= value
                
            return ResolutionCommand(
                resolution_id=f"llm_spd_{conflict.ownship_id}_{datetime.now().strftime('%H%M%S')}",
                target_aircraft=conflict.ownship_id,
                resolution_type=ResolutionType.SPEED_CHANGE,
                new_heading_deg=None,
                new_speed_kt=new_speed,
                new_altitude_ft=None,
                issue_time=datetime.now(),
                is_validated=True,
                safety_margin_nm=5.0
            )
        else:
            log.warning("Unknown LLM action: %s", action)
            return None
            
    except Exception as e:
        log.error("LLM resolution generation failed: %s", e)
        return None


def _execute_llm_resolution(resolution_cmd: ResolutionCommand, bs: BlueSkyClient) -> bool:
    """Execute a resolution command via BlueSky."""
    try:
        if resolution_cmd.resolution_type == ResolutionType.HEADING_CHANGE and resolution_cmd.new_heading_deg is not None:
            result = bs.set_heading(resolution_cmd.target_aircraft, resolution_cmd.new_heading_deg)
            log.info("LLM Resolution: Set heading %s to %.1f° - %s", 
                    resolution_cmd.target_aircraft, resolution_cmd.new_heading_deg, "SUCCESS" if result else "FAILED")
            return result
        elif resolution_cmd.resolution_type == ResolutionType.ALTITUDE_CHANGE and resolution_cmd.new_altitude_ft is not None:
            result = bs.set_altitude(resolution_cmd.target_aircraft, resolution_cmd.new_altitude_ft)
            log.info("LLM Resolution: Set altitude %s to %.0f ft - %s", 
                    resolution_cmd.target_aircraft, resolution_cmd.new_altitude_ft, "SUCCESS" if result else "FAILED")
            return result
        elif resolution_cmd.resolution_type == ResolutionType.SPEED_CHANGE and resolution_cmd.new_speed_kt is not None:
            result = bs.set_speed(resolution_cmd.target_aircraft, resolution_cmd.new_speed_kt)
            log.info("LLM Resolution: Set speed %s to %.0f kt - %s", 
                    resolution_cmd.target_aircraft, resolution_cmd.new_speed_kt, "SUCCESS" if result else "FAILED")
            return result
        else:
            log.warning("Unsupported resolution type: %s", resolution_cmd.resolution_type)
            return False
    except Exception as e:
        log.error("Failed to execute LLM resolution: %s", e)
        return False

# --- angle diff helper ---
def _ang_diff(a, b):
    d = (a - b + 540.0) % 360.0 - 180.0
    return d

def main():
    # --- fast configuration
    cfg = ConfigurationSettings(
        polling_interval_min=STEP_MIN,       # we won't sleep; STEP_MIN drives sim stepping
        lookahead_time_min=LOOKAHEAD_MIN,
        min_horizontal_separation_nm=SEP_NM,
        min_vertical_separation_ft=SEP_FT,
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        llm_max_tokens=512,
        safety_buffer_factor=1.10,
        max_resolution_angle_deg=30,
        max_altitude_change_ft=2000.0,
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=100.0
    )

    # --- Initialize LLM Client and CDR Pipeline
    llm_client = LlamaClient(config=cfg)
    
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
    
    # --- Initialize LLM client for intelligent conflict resolution
    llm_client = LlamaClient(cfg)
    
    bs.sim_reset()
    bs.sim_realtime(False)
    bs.sim_set_dtmult(DTMULT)
    bs.sim_fastforward(300)

    # create ownship at route[0] with correct initial speed
    (lat0, lon0) = route[0]
    if len(route) > 1:
        hdg0 = bearing_deg(lat0, lon0, route[1][0], route[1][1])
    else:
        hdg0 = 90.0
    bs.create_aircraft("OWNSHIP", "A320", lat0, lon0, hdg0, CRUISE_ALT_FT, CRUISE_SPD_KT)
    log.info("Created OWNSHIP with initial speed %d kt at lat=%.4f, lon=%.4f", CRUISE_SPD_KT, lat0, lon0)

    # --- spawn an intruder to cross near the middle of the route
    mid = route[len(route)//2]
    # place intruder start ~40 NM west of mid, flying eastbound (90°)
    intr_lat, intr_lon = destination_point_nm(mid[0], mid[1], 270.0, 40.0)
    bs.create_aircraft("INTRUDER1", "B738", intr_lat, intr_lon, 90.0, CRUISE_ALT_FT, CRUISE_SPD_KT)
    log.info("Created INTRUDER1 with initial speed %d kt at lat=%.4f, lon=%.4f", CRUISE_SPD_KT, intr_lat, intr_lon)
    
    # Wait a moment for aircraft creation to take effect
    log.info("Waiting for aircraft creation to process...")
    time.sleep(2)
    
    # Check if aircraft were created successfully before proceeding
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        states = _states_by_callsign(bs.get_aircraft_states())
        if "OWNSHIP" in states and "INTRUDER1" in states:
            log.info("Both aircraft found in simulation")
            break
        log.info(f"Waiting for aircraft to appear in simulation (attempt {retry_count + 1}/{max_retries})")
        time.sleep(1)
        retry_count += 1
    
    if retry_count >= max_retries:
        log.error("Aircraft creation failed - aircraft not found in simulation")
        return
    
    # Arm the aircraft heading and altitude (speed should already be set)
    bs.set_heading("OWNSHIP", hdg0)
    bs.set_altitude("OWNSHIP", CRUISE_ALT_FT)
    bs.set_speed("OWNSHIP", CRUISE_SPD_KT)
    log.info("Armed OWNSHIP: hdg=%.1f, alt=%d, spd=%d", hdg0, CRUISE_ALT_FT, CRUISE_SPD_KT)
    
    bs.set_heading("INTRUDER1", 90.0)
    bs.set_altitude("INTRUDER1", CRUISE_ALT_FT)
    bs.set_speed("INTRUDER1", CRUISE_SPD_KT)
    log.info("Armed INTRUDER1: hdg=90.0, alt=%d, spd=%d", CRUISE_ALT_FT, CRUISE_SPD_KT)
    
    # Give the aircraft a small time step to initialize their guidance
    log.info("Stepping simulation for aircraft initialization...")
    for i in range(3):
        try:
            success = bs.step_minutes(0.1)
            if not success:
                log.warning(f"Step {i+1} failed")
            time.sleep(0.1)  # Small pause between steps
        except Exception as e:
            log.error(f"Step {i+1} error: {e}")
            break
    
    # Check initial state after initialization
    initial_states = _states_by_callsign(bs.get_aircraft_states())
    if "OWNSHIP" in initial_states:
        own = initial_states["OWNSHIP"]
        log.info("Initial OWNSHIP state: lat=%.4f, lon=%.4f, hdg=%.1f°, spd=%.1fkt, alt=%.0fft", 
                float(own["lat"]), float(own["lon"]), float(own["hdg_deg"]), 
                float(own["spd_kt"]), float(own["alt_ft"]))
    else:
        log.warning("OWNSHIP not found in initial states!")
    
    if "INTRUDER1" in initial_states:
        intr = initial_states["INTRUDER1"]
        log.info("Initial INTRUDER1 state: lat=%.4f, lon=%.4f, hdg=%.1f°, spd=%.1fkt, alt=%.0fft", 
                float(intr["lat"]), float(intr["lon"]), float(intr["hdg_deg"]), 
                float(intr["spd_kt"]), float(intr["alt_ft"]))
    else:
        log.warning("INTRUDER1 not found in initial states!")

    # --- run loop: fly to each waypoint, conflict screen, resolve if needed
    wpt_idx = 1
    last_cmd_hdg = hdg0

    # Give the sim one small step to settle and prime first state
    log.info("Priming simulation for first state snapshot...")
    for attempt in range(5):
        try:
            success = bs.step_minutes(0.5)
            if success:
                break
            log.warning(f"Step attempt {attempt + 1} failed")
        except Exception as e:
            log.error(f"Step attempt {attempt + 1} error: {e}")
            if attempt >= 4:
                log.error("Failed to prime simulation, exiting")
                return
            time.sleep(0.5)

    # prime a first state snapshot so we can compute dist/hdg/speed
    states_now = _states_by_callsign(bs.get_aircraft_states())
    retry_count = 0
    max_state_retries = 10
    while states_now.get("OWNSHIP") is None and retry_count < max_state_retries:
        log.info(f"Waiting for OWNSHIP state (attempt {retry_count + 1}/{max_state_retries})")
        try:
            bs.step_minutes(0.5)
            states_now = _states_by_callsign(bs.get_aircraft_states())
        except Exception as e:
            log.error(f"Error stepping simulation: {e}")
        retry_count += 1
        time.sleep(0.5)
    
    if states_now.get("OWNSHIP") is None:
        log.error("Could not get OWNSHIP state after multiple attempts, exiting")
        return    # compute first metrics
    me = states_now["OWNSHIP"]
    me_lat = float(me["lat"]);  me_lon = float(me["lon"])
    me_hdg = float(me["hdg_deg"]);  me_kt = max(float(me["spd_kt"]), 1.0)
    tgt_lat, tgt_lon = route[wpt_idx]
    dist_to_wpt = haversine_nm((me_lat, me_lon), (tgt_lat, tgt_lon))

    # now create the progress bar (we finally have values to show)
    pbar = tqdm(total=len(route) - 1, desc="Waypoints reached", unit="wpt", mininterval=0.5)
    pbar.set_postfix({"dist_nm": f"{dist_to_wpt:6.2f}", "hdg": f"{me_hdg:6.1f}", "kt": f"{me_kt:5.0f}"})

    # optional watchdog
    stuck_counter = 0
    prev_dist = dist_to_wpt

    # main loop
    done = False
    while not done:
        states_now = _states_by_callsign(bs.get_aircraft_states())
        me = states_now.get("OWNSHIP")
        it = states_now.get("INTRUDER1")

        if me is None:
            bs.step_minutes(STEP_MIN)
            continue

        me_lat = float(me["lat"]);  me_lon = float(me["lon"])
        me_hdg = float(me["hdg_deg"]);  me_kt = max(float(me["spd_kt"]), 1.0)

        tgt_lat, tgt_lon = route[wpt_idx]
        dist_to_wpt = haversine_nm((me_lat, me_lon), (tgt_lat, tgt_lon))

        # Debug info every 10 iterations
        if stuck_counter % 10 == 0:
            log.info("Debug: pos=(%.4f,%.4f), hdg=%.1f°, spd=%.1fkt, dist=%.2fNM to wpt %d", 
                    me_lat, me_lon, me_hdg, me_kt, dist_to_wpt, wpt_idx)

        # progress bar live metrics
        pbar.set_postfix({"dist_nm": f"{dist_to_wpt:6.2f}", "hdg": f"{me_hdg:6.1f}", "kt": f"{me_kt:5.0f}"})

        # Check if aircraft speed is zero and re-arm if needed (but not every iteration)
        if me_kt <= 1.0 and stuck_counter % 5 == 0:  # Only every 5 iterations
            log.warning("Aircraft speed is %.1f kt, re-arming speed to %d kt", me_kt, CRUISE_SPD_KT)
            bs.set_speed("OWNSHIP", CRUISE_SPD_KT)
            if it is not None:
                bs.set_speed("INTRUDER1", CRUISE_SPD_KT)

        # waypoint handling
        if dist_to_wpt <= WPT_HIT_RADIUS_NM:
            pbar.update(1)
            wpt_idx += 1
            if wpt_idx >= len(route):
                pbar.close()
                log.info("Final waypoint reached.")
                break
            tgt_lat, tgt_lon = route[wpt_idx]

        # steer toward next waypoint
        desired_hdg = bearing_deg(me_lat, me_lon, tgt_lat, tgt_lon)
        delta = _ang_diff(desired_hdg, me_hdg)
        cmd = normalize_heading_deg(me_hdg + max(-MAX_TURN_PER_TICK, min(MAX_TURN_PER_TICK, delta)))
        if abs(delta) > 1.0:
            result = bs.set_heading("OWNSHIP", cmd)
            if stuck_counter % 20 == 0:  # Debug every 20 iterations
                log.info("Heading command: desired=%.1f°, current=%.1f°, delta=%.1f°, cmd=%.1f°, result=%s", 
                        desired_hdg, me_hdg, delta, cmd, result)

        # LLM-based conflict detection and resolution
        if it is not None:
            it_lat = float(it["lat"]);  it_lon = float(it["lon"])
            it_hdg = float(it["hdg_deg"]);  it_kt = max(float(it["spd_kt"]), 1.0)
            
            # Convert aircraft states for LLM analysis
            ownship_state = _convert_to_aircraft_state(me, "OWNSHIP")
            intruder_state = _convert_to_aircraft_state(it, "INTRUDER1")
            
            # Use CDR system for conflict detection and resolution
            try:
                # Use the public predict_conflicts function from detect module
                conflicts = predict_conflicts(
                    ownship=ownship_state,
                    traffic=[intruder_state],
                    lookahead_minutes=LOOKAHEAD_MIN,
                    time_step_seconds=30.0
                )
                
                if conflicts:
                    log.info("LLM detected %d conflicts", len(conflicts))
                    for conflict in conflicts:
                        log.info("Conflict: %s vs %s, CPA in %.1f min, sep %.2f NM", 
                                conflict.ownship_id, conflict.intruder_id, 
                                conflict.time_to_cpa_min, conflict.distance_at_cpa_nm)
                        
                        # Generate LLM-based resolution using llm_client
                        resolution = _generate_llm_resolution(conflict, ownship_state, [intruder_state], llm_client)
                        if resolution:
                            # Execute the resolution using our helper function
                            success = _execute_llm_resolution(resolution, bs)
                            if success:
                                log.info("LLM resolution executed: %s", resolution.resolution_type.value)
                            else:
                                log.warning("LLM resolution execution failed")
                        else:
                            log.warning("No LLM resolution generated for conflict")
                else:
                    # Still do basic conflict check for immediate threats using legacy method
                    a = MovingPoint(me_lat, me_lon, me_hdg, me_kt)
                    b = MovingPoint(it_lat, it_lon, it_hdg, it_kt)
                    t_cpa, h_sep, _ = project_cpa_minutes(a, b)
                    if 0.0 <= t_cpa <= LOOKAHEAD_MIN and h_sep < SEP_NM:
                        log.info("Legacy conflict detection: CPA in %.1f min at %.1f NM", t_cpa, h_sep)
                        # Create a mock conflict for LLM resolution
                        mock_conflict = ConflictPrediction(
                            ownship_id="OWNSHIP",
                            intruder_id="INTRUDER1", 
                            time_to_cpa_min=t_cpa,
                            distance_at_cpa_nm=h_sep,
                            altitude_diff_ft=0.0,
                            is_conflict=True,
                            severity_score=0.8,
                            conflict_type="horizontal",
                            prediction_time=datetime.now(),
                            confidence=0.9
                        )
                        # Generate LLM-based resolution
                        resolution = _generate_llm_resolution(mock_conflict, ownship_state, [intruder_state], llm_client)
                        if resolution:
                            success = _execute_llm_resolution(resolution, bs)
                            if success:
                                log.info("LLM legacy resolution executed: %s", resolution.resolution_type.value)
                        else:
                            # Fallback to simple resolution if LLM fails
                            new_hdg = normalize_heading_deg(me_hdg + 30.0)
                            log.info("Fallback resolution: Turning to %3.0f°", new_hdg)
                            bs.set_heading("OWNSHIP", new_hdg)
                            
            except Exception as e:
                log.error("CDR system error: %s", e)
                # Fallback to simple conflict logic if CDR system fails
                a = MovingPoint(me_lat, me_lon, me_hdg, me_kt)
                b = MovingPoint(it_lat, it_lon, it_hdg, it_kt)
                t_cpa, h_sep, _ = project_cpa_minutes(a, b)
                if 0.0 <= t_cpa <= LOOKAHEAD_MIN and h_sep < SEP_NM:
                    new_hdg = normalize_heading_deg(me_hdg + 30.0)
                    log.info("Fallback conflict resolution: Turning to %3.0f°", new_hdg)
                    bs.set_heading("OWNSHIP", new_hdg)

        # watchdog: detect lack of closure for several ticks
        if dist_to_wpt > prev_dist - 0.05:
            stuck_counter += 1
        else:
            stuck_counter = 0
        prev_dist = dist_to_wpt

        if stuck_counter >= 10:
            log.warning("No progress; re-arming guidance and nudging time.")
            bs.set_speed("OWNSHIP", CRUISE_SPD_KT)
            bs.set_altitude("OWNSHIP", CRUISE_ALT_FT)
            bs.set_heading("OWNSHIP", desired_hdg)
            stuck_counter = 0

        # Step simulation with error handling and timeout
        step_success = False
        for step_attempt in range(3):
            try:
                step_success = bs.step_minutes(STEP_MIN)
                if step_success:
                    break
                log.warning(f"Simulation step failed, attempt {step_attempt + 1}/3")
                time.sleep(0.1)
            except KeyboardInterrupt:
                log.info("Simulation interrupted by user")
                done = True
                break
            except Exception as e:
                log.error(f"Simulation step error (attempt {step_attempt + 1}/3): {e}")
                time.sleep(0.2)
                
        if not step_success and not done:
            log.error("All simulation step attempts failed, exiting")
            break


    log.info("Route follower finished.")

    # Collect final metrics and create summary
    try:
        final_states = _states_by_callsign(bs.get_aircraft_states())
        if "OWNSHIP" in final_states:
            final_own = final_states["OWNSHIP"]
            log.info("Final OWNSHIP position: lat=%.4f, lon=%.4f, alt=%.0f ft", 
                    float(final_own["lat"]), float(final_own["lon"]), float(final_own["alt_ft"]))
            
            # Calculate total distance traveled
            final_pos = (float(final_own["lat"]), float(final_own["lon"]))
            total_distance = haversine_nm(route[0], final_pos)
            log.info("Total distance from start: %.2f NM", total_distance)
            
            # Create a simple results summary
            results = {
                "simulation_completed": True,
                "waypoints_reached": wpt_idx,
                "total_waypoints": len(route),
                "final_position": final_pos,
                "total_distance_nm": total_distance,
                "completion_percentage": (wpt_idx / len(route)) * 100
            }
            
            import json
            with open("simulation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            log.info("Results saved to simulation_results.json")
            log.info("Simulation Summary: %d/%d waypoints reached (%.1f%% complete)", 
                    wpt_idx, len(route), results["completion_percentage"])
        else:
            log.warning("Could not get final OWNSHIP state for metrics")
    except Exception as e:
        log.error("Error collecting final metrics: %s", e)

if __name__ == "__main__":
    main()
