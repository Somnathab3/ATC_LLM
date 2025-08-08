# scripts/run_scat_once.py
import logging
from pathlib import Path
from datetime import timedelta
import src.cdr
# scripts/run_scat_once.py
import logging
from pathlib import Path
from datetime import timedelta

from src.cdr.schemas import ConfigurationSettings, AircraftState
from src.cdr.pipeline import CDRPipeline
from src.cdr.scat_adapter import SCATAdapter
from src.cdr.bluesky_io import BlueSkyClient      # <-- missing before
from src.cdr.reporting import Sprint5Reporter     # <-- needed for reports

log = logging.getLogger("run_scat")
logging.basicConfig(level=logging.INFO)

SCAT_DIR = r"F:\SCAT_extracted"
SCAT_FILE = "100000.json"                # ownship track
AIRSPACE_JSON = r"F:\SCAT_extracted\airspace.json"  # not used in v1


def pick_window(states, minutes=30):
    """Take a time slice of `minutes` starting at the first timestamp."""
    t0 = min(s.timestamp for s in states)
    return [s for s in states if t0 <= s.timestamp <= t0 + timedelta(minutes=minutes)]


def main():
    cfg = ConfigurationSettings(
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        model_name="llama3.1:8b",
        # if your schema actually has these next four fields, keep them; otherwise drop them
        llm_host="http://127.0.0.1:11434",
        llm_temperature=0.1,
        llm_max_tokens=1024,
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        safety_buffer_factor=1.10,   # must be >1.0 unless you changed schema to ge=1.0
        max_resolution_angle_deg=30,
    )
    print("Effective safety_buffer_factor:", cfg.safety_buffer_factor)

    adapter = SCATAdapter(SCAT_DIR)
    rec = adapter.load_flight_record(Path(SCAT_DIR) / SCAT_FILE)
    states = adapter.extract_aircraft_states(rec)
    window = pick_window(states, minutes=25)
    if not window:
        raise SystemExit("No states in the selected time window.")

    # Pydantic v2: model_copy instead of copy
    own0: AircraftState = window[0].model_copy(update={"aircraft_id": "OWNSHIP"})
    intruders = [s.model_copy(update={"aircraft_id": f"INTRUDER_{i}"})
                 for i, s in enumerate(window[1:6], start=1)]

    # Start BlueSky once and reuse
    bs = BlueSkyClient(cfg)
    assert bs.connect(), "BlueSky connect failed"

    bs.create_aircraft(
        own0.aircraft_id, "A320",
        own0.latitude, own0.longitude,
        own0.heading_deg, own0.altitude_ft, own0.ground_speed_kt,
    )
    for t in intruders:
        bs.create_aircraft(
            t.aircraft_id, "B738",
            t.latitude, t.longitude,
            t.heading_deg, t.altitude_ft, t.ground_speed_kt,
        )
    bs.step_minutes(1.0)

    # BASELINE (LLM disabled)
    log.info("=== BASELINE run ===")
    base_cfg = cfg.model_copy(update={"llm_enabled": False})
    base_pipe = CDRPipeline(base_cfg)
    base_pipe.bluesky_client = bs     # <-- reuse the connected client
    base_pipe.run(max_cycles=2, ownship_id=own0.aircraft_id)

    # LLM run
    log.info("=== LLM run ===")
    llm_cfg = cfg.model_copy(update={"llm_enabled": True})
    llm_pipe = CDRPipeline(llm_cfg)
    llm_pipe.bluesky_client = bs      # <-- reuse the same client
    llm_pipe.run(max_cycles=2, ownship_id=own0.aircraft_id)

    # Reports
    m1 = base_pipe.metrics.generate_summary()
    m2 = llm_pipe.metrics.generate_summary()
    reporter = Sprint5Reporter(output_dir="reports/scat_demo")
    reporter.generate_metrics_csv([m1, m2], filename="metrics.csv")
    reporter.generate_performance_charts([m1, m2], stress_results=[])


if __name__ == "__main__":
    main()

