# scripts/repo_healthcheck.py
import json, os, sys, time, traceback
from pathlib import Path

from src.cdr.schemas import ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.scat_adapter import SCATAdapter
from src.cdr.pipeline import CDRPipeline
from src.cdr.reporting import Sprint5Reporter

SCAT_DIR = os.environ.get("SCAT_DIR", r"F:\SCAT_extracted")
SCAT_FILE = os.environ.get("SCAT_FILE", "100000.json")

def main():
    print("[1/5] SCAT parse")
    adapter = SCATAdapter(SCAT_DIR)
    rec = adapter.load_flight_record(Path(SCAT_DIR) / SCAT_FILE)
    states = adapter.extract_aircraft_states(rec)
    assert states, "No states from SCAT"

    own = states[0].model_copy(update={"aircraft_id":"OWNSHIP"})
    intr = [s.model_copy(update={"aircraft_id":f"INTRUDER_{i+1}"}) for i,s in enumerate(states[1:4])]
    print(f"   OK: states={len(states)} own={own.aircraft_id} intruders={len(intr)}")

    print("[2/5] BlueSky connect")
    cfg = ConfigurationSettings(fast_time=True, polling_interval_min=1.0, lookahead_time_min=5.0)
    bs = BlueSkyClient(cfg); assert bs.connect(), "BlueSky connect failed"

    bs.create_aircraft(own.aircraft_id, "A320", own.latitude, own.longitude, own.heading_deg, own.altitude_ft, own.ground_speed_kt)
    for t in intr:
        bs.create_aircraft(t.aircraft_id, "B738", t.latitude, t.longitude, t.heading_deg, t.altitude_ft, t.ground_speed_kt)
    bs.step_minutes(0.5)

    print("[3/5] Pipeline baseline run (LLM disabled)")
    base_cfg = cfg.model_copy(update={"llm_enabled": False})
    pipe = CDRPipeline(base_cfg); pipe.bluesky_client = bs
    pipe.run(max_cycles=2, ownship_id=own.aircraft_id)

    print("[4/5] Metrics check")
    m = pipe.metrics.generate_summary()
    outdir = Path("reports/healthcheck"); outdir.mkdir(parents=True, exist_ok=True)
    reporter = Sprint5Reporter(output_dir=str(outdir))
    reporter.generate_metrics_csv([m], filename="metrics.csv")
    assert (outdir / "metrics.csv").exists(), "metrics.csv missing"

    print("[5/5] Cleanup")
    try: bs.close()
    except: pass
    print("HEALTHCHECK OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(2)
