#!/usr/bin/env python3
"""
Enhanced conflict scenario creator that ensures a real close encounter.
"""

import json
import copy
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_close_encounter():
    """Create a guaranteed close encounter scenario"""
    
    # Load the original NAX3580 data
    original_file = Path("src/data/scat/100000.json")
    
    with open(original_file, 'r') as f:
        nax3580_data = json.load(f)
    
    print(f"Analyzing NAX3580 trajectory...")
    
    # Get middle section of NAX3580 flight for creating conflict
    total_plots = len(nax3580_data['plots'])
    middle_start = total_plots // 3
    middle_end = 2 * total_plots // 3
    
    print(f"Total plots: {total_plots}")
    print(f"Using middle section: {middle_start} to {middle_end}")
    
    # Get trajectory details from middle section
    middle_plot = nax3580_data['plots'][middle_start + 5]  # A bit past middle start
    target_lat = middle_plot['I062/105']['lat']
    target_lon = middle_plot['I062/105']['lon']
    target_alt = middle_plot['I062/136']['measured_flight_level']
    target_time = middle_plot['time_of_track']
    
    print(f"Conflict point: {target_lat:.4f}, {target_lon:.4f} at FL{target_alt} at {target_time}")
    
    # Create conflicting aircraft
    conflicting_data = copy.deepcopy(nax3580_data)
    conflicting_data['id'] = 100001
    conflicting_data['fpl']['fpl_base'][0]['callsign'] = 'CONFLICT1'
    conflicting_data['fpl']['fpl_base'][0]['aircraft_type'] = 'A320'
    conflicting_data['fpl']['fpl_base'][0]['adep'] = 'EGLL'  # London
    conflicting_data['fpl']['fpl_base'][0]['ades'] = 'EDDM'  # Munich
    
    # Create a trajectory that intersects NAX3580 at the target point
    # Calculate approach vector to intersection point (coming from different direction)
    approach_distance_nm = 20  # Approach from 20 nm away
    
    # Convert nm to degrees (rough approximation)
    nm_to_deg = 1.0 / 60.0  # 1 degree = 60 nm roughly
    approach_distance_deg = approach_distance_nm * nm_to_deg
    
    # Approach from northwest (opposite of NAX3580's southeast direction)
    approach_lat = target_lat + approach_distance_deg * 0.7
    approach_lon = target_lon - approach_distance_deg * 0.7
    
    # Exit to southeast 
    exit_lat = target_lat - approach_distance_deg * 0.7
    exit_lon = target_lon + approach_distance_deg * 0.7
    
    print(f"Creating crossing trajectory:")
    print(f"  From: {approach_lat:.4f}, {approach_lon:.4f}")
    print(f"  Through: {target_lat:.4f}, {target_lon:.4f}")
    print(f"  To: {exit_lat:.4f}, {exit_lon:.4f}")
    
    # Parse target time and create timeline
    target_dt = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
    
    # Create new trajectory with 20 points over 10 minutes (30 seconds apart)
    new_plots = []
    
    for i in range(20):
        # Linear interpolation between approach and exit points
        t_factor = i / 19.0  # 0 to 1
        
        lat = approach_lat + (exit_lat - approach_lat) * t_factor
        lon = approach_lon + (exit_lon - approach_lon) * t_factor
        
        # Time: start 5 minutes before target, intersect at target time
        time_offset_sec = -300 + (i * 30)  # -5 min to +5 min in 30 sec steps
        plot_time = target_dt + timedelta(seconds=time_offset_sec)
        
        # Altitude: similar to NAX3580 (slight variation)
        altitude = target_alt + (-10 + i)  # Slight altitude variation
        
        # Ground speed similar to NAX3580 but opposite direction
        speed_kt = 300
        heading = 135  # Southeast
        
        # Convert to velocity components (rough approximation)
        heading_rad = np.radians(heading)
        vx = speed_kt * np.sin(heading_rad) * 0.5  # Scale factor for BlueSky
        vy = speed_kt * np.cos(heading_rad) * 0.5
        
        plot = {
            "I062/105": {
                "lat": lat,
                "lon": lon
            },
            "I062/136": {
                "measured_flight_level": altitude
            },
            "I062/185": {
                "vx": vx,
                "vy": vy
            },
            "I062/200": {
                "adf": False,
                "long": 0,
                "trans": 0,
                "vert": 1
            },
            "I062/220": {
                "rocd": 0.0 if i == 10 else (100.0 if i < 10 else -100.0)
            },
            "I062/380": {
                "subitem3": {
                    "mag_hdg": heading
                },
                "subitem6": {
                    "altitude": int(altitude * 100),
                    "sas": False,
                    "source": 0
                },
                "subitem26": {
                    "ias": int(speed_kt * 0.85)  # IAS approximation
                },
                "subitem27": {
                    "mach": 0.75
                }
            },
            "time_of_track": plot_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '437'
        }
        new_plots.append(plot)
    
    # Replace the plots
    conflicting_data['plots'] = new_plots
    
    # Update predicted trajectory to match new flight path
    conflicting_data['predicted_trajectory'] = [{
        "route": [
            {
                "afl_unit": "F",
                "afl_value": target_alt,
                "eto": target_dt.strftime('%Y-%m-%dT%H:%M:%S'),
                "fix_kind": "CONFLICT",
                "fix_name": "INTERSECTION",
                "lat": target_lat,
                "lon": target_lon,
                "rfl_unit": "F",
                "rfl_value": target_alt,
                "rule": "I"
            }
        ],
        "time_stamp": target_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '000'
    }]
    
    # Save the conflicting aircraft
    conflict_file = Path("src/data/scat/100001.json")
    with open(conflict_file, 'w') as f:
        json.dump(conflicting_data, f, indent=2)
    
    print(f"\nClose encounter scenario created:")
    print(f"- NAX3580: Original trajectory, passes through conflict point at {target_time}")
    print(f"- CONFLICT1: Crossing trajectory, intersects at same time and place")
    print(f"- Intersection: {target_lat:.4f}, {target_lon:.4f} at FL{target_alt}")
    print(f"- Saved to: {conflict_file}")
    
    return conflict_file

def main():
    print("Creating guaranteed close encounter scenario...")
    print("=" * 60)
    
    create_close_encounter()
    
    print("\n" + "=" * 60)
    print("Close encounter scenario ready!")
    print("\nTo test conflict detection:")
    print("1. Run: atc-llm run-e2e --scat-path src/data/scat")
    print("2. Both geometric and LLM methods should detect the conflict")
    print("3. Check reports for comparison results")

if __name__ == "__main__":
    main()
