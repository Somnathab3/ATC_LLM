#!/usr/bin/env python3
"""
Script to add an artificial conflict to SCAT data for testing conflict detection methods.
Creates a second aircraft that will conflict with NAX3580 (100000.json).
"""

import json
import copy
from datetime import datetime, timedelta
from pathlib import Path

def create_conflicting_aircraft():
    """Create a conflicting aircraft based on NAX3580 data"""
    
    # Load the original NAX3580 data
    original_file = Path("src/data/scat/100000.json")
    
    with open(original_file, 'r') as f:
        nax3580_data = json.load(f)
    
    print(f"Original NAX3580 data loaded:")
    print(f"- ID: {nax3580_data['id']}")
    print(f"- Aircraft type: {nax3580_data['fpl']['fpl_base'][0]['aircraft_type']}")
    print(f"- Callsign: {nax3580_data['fpl']['fpl_base'][0]['callsign']}")
    print(f"- Number of plots: {len(nax3580_data['plots'])}")
    
    # Create conflicting aircraft by copying and modifying NAX3580
    conflicting_data = copy.deepcopy(nax3580_data)
    
    # Change basic identifiers
    conflicting_data['id'] = 100001
    conflicting_data['fpl']['fpl_base'][0]['callsign'] = 'SAS4321'  # Different callsign
    conflicting_data['fpl']['fpl_base'][0]['aircraft_type'] = 'B737'  # Different aircraft type
    
    # Modify the flight plan to create a crossing path
    # Change departure/arrival to create conflict scenario
    conflicting_data['fpl']['fpl_base'][0]['adep'] = 'LKPR'  # Prague (reverse direction)
    conflicting_data['fpl']['fpl_base'][0]['ades'] = 'EKCH'  # Copenhagen
    
    # Modify position plots to create conflict
    # We'll offset the aircraft slightly and adjust timing to create intersection
    lat_offset = 0.05  # Small latitude offset
    lon_offset = -0.02  # Small longitude offset  
    time_offset = timedelta(minutes=5)  # 5 minute time offset
    
    for plot in conflicting_data['plots']:
        # Adjust position to create crossing trajectory
        plot['I062/105']['lat'] += lat_offset
        plot['I062/105']['lon'] += lon_offset
        
        # Adjust velocity to create crossing pattern
        plot['I062/185']['vx'] = -plot['I062/185']['vx']  # Reverse x velocity
        plot['I062/185']['vy'] = plot['I062/185']['vy'] * 0.8  # Slightly different y velocity
        
        # Adjust timing
        original_time = datetime.fromisoformat(plot['time_of_track'].replace('Z', '+00:00'))
        new_time = original_time + time_offset
        plot['time_of_track'] = new_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '37'
    
    # Adjust predicted trajectory
    for trajectory in conflicting_data['predicted_trajectory']:
        for route_point in trajectory['route']:
            route_point['lat'] += lat_offset
            route_point['lon'] += lon_offset
        
        # Adjust trajectory timestamp
        original_time = datetime.fromisoformat(trajectory['time_stamp'].replace('Z', '+00:00'))
        new_time = original_time + time_offset
        trajectory['time_stamp'] = new_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '000'
    
    # Save the conflicting aircraft data
    conflict_file = Path("src/data/scat/100001.json")
    with open(conflict_file, 'w') as f:
        json.dump(conflicting_data, f, indent=2)
    
    print(f"\nConflicting aircraft created:")
    print(f"- ID: {conflicting_data['id']}")
    print(f"- Callsign: {conflicting_data['fpl']['fpl_base'][0]['callsign']}")
    print(f"- Aircraft type: {conflicting_data['fpl']['fpl_base'][0]['aircraft_type']}")
    print(f"- Route: {conflicting_data['fpl']['fpl_base'][0]['adep']} -> {conflicting_data['fpl']['fpl_base'][0]['ades']}")
    print(f"- Saved to: {conflict_file}")
    
    # Show conflict details
    print(f"\nConflict scenario:")
    print(f"- NAX3580: B738, EKCH->LKPR, climbing through FL250")
    print(f"- SAS4321: B737, LKPR->EKCH, similar altitude, crossing path")
    print(f"- Time separation: 5 minutes")
    print(f"- Position offset: {lat_offset}° lat, {lon_offset}° lon")
    
    return conflict_file

def verify_conflict_data():
    """Verify the conflict data looks reasonable"""
    
    files = [
        Path("src/data/scat/100000.json"),  # Original NAX3580
        Path("src/data/scat/100001.json")   # New SAS4321
    ]
    
    print("\nVerifying conflict scenario:")
    
    for file_path in files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get first and last positions
        first_plot = data['plots'][0]
        last_plot = data['plots'][-1]
        
        print(f"\n✅ {data['fpl']['fpl_base'][0]['callsign']} ({data['fpl']['fpl_base'][0]['aircraft_type']}):")
        print(f"   Route: {data['fpl']['fpl_base'][0]['adep']} -> {data['fpl']['fpl_base'][0]['ades']}")
        print(f"   Start: {first_plot['I062/105']['lat']:.3f}, {first_plot['I062/105']['lon']:.3f} at FL{first_plot['I062/136']['measured_flight_level']}")
        print(f"   End:   {last_plot['I062/105']['lat']:.3f}, {last_plot['I062/105']['lon']:.3f} at FL{last_plot['I062/136']['measured_flight_level']}")
        print(f"   Time:  {first_plot['time_of_track']} to {last_plot['time_of_track']}")

def main():
    """Main function to create conflict test scenario"""
    
    print("Creating artificial conflict scenario for testing...")
    print("=" * 60)
    
    # Create the conflicting aircraft
    create_conflicting_aircraft()
    
    # Verify the data
    verify_conflict_data()
    
    print("\n" + "=" * 60)
    print("Conflict scenario ready!")
    print("\nTo test both detection methods:")
    print("1. Run: atc-llm run-e2e")
    print("2. Compare results from both geometric and LLM-based detection")
    print("3. Both methods should detect the same conflict between NAX3580 and SAS4321")

if __name__ == "__main__":
    main()
