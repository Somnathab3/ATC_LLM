#!/usr/bin/env python3
"""
Fix CONFLICT4 to match the exact SCAT format as CONFLICT1.
"""

import json
import datetime

def fix_conflict4():
    """Fix CONFLICT4 JSON format to match working CONFLICT1."""
    
    # Load the existing CONFLICT4
    with open('src/data/scat/100004.json', 'r') as f:
        conflict_data = json.load(f)
    
    # Load CONFLICT1 to see the exact format
    with open('src/data/scat/100001.json', 'r') as f:
        reference_data = json.load(f)
    
    print("Fixing CONFLICT4 format to match CONFLICT1...")
    
    # Fix each plot to match CONFLICT1 format exactly
    for i, plot in enumerate(conflict_data['plots']):
        # Fix coordinate case (LAT/LON -> lat/lon)
        lat = plot['I062/105']['LAT']
        lon = plot['I062/105']['LON']
        plot['I062/105'] = {'lat': lat, 'lon': lon}
        
        # Remove timezone from time_of_track
        time_str = plot['time_of_track']
        if '+00:00' in time_str:
            time_str = time_str.replace('+00:00', '')
        plot['time_of_track'] = time_str
        
        # Remove extra fields that CONFLICT1 doesn't have
        if 'I062/010' in plot:
            del plot['I062/010']
        if 'I062/015' in plot:
            del plot['I062/015']
        
        # Add missing fields that CONFLICT1 has
        plot['I062/200'] = {"vx": 0.0, "vy": 0.0}  # Default velocity
        plot['I062/380'] = {"adr": "400001", "I062/380#4": {"cnf": 0, "rad": 0, "dou": 0, "mah": 0, "cdm": 0}}
        
        # Ensure I062/185 format matches (it should be vx, vy not VX, VY)
        if 'VX' in plot['I062/185']:
            vx = plot['I062/185']['VX']
            vy = plot['I062/185']['VY']
            plot['I062/185'] = {'vx': vx, 'vy': vy}
    
    # Save the fixed version
    with open('src/data/scat/100004.json', 'w') as f:
        json.dump(conflict_data, f, indent=2)
    
    print(f"Fixed CONFLICT4 with {len(conflict_data['plots'])} plots")
    print("Sample fixed plot keys:", list(conflict_data['plots'][0].keys()))
    print("Sample I062/105:", conflict_data['plots'][0]['I062/105'])
    print("Sample time_of_track:", conflict_data['plots'][0]['time_of_track'])
    
    return 'src/data/scat/100004.json'

if __name__ == "__main__":
    fix_conflict4()
