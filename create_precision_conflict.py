#!/usr/bin/env python3
"""
Create a precision conflict scenario using actual NAX3580 trajectory data.
This script creates CONFLICT2 that will intersect NAX3580's actual path very closely.
"""

import json
import datetime
import numpy as np
from typing import List, Dict, Any

def calculate_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in nautical miles."""
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # Earth radius in nautical miles
    R = 3440.065
    distance = R * c
    return distance

def create_precision_conflict():
    """Create CONFLICT2 that intersects NAX3580 at actual trajectory point."""
    
    # Use actual NAX3580 position from the JSONL output at intersection time
    # This is the exact point where NAX3580 will be at 11:41:52.695312
    target_time = "2016-10-20T11:41:52.695312"
    nax3580_lat = 55.177266
    nax3580_lon = 13.170005
    nax3580_alt_ft = 19600.0
    
    print(f"Creating CONFLICT2 to intersect NAX3580 at:")
    print(f"  Time: {target_time}")
    print(f"  Position: ({nax3580_lat:.6f}, {nax3580_lon:.6f})")
    print(f"  Altitude: {nax3580_alt_ft} ft")
    
    # Create CONFLICT2 trajectory that crosses this exact point
    # Start from northeast, heading southwest to cross NAX3580's path
    intersection_time = datetime.datetime.fromisoformat(target_time.replace('Z', '+00:00'))
    
    # Generate 20 position plots over 10 minutes (30-second intervals)
    plots = []
    num_plots = 20
    total_duration = datetime.timedelta(minutes=10)
    interval = total_duration / (num_plots - 1)
    
    # Flight starts 5 minutes before intersection
    start_time = intersection_time - datetime.timedelta(minutes=5)
    
    # Create a crossing path from northeast to southwest
    # Start point: northeast of intersection
    start_lat = nax3580_lat + 0.15  # ~9 NM north
    start_lon = nax3580_lon + 0.15  # ~9 NM east
    
    # End point: southwest of intersection  
    end_lat = nax3580_lat - 0.15  # ~9 NM south
    end_lon = nax3580_lon - 0.15  # ~9 NM west
    
    # Create altitude profile - level at FL196
    base_altitude = 19600.0
    
    print(f"CONFLICT2 path: ({start_lat:.6f}, {start_lon:.6f}) -> ({end_lat:.6f}, {end_lon:.6f})")
    
    for i in range(num_plots):
        # Linear interpolation for position
        t = i / (num_plots - 1)  # 0 to 1
        
        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)
        alt = base_altitude  # Constant altitude for simplicity
        
        # Calculate time for this plot
        plot_time = start_time + i * interval
        
        # Calculate speed and heading for this segment
        if i > 0:
            prev_lat = start_lat + (i-1)/(num_plots-1) * (end_lat - start_lat)
            prev_lon = start_lon + (i-1)/(num_plots-1) * (end_lon - start_lon)
            
            # Distance in nautical miles
            distance_nm = calculate_distance_nm(prev_lat, prev_lon, lat, lon)
            # Time interval in hours
            time_hours = interval.total_seconds() / 3600
            speed_kt = distance_nm / time_hours
            
            # Calculate heading (bearing)
            dlat = np.radians(lat - prev_lat)
            dlon = np.radians(lon - prev_lon)
            y = np.sin(dlon) * np.cos(np.radians(lat))
            x = (np.cos(np.radians(prev_lat)) * np.sin(np.radians(lat)) - 
                 np.sin(np.radians(prev_lat)) * np.cos(np.radians(lat)) * np.cos(dlon))
            heading = np.degrees(np.arctan2(y, x))
            if heading < 0:
                heading += 360
        else:
            speed_kt = 180.0  # Initial guess
            heading = 225.0   # Southwest heading
        
        # Check if this is the intersection plot
        is_intersection = abs((plot_time - intersection_time).total_seconds()) < 15
        
        if is_intersection:
            # Ensure exact intersection coordinates
            lat = nax3580_lat
            lon = nax3580_lon
            alt = nax3580_alt_ft
            print(f"Plot {i+1}: INTERSECTION POINT - ({lat:.6f}, {lon:.6f}) at {plot_time}")
        
        plot = {
            "I062/010": {
                "SAC": 254,
                "SIC": 1
            },
            "I062/015": {
                "SI": 254
            },
            "I062/105": {
                "LAT": lat,
                "LON": lon
            },
            "I062/136": {
                "FL": alt / 100  # Convert to flight level
            },
            "I062/185": {
                "VX": speed_kt * np.cos(np.radians(heading)) * 0.514444,  # Convert to m/s
                "VY": speed_kt * np.sin(np.radians(heading)) * 0.514444
            },
            "I062/220": {
                "ROCD": 0  # Level flight
            },
            "time_of_track": plot_time.isoformat()
        }
        
        plots.append(plot)
    
    # Create the SCAT JSON structure for CONFLICT2
    conflict_data = {
        "centre_ctrl": [
            {
                "centre_id": 1,
                "start_time": "2016-10-20T11:37:51.764000"
            }
        ],
        "fpl": {
            "fpl_arr": [
                {
                    "approach_clearance": False,
                    "arrival_runway": None,
                    "ata": None,
                    "missed_approach_flag": False,
                    "star": None,
                    "time_stamp": "2016-10-20T08:46:02.056000"
                }
            ],
            "fpl_base": [
                {
                    "adar": None,
                    "adep": "EHAM",
                    "ades": "LOWW", 
                    "aircraft_type": "B737",
                    "callsign": "CONFLICT2",
                    "equip_status_rvsm": True,
                    "flight_rules": "I",
                    "time_stamp": "2016-10-20T08:46:02.056000",
                    "wtc": "M"
                }
            ],
            "fpl_clearance": [
                {
                    "assign_heading_beacon": None,
                    "assigned_heading_val": None,
                    "assigned_speed_unit": None,
                    "assigned_speed_val": None,
                    "cfl": None,
                    "cfl_unit": None,
                    "time_stamp": "2016-10-20T08:46:02.056000"
                },
                {
                    "assign_heading_beacon": None,
                    "assigned_heading_val": None,
                    "assigned_speed_unit": None,
                    "assigned_speed_val": None,
                    "cfl": 196,
                    "cfl_unit": "F",
                    "time_stamp": "2016-10-20T11:31:05.006000"
                }
            ],
            "fpl_dep": [
                {
                    "departure_runway": None,
                    "sid": None,
                    "time_stamp": "2016-10-20T08:46:02.056000"
                }
            ],
            "fpl_route": [
                {
                    "navaid": "EHAM",
                    "route": "EHAM DCT LOWW",
                    "time_stamp": "2016-10-20T08:46:02.056000"
                }
            ]
        },
        "predicted_trajectory": {
            "aircr_id": "CONFLICT2",
            "points": [
                {
                    "lat": start_lat,
                    "lon": start_lon, 
                    "alt": base_altitude,
                    "time": start_time.isoformat()
                },
                {
                    "lat": nax3580_lat,
                    "lon": nax3580_lon,
                    "alt": nax3580_alt_ft,
                    "time": intersection_time.isoformat()
                },
                {
                    "lat": end_lat,
                    "lon": end_lon,
                    "alt": base_altitude,
                    "time": (start_time + total_duration).isoformat()
                }
            ]
        },
        "plots": plots
    }
    
    # Save to file
    output_file = "src/data/scat/100002.json"
    with open(output_file, 'w') as f:
        json.dump(conflict_data, f, indent=2)
    
    print(f"\nCONFLICT2 created with {len(plots)} plots")
    print(f"Saved to: {output_file}")
    
    # Verify intersection point
    intersection_plot = None
    for plot in plots:
        plot_time = datetime.datetime.fromisoformat(plot['time_of_track'])
        if abs((plot_time - intersection_time).total_seconds()) < 15:
            intersection_plot = plot
            break
    
    if intersection_plot:
        conflict_lat = intersection_plot['I062/105']['LAT']
        conflict_lon = intersection_plot['I062/105']['LON']
        conflict_alt = intersection_plot['I062/136']['FL'] * 100
        
        distance = calculate_distance_nm(nax3580_lat, nax3580_lon, conflict_lat, conflict_lon)
        alt_diff = abs(nax3580_alt_ft - conflict_alt)
        
        print(f"\nIntersection verification:")
        print(f"  NAX3580: ({nax3580_lat:.6f}, {nax3580_lon:.6f}, {nax3580_alt_ft} ft)")
        print(f"  CONFLICT2: ({conflict_lat:.6f}, {conflict_lon:.6f}, {conflict_alt} ft)")
        print(f"  Horizontal separation: {distance:.3f} NM")
        print(f"  Vertical separation: {alt_diff} ft")
        
        if distance < 0.1 and alt_diff < 100:
            print("  ✅ PERFECT INTERSECTION - Should trigger conflict detection!")
        else:
            print("  ⚠️  Intersection may not be close enough for conflict detection")
    
    return output_file

if __name__ == "__main__":
    create_precision_conflict()
