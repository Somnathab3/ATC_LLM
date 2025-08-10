#!/usr/bin/env python3
"""
Create a highly precise conflict scenario using exact NAX3580 trajectory coordinates.
This creates CONFLICT3 that will pass through the exact same point as NAX3580.
"""

import json
import datetime
import numpy as np

def create_exact_conflict():
    """Create CONFLICT3 that intersects NAX3580 at exact coordinates."""
    
    # Exact NAX3580 position at intersection time (from actual data)
    target_time = "2016-10-20T11:41:52.695312+00:00"
    exact_lat = 55.177267
    exact_lon = 13.170006
    exact_alt_ft = 19600.0
    
    print(f"Creating CONFLICT3 for EXACT intersection at:")
    print(f"  Time: {target_time}")
    print(f"  Exact Position: ({exact_lat:.6f}, {exact_lon:.6f})")
    print(f"  Exact Altitude: {exact_alt_ft} ft")
    
    intersection_time = datetime.datetime.fromisoformat(target_time)
    
    # Create trajectory that passes through the EXACT same point
    plots = []
    num_plots = 20
    total_duration = datetime.timedelta(minutes=10)
    interval = total_duration / (num_plots - 1)
    
    # Start 5 minutes before intersection
    start_time = intersection_time - datetime.timedelta(minutes=5)
    
    # Create perpendicular crossing - if NAX3580 is going southeast, we go northeast-southwest
    # Create a trajectory that crosses at exact 90 degrees
    offset = 0.08  # About 5 NM offset
    
    start_lat = exact_lat + offset
    start_lon = exact_lon - offset  # Northeast of intersection
    end_lat = exact_lat - offset
    end_lon = exact_lon + offset    # Southwest of intersection
    
    print(f"CONFLICT3 perpendicular crossing:")
    print(f"  From: ({start_lat:.6f}, {start_lon:.6f})")
    print(f"  Through: ({exact_lat:.6f}, {exact_lon:.6f}) <- EXACT MATCH")
    print(f"  To: ({end_lat:.6f}, {end_lon:.6f})")
    
    for i in range(num_plots):
        # Calculate time for this plot
        plot_time = start_time + i * interval
        
        # Check if this is the exact intersection plot
        time_to_intersection = (intersection_time - plot_time).total_seconds()
        is_intersection = abs(time_to_intersection) < 15  # Within 15 seconds
        
        if is_intersection:
            # Use EXACT coordinates for intersection
            lat = exact_lat
            lon = exact_lon
            alt = exact_alt_ft
            print(f"Plot {i+1}: EXACT INTERSECTION at {plot_time}")
        else:
            # Linear interpolation for other points
            t = i / (num_plots - 1)  # 0 to 1
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            alt = exact_alt_ft  # Same altitude level
        
        # Calculate speed and heading
        if i > 0:
            prev_t = (i-1) / (num_plots - 1)
            prev_lat = start_lat + prev_t * (end_lat - start_lat)
            prev_lon = start_lon + prev_t * (end_lon - start_lon)
            
            # Distance in nautical miles
            dlat = np.radians(lat - prev_lat)
            dlon = np.radians(lon - prev_lon)
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(prev_lat)) * np.cos(np.radians(lat)) * np.sin(dlon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance_nm = 3440.065 * c
            
            # Time interval in hours
            time_hours = interval.total_seconds() / 3600
            speed_kt = distance_nm / time_hours if time_hours > 0 else 180.0
            
            # Calculate heading
            y = np.sin(dlon) * np.cos(np.radians(lat))
            x = (np.cos(np.radians(prev_lat)) * np.sin(np.radians(lat)) - 
                 np.sin(np.radians(prev_lat)) * np.cos(np.radians(lat)) * np.cos(dlon))
            heading = np.degrees(np.arctan2(y, x))
            if heading < 0:
                heading += 360
        else:
            speed_kt = 200.0
            heading = 135.0  # Southeast
        
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
    
    # Create SCAT JSON structure
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
                    "adep": "ESSA",
                    "ades": "EDDF",
                    "aircraft_type": "A321",
                    "callsign": "CONFLICT3",
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
                    "navaid": "ESSA",
                    "route": "ESSA DCT EDDF",
                    "time_stamp": "2016-10-20T08:46:02.056000"
                }
            ]
        },
        "predicted_trajectory": {
            "aircr_id": "CONFLICT3",
            "points": [
                {
                    "lat": start_lat,
                    "lon": start_lon,
                    "alt": exact_alt_ft,
                    "time": start_time.isoformat()
                },
                {
                    "lat": exact_lat,
                    "lon": exact_lon,
                    "alt": exact_alt_ft,
                    "time": intersection_time.isoformat()
                },
                {
                    "lat": end_lat,
                    "lon": end_lon,
                    "alt": exact_alt_ft,
                    "time": (start_time + total_duration).isoformat()
                }
            ]
        },
        "plots": plots
    }
    
    # Save to file
    output_file = "src/data/scat/100003.json"
    with open(output_file, 'w') as f:
        json.dump(conflict_data, f, indent=2)
    
    print(f"\nCONFLICT3 created with {len(plots)} plots")
    print(f"Saved to: {output_file}")
    
    # Find and verify the intersection plot
    intersection_plot = None
    min_time_diff = float('inf')
    
    for i, plot in enumerate(plots):
        plot_time = datetime.datetime.fromisoformat(plot['time_of_track'])
        time_diff = abs((plot_time - intersection_time).total_seconds())
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            intersection_plot = (i, plot)
    
    if intersection_plot:
        i, plot = intersection_plot
        conflict_lat = plot['I062/105']['LAT']
        conflict_lon = plot['I062/105']['LON']
        conflict_alt = plot['I062/136']['FL'] * 100
        
        # Calculate separation
        dlat = np.radians(conflict_lat - exact_lat)
        dlon = np.radians(conflict_lon - exact_lon)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(exact_lat)) * np.cos(np.radians(conflict_lat)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance_nm = 3440.065 * c
        
        alt_diff = abs(exact_alt_ft - conflict_alt)
        
        print(f"\nIntersection verification (Plot {i+1}):")
        print(f"  NAX3580: ({exact_lat:.6f}, {exact_lon:.6f}, {exact_alt_ft} ft)")
        print(f"  CONFLICT3: ({conflict_lat:.6f}, {conflict_lon:.6f}, {conflict_alt} ft)")
        print(f"  Horizontal separation: {distance_nm:.6f} NM")
        print(f"  Vertical separation: {alt_diff} ft")
        print(f"  Time difference: {min_time_diff:.3f} seconds")
        
        if distance_nm < 0.001 and alt_diff < 1:
            print("  ✅ PERFECT EXACT INTERSECTION - GUARANTEED conflict detection!")
        elif distance_nm < 0.1 and alt_diff < 100:
            print("  ✅ VERY CLOSE - Should trigger conflict detection!")
        else:
            print("  ⚠️  May not be close enough for reliable detection")
    
    return output_file

if __name__ == "__main__":
    create_exact_conflict()
