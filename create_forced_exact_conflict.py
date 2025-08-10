#!/usr/bin/env python3
"""
Create CONFLICT4 with forced exact intersection at NAX3580 coordinates.
This version explicitly places one plot at the exact same coordinates and time.
"""

import json
import datetime
import numpy as np

def create_forced_exact_conflict():
    """Create CONFLICT4 with one plot forced to exact NAX3580 coordinates."""
    
    # Exact intersection details
    target_time_str = "2016-10-20T11:41:52.695312+00:00"
    exact_lat = 55.177267
    exact_lon = 13.170006  
    exact_alt_ft = 19600.0
    
    print(f"Creating CONFLICT4 with FORCED exact intersection:")
    print(f"  Time: {target_time_str}")
    print(f"  Coordinates: ({exact_lat:.6f}, {exact_lon:.6f})")
    print(f"  Altitude: {exact_alt_ft} ft")
    
    intersection_time = datetime.datetime.fromisoformat(target_time_str)
    
    # Create 19 other plots around the intersection
    plots = []
    
    # Create trajectory: approach from northwest, pass through exact point, continue southeast
    # Total flight time: 10 minutes, with intersection in the middle
    total_duration = datetime.timedelta(minutes=10)
    
    # Start 5 minutes before intersection
    start_time = intersection_time - datetime.timedelta(minutes=5)
    
    # Define start and end points for the trajectory
    offset = 0.05  # About 3 NM offset
    start_lat = exact_lat + offset
    start_lon = exact_lon - offset  # Northwest
    end_lat = exact_lat - offset  
    end_lon = exact_lon + offset    # Southeast
    
    print(f"Trajectory: ({start_lat:.6f}, {start_lon:.6f}) -> ({exact_lat:.6f}, {exact_lon:.6f}) -> ({end_lat:.6f}, {end_lon:.6f})")
    
    # Create 20 plots total
    num_plots = 20
    intersection_plot_index = 10  # Middle plot will be the exact intersection
    
    for i in range(num_plots):
        plot_time = start_time + i * (total_duration / (num_plots - 1))
        
        if i == intersection_plot_index:
            # FORCE exact intersection coordinates and time
            lat = exact_lat
            lon = exact_lon
            alt = exact_alt_ft
            plot_time = intersection_time  # Force exact time too
            print(f"Plot {i+1}: FORCED EXACT INTERSECTION at {plot_time}")
        else:
            # Linear interpolation for other plots
            if i < intersection_plot_index:
                # Before intersection: interpolate from start to intersection
                t = i / intersection_plot_index
                lat = start_lat + t * (exact_lat - start_lat)
                lon = start_lon + t * (exact_lon - start_lon)
            else:
                # After intersection: interpolate from intersection to end
                t = (i - intersection_plot_index) / (num_plots - 1 - intersection_plot_index)
                lat = exact_lat + t * (end_lat - exact_lat)
                lon = exact_lon + t * (end_lon - exact_lon)
            
            alt = exact_alt_ft  # Same altitude throughout
        
        # Calculate realistic speed and heading
        if i > 0:
            # Get previous position
            if i-1 == intersection_plot_index:
                prev_lat, prev_lon = exact_lat, exact_lon
            elif i-1 < intersection_plot_index:
                t_prev = (i-1) / intersection_plot_index
                prev_lat = start_lat + t_prev * (exact_lat - start_lat)
                prev_lon = start_lon + t_prev * (exact_lon - start_lon)
            else:
                t_prev = (i-1 - intersection_plot_index) / (num_plots - 1 - intersection_plot_index)
                prev_lat = exact_lat + t_prev * (end_lat - exact_lat)
                prev_lon = exact_lon + t_prev * (end_lon - exact_lon)
            
            # Calculate distance and speed
            dlat = np.radians(lat - prev_lat)
            dlon = np.radians(lon - prev_lon)
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(prev_lat)) * np.cos(np.radians(lat)) * np.sin(dlon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance_nm = 3440.065 * c
            
            # Time since previous plot
            if i == intersection_plot_index:
                prev_time = start_time + (i-1) * (total_duration / (num_plots - 1))
            elif i-1 == intersection_plot_index:
                prev_time = intersection_time
            else:
                prev_time = start_time + (i-1) * (total_duration / (num_plots - 1))
            
            time_hours = (plot_time - prev_time).total_seconds() / 3600
            speed_kt = distance_nm / time_hours if time_hours > 0 else 200.0
            
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
                "FL": alt / 100
            },
            "I062/185": {
                "VX": speed_kt * np.cos(np.radians(heading)) * 0.514444,
                "VY": speed_kt * np.sin(np.radians(heading)) * 0.514444
            },
            "I062/220": {
                "ROCD": 0
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
                    "adep": "ENGM",
                    "ades": "LIRF",
                    "aircraft_type": "B738",
                    "callsign": "CONFLICT4",
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
                    "navaid": "ENGM",
                    "route": "ENGM DCT LIRF",
                    "time_stamp": "2016-10-20T08:46:02.056000"
                }
            ]
        },
        "predicted_trajectory": {
            "aircr_id": "CONFLICT4",
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
    output_file = "src/data/scat/100004.json"
    with open(output_file, 'w') as f:
        json.dump(conflict_data, f, indent=2)
    
    print(f"\nCONFLICT4 created with {len(plots)} plots")
    print(f"Saved to: {output_file}")
    
    # Verify the exact intersection plot
    intersection_plot = plots[intersection_plot_index]
    conflict_lat = intersection_plot['I062/105']['LAT']
    conflict_lon = intersection_plot['I062/105']['LON']
    conflict_alt = intersection_plot['I062/136']['FL'] * 100
    conflict_time = intersection_plot['time_of_track']
    
    print(f"\nIntersection verification (Plot {intersection_plot_index+1}):")
    print(f"  NAX3580: ({exact_lat:.6f}, {exact_lon:.6f}, {exact_alt_ft} ft) at {target_time_str}")
    print(f"  CONFLICT4: ({conflict_lat:.6f}, {conflict_lon:.6f}, {conflict_alt} ft) at {conflict_time}")
    
    # Should be exactly zero separation
    distance_nm = 0.0
    alt_diff = abs(exact_alt_ft - conflict_alt)
    
    print(f"  Horizontal separation: {distance_nm:.6f} NM")
    print(f"  Vertical separation: {alt_diff} ft")
    
    if distance_nm == 0.0 and alt_diff == 0.0:
        print("  ✅ PERFECT ZERO SEPARATION - GUARANTEED conflict detection!")
    else:
        print("  ⚠️  Unexpected separation")
    
    return output_file

if __name__ == "__main__":
    create_forced_exact_conflict()
