#!/usr/bin/env python3
"""
Simple visualization of conflict detection for Sprint 2 demo.

This script creates a basic plot showing:
- Aircraft positions and trajectories  
- Detected conflicts
- CPA points and timings

Usage:
    python visualize_conflicts.py
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

# Simulate the core detection logic without imports
def simple_haversine_nm(lat1, lon1, lat2, lon2):
    """Simplified haversine distance in nautical miles."""
    R = 3440.065  # Earth radius in NM
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def simple_cpa(own_lat, own_lon, own_spd, own_hdg, 
               intr_lat, intr_lon, intr_spd, intr_hdg):
    """Simplified CPA calculation."""
    # Convert to relative coordinates (simplified)
    dx = (intr_lon - own_lon) * 60  # Rough conversion to NM
    dy = (intr_lat - own_lat) * 60
    
    # Velocity components (simplified)
    own_vx = own_spd * np.sin(np.radians(own_hdg)) / 60  # nm/min
    own_vy = own_spd * np.cos(np.radians(own_hdg)) / 60
    
    intr_vx = intr_spd * np.sin(np.radians(intr_hdg)) / 60
    intr_vy = intr_spd * np.cos(np.radians(intr_hdg)) / 60
    
    # Relative velocity
    dvx = own_vx - intr_vx
    dvy = own_vy - intr_vy
    
    # Time to CPA
    dv_squared = dvx**2 + dvy**2
    if dv_squared < 1e-6:
        tmin = 0
        dmin = np.sqrt(dx**2 + dy**2)
    else:
        tmin = max(0, -(dx * dvx + dy * dvy) / dv_squared)
        dx_cpa = dx + dvx * tmin
        dy_cpa = dy + dvy * tmin
        dmin = np.sqrt(dx_cpa**2 + dy_cpa**2)
    
    return dmin, tmin

def create_conflict_scenario():
    """Create a test scenario with aircraft."""
    
    # Ownship (flying east)
    ownship = {
        'id': 'OWNSHIP',
        'lat': 59.3,
        'lon': 18.1,
        'alt': 35000,
        'speed': 450,
        'heading': 90,
        'color': 'blue'
    }
    
    # Traffic aircraft
    traffic = [
        # Converging from northeast (potential conflict)
        {
            'id': 'TRF001',
            'lat': 59.4,
            'lon': 18.7,
            'alt': 35000,
            'speed': 450,
            'heading': 270,  # West (head-on)
            'color': 'red'
        },
        
        # Crossing from south (potential conflict)
        {
            'id': 'TRF002',
            'lat': 59.0,
            'lon': 18.3,
            'alt': 35000,
            'speed': 480,
            'heading': 0,  # North
            'color': 'orange'
        },
        
        # High altitude (no conflict)
        {
            'id': 'TRF003',
            'lat': 59.3,
            'lon': 18.2,
            'alt': 41000,
            'speed': 450,
            'heading': 90,
            'color': 'green'
        },
        
        # Diverging (no conflict)
        {
            'id': 'TRF004',
            'lat': 59.25,
            'lon': 18.0,
            'alt': 35000,
            'speed': 450,
            'heading': 180,  # South (diverging)
            'color': 'gray'
        }
    ]
    
    return ownship, traffic

def detect_conflicts(ownship, traffic):
    """Simple conflict detection."""
    conflicts = []
    
    for aircraft in traffic:
        # Filter by distance (100 NM)
        distance = simple_haversine_nm(
            ownship['lat'], ownship['lon'],
            aircraft['lat'], aircraft['lon']
        )
        
        if distance > 100:
            continue
            
        # Filter by altitude (±5000 ft)
        alt_diff = abs(ownship['alt'] - aircraft['alt'])
        if alt_diff > 5000:
            continue
            
        # Calculate CPA
        dmin, tmin = simple_cpa(
            ownship['lat'], ownship['lon'], ownship['speed'], ownship['heading'],
            aircraft['lat'], aircraft['lon'], aircraft['speed'], aircraft['heading']
        )
        
        # Check conflict criteria
        is_conflict = (dmin < 5.0 and alt_diff < 1000 and tmin <= 10.0 and tmin > 0)
        
        conflicts.append({
            'intruder': aircraft['id'],
            'distance_nm': distance,
            'dmin_nm': dmin,
            'tmin_min': tmin,
            'alt_diff_ft': alt_diff,
            'is_conflict': is_conflict,
            'aircraft': aircraft
        })
    
    return conflicts

def plot_scenario(ownship, traffic, conflicts):
    """Create visualization of the scenario."""
    
    plt.figure(figsize=(12, 8))
    
    # Plot aircraft positions
    plt.scatter(ownship['lon'], ownship['lat'], 
               s=200, c=ownship['color'], marker='^', 
               label=f"{ownship['id']} (Ownship)", zorder=5)
    
    for aircraft in traffic:
        plt.scatter(aircraft['lon'], aircraft['lat'],
                   s=150, c=aircraft['color'], marker='o',
                   label=f"{aircraft['id']} (FL{aircraft['alt']//100})", zorder=5)
    
    # Plot velocity vectors (scaled for visibility)
    scale = 0.01
    for aircraft in [ownship] + traffic:
        dx = scale * aircraft['speed'] * np.sin(np.radians(aircraft['heading']))
        dy = scale * aircraft['speed'] * np.cos(np.radians(aircraft['heading']))
        
        plt.arrow(aircraft['lon'], aircraft['lat'], dx, dy,
                 head_width=0.01, head_length=0.01, 
                 fc=aircraft['color'], ec=aircraft['color'], alpha=0.7)
    
    # Plot conflict predictions
    for conflict in conflicts:
        if conflict['is_conflict']:
            # Draw line between conflicting aircraft
            intr = conflict['aircraft']
            plt.plot([ownship['lon'], intr['lon']], 
                    [ownship['lat'], intr['lat']], 
                    'r--', linewidth=2, alpha=0.7, zorder=1)
            
            # Add conflict info
            mid_lon = (ownship['lon'] + intr['lon']) / 2
            mid_lat = (ownship['lat'] + intr['lat']) / 2
            plt.text(mid_lon, mid_lat, 
                    f"CONFLICT\n{conflict['tmin_min']:.1f} min\n{conflict['dmin_nm']:.1f} NM",
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                    fontsize=8, color='white', weight='bold')
    
    # Formatting
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title('Sprint 2: Conflict Detection Demonstration\n(Stockholm Airspace)', fontsize=14, weight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add text summary
    total_traffic = len(traffic)
    conflicts_detected = sum(1 for c in conflicts if c['is_conflict'])
    
    summary_text = f"""Detection Summary:
Total Traffic: {total_traffic}
Conflicts Detected: {conflicts_detected}
Lookahead: 10 minutes
Separation Standards: 5 NM / 1000 ft"""
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return conflicts_detected

def create_conflict_table(conflicts):
    """Create a detailed conflict analysis table."""
    
    print("\n" + "="*80)
    print("CONFLICT DETECTION ANALYSIS")
    print("="*80)
    print(f"{'Aircraft':<8} {'Distance':<10} {'CPA Dist':<10} {'CPA Time':<10} {'Alt Diff':<10} {'Conflict?':<10}")
    print("-"*80)
    
    for conflict in conflicts:
        aircraft_id = conflict['intruder']
        distance = conflict['distance_nm']
        dmin = conflict['dmin_nm']
        tmin = conflict['tmin_min']
        alt_diff = conflict['alt_diff_ft']
        is_conflict = "YES" if conflict['is_conflict'] else "NO"
        
        print(f"{aircraft_id:<8} {distance:<10.1f} {dmin:<10.1f} {tmin:<10.1f} {alt_diff:<10.0f} {is_conflict:<10}")
    
    print("-"*80)
    print(f"Total conflicts detected: {sum(1 for c in conflicts if c['is_conflict'])}")
    print("Separation standards: 5 NM horizontal, 1000 ft vertical")
    print("Lookahead time: 10 minutes")

def save_results(conflicts):
    """Save detection results to JSON."""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'detection_summary': {
            'total_aircraft_analyzed': len(conflicts),
            'conflicts_detected': sum(1 for c in conflicts if c['is_conflict']),
            'separation_standards': {
                'horizontal_nm': 5.0,
                'vertical_ft': 1000.0
            },
            'lookahead_minutes': 10.0
        },
        'conflicts': [
            {
                'intruder_id': c['intruder'],
                'current_distance_nm': round(c['distance_nm'], 2),
                'cpa_distance_nm': round(c['dmin_nm'], 2),
                'time_to_cpa_min': round(c['tmin_min'], 2),
                'altitude_diff_ft': c['alt_diff_ft'],
                'is_conflict': c['is_conflict']
            }
            for c in conflicts
        ]
    }
    
    with open('conflict_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'conflict_detection_results.json'")

def main():
    """Run the conflict detection visualization."""
    
    print("Sprint 2: Conflict Detection Demonstration")
    print("=========================================")
    
    # Create scenario
    ownship, traffic = create_conflict_scenario()
    
    # Detect conflicts
    conflicts = detect_conflicts(ownship, traffic)
    
    # Create visualization
    conflicts_detected = plot_scenario(ownship, traffic, conflicts)
    
    # Show detailed analysis
    create_conflict_table(conflicts)
    
    # Save results
    save_results(conflicts)
    
    # Show plot
    plt.savefig('conflict_detection_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'conflict_detection_demo.png'")
    
    try:
        plt.show()
    except:
        print("Could not display plot (running headless)")
    
    print(f"\nSprint 2 demonstration completed!")
    print(f"Conflicts detected: {conflicts_detected}")

if __name__ == "__main__":
    main()
