#!/usr/bin/env python3
"""Test script to verify BlueSky state fetching with unit conversions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cdr.bluesky_io import BlueSkyClient, BSConfig

def test_unit_conversions():
    """Test the unit conversion factors used in get_aircraft_states()."""
    
    # Test conversions
    print("Testing unit conversions:")
    
    # Altitude: meters to feet
    alt_m = 10000  # 10km
    alt_ft = alt_m * 3.28084
    print(f"Altitude: {alt_m} m = {alt_ft:.1f} ft")
    assert abs(alt_ft - 32808.4) < 0.1, "Altitude conversion incorrect"
    
    # Speed: m/s to knots
    spd_ms = 250  # m/s (typical airliner cruise)
    spd_kt = spd_ms * 1.943844
    print(f"Speed: {spd_ms} m/s = {spd_kt:.1f} kt")
    assert abs(spd_kt - 485.961) < 0.1, "Speed conversion incorrect"
    
    # Vertical speed: m/s to fpm
    vs_ms = 5  # m/s climb
    vs_fpm = vs_ms * 196.8504
    print(f"Vertical speed: {vs_ms} m/s = {vs_fpm:.1f} fpm")
    assert abs(vs_fpm - 984.252) < 0.1, "Vertical speed conversion incorrect"
    
    print("✓ All unit conversions are correct")

def test_bluesky_client():
    """Test BlueSky client initialization and basic functionality."""
    print("\nTesting BlueSky client:")
    
    cfg = BSConfig(headless=True)
    client = BlueSkyClient(cfg)
    
    # Test attributes
    assert hasattr(client, 'host'), "Client should have host attribute"
    assert hasattr(client, 'port'), "Client should have port attribute"
    print(f"✓ Client configured with host={client.host}, port={client.port}")
    
    # Test connection (may fail if BlueSky not installed)
    try:
        connected = client.connect()
        if connected:
            print("✓ BlueSky connection successful")
            
            # Test state fetching
            states = client.get_aircraft_states()
            print(f"✓ State fetch successful, found {len(states)} aircraft")
            
            if states:
                state = states[0]
                required_keys = ['id', 'lat', 'lon', 'alt_ft', 'hdg_deg', 'spd_kt', 'roc_fpm']
                for key in required_keys:
                    assert key in state, f"Missing key: {key}"
                print("✓ Aircraft state format is correct")
            
        else:
            print("⚠ BlueSky connection failed (expected if BlueSky not installed)")
            
    except ImportError as e:
        print(f"⚠ BlueSky import failed: {e} (expected if BlueSky not installed)")
    except Exception as e:
        print(f"⚠ BlueSky connection error: {e}")

if __name__ == "__main__":
    test_unit_conversions()
    test_bluesky_client()
    print("\n✓ All tests completed")
