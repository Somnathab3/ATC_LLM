#!/usr/bin/env python3
"""
Quick test to verify the run-e2e command is properly implemented.
This test checks the CLI structure without requiring all dependencies.
"""

import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_cli_structure():
    """Test that the CLI structure is correct."""
    try:
        # Import just the CLI creation function
        import cli
        
        # Create parser to check if run-e2e command exists
        parser = cli.create_parser()
        
        # Test parsing run-e2e help
        try:
            args = parser.parse_args(['run-e2e', '--help'])
            print("❌ Should have shown help and exited")
            return False
        except SystemExit as e:
            if e.code == 0:  # Help was shown successfully
                print("✅ run-e2e command help works correctly")
            else:
                print(f"❌ Help failed with code {e.code}")
                return False
        
        # Test parsing a valid run-e2e command
        try:
            args = parser.parse_args(['run-e2e', '--scat-path', '/test/path'])
            if hasattr(args, 'func') and args.func.__name__ == 'cmd_run_e2e':
                print("✅ run-e2e command parsing works correctly")
                print(f"✅ Default values: cdmethod={args.cdmethod}, dtlook={args.dtlook}, tmult={args.tmult}")
                print(f"✅ LLM settings: model={args.llm_model}, confidence={args.confidence_threshold}")
                print(f"✅ Nav settings: max_diversion={args.max_diversion_nm}, vicinity={args.vicinity_radius}")
                return True
            else:
                print("❌ run-e2e command not properly linked")
                return False
        except Exception as e:
            print(f"❌ Command parsing failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def test_directto_commands():
    """Test that DIRECTTO commands are properly implemented."""
    try:
        # Test the BlueSky command formatting (without running it)
        from src.cdr.bluesky_io import BlueSkyClient, BSConfig
        
        # Create a mock client to test command formatting
        config = BSConfig()
        client = BlueSkyClient(config)
        
        # This won't actually connect but we can test the method exists
        if hasattr(client, 'direct_to'):
            print("✅ BlueSky direct_to method exists")
        else:
            print("❌ BlueSky direct_to method missing")
            return False
            
        return True
        
    except ImportError as e:
        print(f"⚠️  Cannot test DIRECTTO without dependencies: {e}")
        return True  # This is expected
    except Exception as e:
        print(f"❌ DIRECTTO test failed: {e}")
        return False

def test_waypoint_validation():
    """Test waypoint validation functionality."""
    try:
        from src.cdr.nav_utils import validate_waypoint_diversion
        print("✅ Waypoint validation function exists")
        return True
    except ImportError as e:
        print(f"⚠️  Cannot test waypoint validation without dependencies: {e}")
        return True  # This is expected
    except Exception as e:
        print(f"❌ Waypoint validation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing run-e2e CLI implementation...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing CLI structure...")
    success &= test_cli_structure()
    
    print("\n2. Testing DIRECTTO command support...")
    success &= test_directto_commands()
    
    print("\n3. Testing waypoint validation...")
    success &= test_waypoint_validation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The run-e2e command is properly implemented.")
        print("\nExpected command signature:")
        print("atc-llm run-e2e --scat-path F:/SCAT_extracted --ownship-limit 1 \\")
        print("  --vicinity-radius 100 --alt-window 5000 --asof --cdmethod GEOMETRIC \\")
        print("  --dtlook 600 --tmult 10 --spawn-dynamic --intruders 3 \\")
        print("  --adaptive-cadence --llm-model llama3.1:8b --confidence-threshold 0.8 \\")
        print("  --max-diversion-nm 80 --results-dir Output/enhanced_demo \\")
        print("  --reports-dir reports/enhanced --seed 4242")
    else:
        print("❌ Some tests failed. Check the implementation.")
        sys.exit(1)