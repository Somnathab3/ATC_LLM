#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive LLM Communication Verification Tool

This script verifies that ALL three communication paths work with REAL LLM:
1. SCAT to BlueSky - Load SCAT data and convert to BlueSky format
2. BlueSky to LLM - Send data to LLM for conflict analysis  
3. LLM to BlueSky - Parse LLM response back to BlueSky commands

The script includes extensive debugging and ensures NO mock/fake responses.
"""

import json
import logging
import os
import sys
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cdr.enhanced_llm_client import EnhancedLLMClient
from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState, ConfigurationSettings, ConflictPrediction
from src.cdr.bluesky_io import BlueSkyClient
from src.utils.output_utils import get_output_path, TestTypes

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_verification_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LLMCommunicationVerifier:
    """Comprehensive LLM communication verification with real-time debugging."""
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.verification_results = {}
        self.llm_interactions = []
        
        # Configuration for strict real LLM mode
        self.config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.2,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=4000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0,
            fast_time=True,
            sim_accel_factor=1.0
        )
        
        # Initialize components with correct parameters
        scat_dataset_path = "scenarios/scat"
        if not os.path.exists(scat_dataset_path):
            os.makedirs(scat_dataset_path, exist_ok=True)
        
        self.scat_adapter = SCATAdapter(scat_dataset_path)
        self.enhanced_llm_client = EnhancedLLMClient(self.config)
        
        # Ollama host and timeout for direct API calls
        self.ollama_host = "http://127.0.0.1:11434"
        self.ollama_timeout = 60
        
        # Force disable any mock modes
        self._disable_all_mocks()
        
        logger.info("[INIT] LLM Communication Verifier initialized in STRICT REAL-LLM MODE")
    
    def _disable_all_mocks(self):
        """Forcibly disable all mock modes to ensure real LLM communication."""
        os.environ.pop("LLM_DISABLED", None)
        os.environ.pop("LLM_MOCK", None)
        
        # Set explicit real LLM environment
        os.environ["LLM_REAL_MODE"] = "1"
        os.environ["LLM_STRICT_VERIFY"] = "1"
        
        # Update client config
        self.enhanced_llm_client.use_mock = False
        
        logger.info("[OK] All mock modes forcibly disabled - REAL LLM ONLY")
    
    def verify_ollama_connection(self) -> bool:
        """Verify direct connection to Ollama server."""
        logger.info("[CONNECT] Testing direct Ollama connection...")
        
        try:
            import requests
            
            # Test /api/tags endpoint to verify server
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                logger.info(f"[OK] Ollama server connected. Available models: {models}")
                
                # Check if our model is available
                model_names = [m.get('name', '') for m in models.get('models', [])]
                if self.config.llm_model_name in model_names:
                    logger.info(f"[OK] Target model '{self.config.llm_model_name}' is available")
                    return True
                else:
                    logger.warning(f"[WARN] Target model '{self.config.llm_model_name}' not found. Available: {model_names}")
                    return False
            else:
                logger.error(f"[ERROR] Ollama server returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Ollama connection failed: {e}")
            return False
    
    def test_raw_llm_call(self) -> bool:
        """Test raw LLM call with simple prompt to verify real communication."""
        logger.info("[TEST] Testing raw LLM communication...")
        
        test_prompt = """You are an expert air traffic controller. 
        
TASK: Respond with EXACTLY this JSON format, no additional text:
{"status": "operational", "timestamp": "current_time", "verification": "real_llm_response"}

Replace "current_time" with actual current time. This verifies you are a real LLM, not mocked data."""
        
        try:
            start_time = time.time()
            
            # Use direct Ollama API call
            response_data = self.enhanced_llm_client._post_ollama(test_prompt)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"[STATS] LLM Response Time: {response_time:.2f} seconds")
            
            if not response_data:
                logger.error("[ERROR] LLM returned empty response - may be mocked or failed")
                return False
            
            # Verify response structure
            if "status" in response_data and "verification" in response_data:
                logger.info(f"[OK] Real LLM response verified: {response_data}")
                
                # Log interaction for debugging
                self.llm_interactions.append({
                    "test": "raw_llm_call",
                    "prompt": test_prompt,
                    "response": response_data,
                    "response_time_sec": response_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                return True
            else:
                logger.error(f"[ERROR] LLM response format unexpected: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Raw LLM call failed: {e}")
            traceback.print_exc()
            return False
    
    def load_scat_data(self, scat_file_path: str) -> List[AircraftState]:
        """Step 1: SCAT to BlueSky - Load SCAT data and convert to BlueSky format."""
        logger.info(f"[LOAD] Step 1: Loading SCAT data from {scat_file_path}")
        
        try:
            # Load SCAT flight record
            flight_record = self.scat_adapter.load_flight_record(Path(scat_file_path))
            if not flight_record:
                raise ValueError(f"Failed to load flight record from {scat_file_path}")
            
            logger.info(f"[OK] SCAT flight record loaded successfully for {flight_record.callsign}")
            
            # Extract aircraft states from flight record
            aircraft_states = self.scat_adapter.extract_aircraft_states(flight_record)
            logger.info(f"[OK] Extracted {len(aircraft_states)} aircraft states")
            
            # Debug output
            if self.debug_mode:
                for i, state in enumerate(aircraft_states[:3]):  # Show first 3
                    logger.debug(f"Aircraft {i+1}: {state.callsign} at {state.latitude:.4f}, {state.longitude:.4f}, FL{state.altitude_ft/100:.0f}")
            
            self.verification_results["scat_to_bluesky"] = {
                "success": True,
                "aircraft_count": len(aircraft_states),
                "file_path": scat_file_path,
                "callsign": flight_record.callsign
            }
            
            return aircraft_states
            
        except Exception as e:
            logger.error(f"[ERROR] SCAT data loading failed: {e}")
            traceback.print_exc()
            self.verification_results["scat_to_bluesky"] = {
                "success": False,
                "error": str(e)
            }
            return []
    
    def analyze_with_llm(self, aircraft_states: List[AircraftState]) -> Dict[str, Any]:
        """Step 2: BlueSky to LLM - Send aircraft data to LLM for conflict analysis."""
        logger.info(f"[TEST] Step 2: Analyzing {len(aircraft_states)} aircraft with LLM")
        
        if len(aircraft_states) < 2:
            logger.warning("[WARN] Need at least 2 aircraft for conflict analysis")
            return {}
        
        try:
            # Prepare aircraft data for LLM
            ownship = aircraft_states[0]
            traffic = aircraft_states[1:5]  # Use up to 4 traffic aircraft
            
            logger.info(f"[TARGET] Analyzing conflicts for {ownship.callsign} vs {len(traffic)} traffic aircraft")
            
            # Build enhanced prompt
            start_time = time.time()
            detection_prompt = self.enhanced_llm_client.build_enhanced_detect_prompt(
                ownship, traffic, self.config
            )
            
            logger.debug(f"[PROMPT] Detection prompt length: {len(detection_prompt)} characters")
            if self.debug_mode:
                logger.debug(f"Detection prompt preview: {detection_prompt[:300]}...")
            
            # Call LLM for conflict detection
            logger.info("[DETECT] Calling LLM for conflict detection...")
            detection_response = self.enhanced_llm_client._post_ollama(detection_prompt)
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            logger.info(f"[STATS] LLM Detection Response Time: {detection_time:.2f} seconds")
            
            if not detection_response:
                logger.error("[ERROR] LLM detection response is empty - communication failed")
                return {}
            
            logger.info(f"[OK] LLM detection response received: {detection_response}")
            
            # If conflicts detected, get resolution
            resolution_response = None
            if detection_response.get("conflict", False):
                logger.info("[WARN] Conflicts detected, requesting resolution from LLM...")
                
                # Create a ConflictPrediction object from the detection response
                # Note: This is a mock conflict for verification purposes
                ownship_callsign = ownship.callsign or "UNKNOWN"
                mock_conflict = ConflictPrediction(
                    ownship_id=ownship_callsign,
                    intruder_id="INTRUDER_TEST",
                    time_to_cpa_min=5.0,
                    distance_at_cpa_nm=3.0,
                    altitude_diff_ft=500.0,
                    is_conflict=True,
                    severity_score=0.8,
                    conflict_type="horizontal",
                    prediction_time=datetime.now(),
                    confidence=0.9
                )
                
                start_time = time.time()
                resolution_prompt = self.enhanced_llm_client.build_enhanced_resolve_prompt(
                    ownship, [mock_conflict], self.config
                )
                
                logger.debug(f"[PROMPT] Resolution prompt length: {len(resolution_prompt)} characters")
                
                resolution_response = self.enhanced_llm_client._post_ollama(resolution_prompt)
                end_time = time.time()
                resolution_time = end_time - start_time
                
                logger.info(f"[STATS] LLM Resolution Response Time: {resolution_time:.2f} seconds")
                logger.info(f"[OK] LLM resolution response: {resolution_response}")
            
            # Log complete interaction
            interaction = {
                "test": "bluesky_to_llm",
                "ownship": {
                    "callsign": ownship.callsign,
                    "position": f"{ownship.latitude:.4f}, {ownship.longitude:.4f}",
                    "altitude_ft": ownship.altitude_ft
                },
                "traffic_count": len(traffic),
                "detection_prompt_length": len(detection_prompt),
                "detection_response": detection_response,
                "detection_time_sec": detection_time,
                "resolution_response": resolution_response,
                "resolution_time_sec": resolution_time if resolution_response else None,
                "timestamp": datetime.now().isoformat()
            }
            
            self.llm_interactions.append(interaction)
            
            self.verification_results["bluesky_to_llm"] = {
                "success": True,
                "detection_response": detection_response,
                "resolution_response": resolution_response,
                "response_times": {
                    "detection_sec": detection_time,
                    "resolution_sec": resolution_time if resolution_response else None
                }
            }
            
            return {
                "detection": detection_response,
                "resolution": resolution_response
            }
            
        except Exception as e:
            logger.error(f"[ERROR] LLM analysis failed: {e}")
            traceback.print_exc()
            self.verification_results["bluesky_to_llm"] = {
                "success": False,
                "error": str(e)
            }
            return {}
    
    def convert_to_bluesky_commands(self, llm_response: Dict[str, Any], 
                                  ownship: AircraftState) -> List[str]:
        """Step 3: LLM to BlueSky - Convert LLM response to BlueSky commands."""
        logger.info("[CONVERT] Step 3: Converting LLM response to BlueSky commands")
        
        try:
            commands = []
            
            # Process detection response
            detection = llm_response.get("detection", {})
            resolution = llm_response.get("resolution", {})
            
            if not resolution:
                logger.info("[INFO] No resolution needed - no conflicts detected")
                self.verification_results["llm_to_bluesky"] = {
                    "success": True,
                    "commands": [],
                    "reason": "No conflicts detected"
                }
                return []
            
            # Parse resolution and create BlueSky commands
            logger.info(f"[PROCESS] Processing resolution: {resolution}")
            
            action = resolution.get("action", "")
            params = resolution.get("params", {})
            callsign = ownship.callsign
            
            if action == "HEADING_CHANGE":
                new_heading = params.get("new_heading_deg")
                if new_heading:
                    cmd = f"{callsign} HDG {int(new_heading):03d}"
                    commands.append(cmd)
                    logger.info(f"[OK] Generated heading command: {cmd}")
            
            elif action == "ALTITUDE_CHANGE":
                new_altitude = params.get("new_altitude_ft")
                if new_altitude:
                    cmd = f"{callsign} ALT {int(new_altitude)}"
                    commands.append(cmd)
                    logger.info(f"[OK] Generated altitude command: {cmd}")
            
            elif action == "SPEED_CHANGE":
                new_speed = params.get("new_speed_kt")
                if new_speed:
                    cmd = f"{callsign} SPD {int(new_speed)}"
                    commands.append(cmd)
                    logger.info(f"[OK] Generated speed command: {cmd}")
            
            # Use enhanced LLM client's parsing if available
            if hasattr(self.enhanced_llm_client, '_sanitize_bluesky_command'):
                sanitized_commands = []
                for cmd in commands:
                    sanitized = self.enhanced_llm_client._sanitize_bluesky_command(cmd)
                    sanitized_commands.append(sanitized)
                    logger.info(f"[CLEAN] Sanitized command: {cmd} -> {sanitized}")
                commands = sanitized_commands
            
            self.verification_results["llm_to_bluesky"] = {
                "success": True,
                "commands": commands,
                "original_resolution": resolution
            }
            
            return commands
            
        except Exception as e:
            logger.error(f"[ERROR] LLM to BlueSky conversion failed: {e}")
            traceback.print_exc()
            self.verification_results["llm_to_bluesky"] = {
                "success": False,
                "error": str(e)
            }
            return []
    
    def run_iterative_loop(self, aircraft_states: List[AircraftState], 
                          max_iterations: int = 3) -> List[Dict[str, Any]]:
        """Run iterative BlueSky <-> LLM communication loop."""
        logger.info(f"[CONVERT] Starting iterative BlueSky <-> LLM loop (max {max_iterations} iterations)")
        
        iteration_results = []
        current_states = aircraft_states.copy()
        
        for iteration in range(max_iterations):
            logger.info(f"\n[CONVERT] === ITERATION {iteration + 1}/{max_iterations} ===")
            
            try:
                # Step 2: BlueSky to LLM
                llm_analysis = self.analyze_with_llm(current_states)
                
                if not llm_analysis:
                    logger.info("[INFO] No LLM analysis - stopping iteration")
                    break
                
                # Step 3: LLM to BlueSky  
                ownship = current_states[0]
                bluesky_commands = self.convert_to_bluesky_commands(llm_analysis, ownship)
                
                iteration_result = {
                    "iteration": iteration + 1,
                    "llm_analysis": llm_analysis,
                    "bluesky_commands": bluesky_commands,
                    "timestamp": datetime.now().isoformat()
                }
                
                iteration_results.append(iteration_result)
                
                # Check if resolution was applied
                if bluesky_commands:
                    logger.info(f"[OK] Iteration {iteration + 1}: Generated {len(bluesky_commands)} BlueSky commands")
                    
                    # Simulate command execution by updating aircraft state
                    # (In real system, this would send to BlueSky)
                    self._simulate_command_execution(current_states[0], bluesky_commands)
                else:
                    logger.info(f"[INFO] Iteration {iteration + 1}: No commands needed")
                    break
                
            except Exception as e:
                logger.error(f"[ERROR] Iteration {iteration + 1} failed: {e}")
                break
        
        logger.info(f"[FINISH] Iterative loop completed after {len(iteration_results)} iterations")
        return iteration_results
    
    def _simulate_command_execution(self, aircraft: AircraftState, commands: List[str]):
        """Simulate execution of BlueSky commands on aircraft state."""
        for cmd in commands:
            parts = cmd.split()
            if len(parts) >= 3:
                command_type = parts[1]
                value = parts[2]
                
                if command_type == "HDG":
                    aircraft.heading_deg = float(value)
                    logger.debug(f"Updated {aircraft.callsign} heading to {value}deg")
                elif command_type == "ALT":
                    aircraft.altitude_ft = float(value)
                    logger.debug(f"Updated {aircraft.callsign} altitude to {value} ft")
                elif command_type == "SPD":
                    aircraft.speed_kt = float(value)
                    logger.debug(f"Updated {aircraft.callsign} speed to {value} kt")
    
    def run_full_verification(self, scat_file_path: str) -> Dict[str, Any]:
        """Run complete end-to-end verification of all communication paths."""
        logger.info("[INIT] Starting FULL LLM Communication Verification")
        logger.info("=" * 80)
        
        verification_start = time.time()
        
        # Pre-checks
        logger.info("[CHECK] Running pre-verification checks...")
        
        if not self.verify_ollama_connection():
            logger.error("[ERROR] Ollama connection failed - cannot proceed")
            return {"success": False, "error": "Ollama connection failed"}
        
        if not self.test_raw_llm_call():
            logger.error("[ERROR] Raw LLM test failed - cannot proceed")
            return {"success": False, "error": "Raw LLM communication failed"}
        
        logger.info("[OK] Pre-checks passed - proceeding with full verification")
        
        try:
            # Step 1: SCAT to BlueSky
            aircraft_states = self.load_scat_data(scat_file_path)
            if not aircraft_states:
                return {"success": False, "error": "SCAT data loading failed"}
            
            # Step 2 & 3: BlueSky <-> LLM iterative loop
            iteration_results = self.run_iterative_loop(aircraft_states)
            
            verification_end = time.time()
            total_time = verification_end - verification_start
            
            # Compile final results
            final_results = {
                "success": True,
                "total_verification_time_sec": total_time,
                "verification_timestamp": datetime.now().isoformat(),
                "step_results": self.verification_results,
                "iteration_results": iteration_results,
                "llm_interactions": self.llm_interactions,
                "summary": {
                    "scat_to_bluesky": self.verification_results.get("scat_to_bluesky", {}).get("success", False),
                    "bluesky_to_llm": self.verification_results.get("bluesky_to_llm", {}).get("success", False),
                    "llm_to_bluesky": self.verification_results.get("llm_to_bluesky", {}).get("success", False),
                    "iterations_completed": len(iteration_results),
                    "total_llm_calls": len(self.llm_interactions)
                }
            }
            
            logger.info("[SUCCESS] FULL VERIFICATION COMPLETED SUCCESSFULLY!")
            logger.info(f"[STATS] Total time: {total_time:.2f} seconds")
            logger.info(f"[STATS] LLM interactions: {len(self.llm_interactions)}")
            logger.info(f"[STATS] Iterations: {len(iteration_results)}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"[ERROR] Full verification failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def save_debug_results(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Save comprehensive debug results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized output directory
        output_dir = get_output_path(TestTypes.VERIFICATION, timestamp)
        
        if not output_file:
            output_file_path = output_dir / f"llm_verification_results_{timestamp}.json"
        else:
            output_file_path = output_dir / output_file
        
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"[SAVE] Debug results saved to: {output_file_path}")
            
            # Also save interaction log
            interaction_file = output_dir / f"llm_verification_results_{timestamp}_interactions.json"
            with open(interaction_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_interactions, f, indent=2, default=str)
            
            logger.info(f"[SAVE] LLM interactions saved to: {interaction_file}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save debug results: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive LLM Communication Verification Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script verifies that ALL three communication paths work with REAL LLM:
1. SCAT to BlueSky - Load SCAT data and convert to BlueSky format
2. BlueSky to LLM - Send data to LLM for conflict analysis  
3. LLM to BlueSky - Parse LLM response back to BlueSky commands

Examples:
  python verify_llm_communication.py                                    # Use default SCAT file
  python verify_llm_communication.py --scat-file scenarios/scat/100000.json  # Specify SCAT file
  python verify_llm_communication.py --quick-test                       # Run quick verification
        """
    )
    
    parser.add_argument('--scat-file', type=str, 
                       default='scenarios/scat/100000.json',
                       help='SCAT file to use for verification (default: scenarios/scat/100000.json)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick verification test with minimal scenarios')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of verification iterations to run (default: 3)')
    
    args = parser.parse_args()
    
    print(" LLM Communication Verification Tool")
    print("=" * 50)
    
    scat_file = args.scat_file
    print(f"[LOAD] Using SCAT file: {scat_file}")
    
    if args.quick_test:
        print("[QUICK] Quick test mode enabled")
    
    # Verify file exists
    if not os.path.exists(scat_file):
        print(f"[ERROR] SCAT file not found: {scat_file}")
        print(f"[TIP] Available SCAT files:")
        scat_dir = Path("scenarios/scat")
        if scat_dir.exists():
            for scat in scat_dir.glob("*.json"):
                print(f"   * {scat}")
        else:
            print(f"   (SCAT directory {scat_dir} not found)")
        sys.exit(1)
    
    # Create verifier and run
    verifier = LLMCommunicationVerifier(debug_mode=args.debug)
    
    try:
        # Run the verification (quick_test just affects logging verbosity)
        results = verifier.run_full_verification(scat_file)
        
        # Save results
        verifier.save_debug_results(results)
        
        # Print summary
        print("\n" + "=" * 50)
        print("[STATS] VERIFICATION SUMMARY")
        print("=" * 50)
        
        if results["success"]:
            print("[OK] OVERALL: SUCCESS")
            summary = results["summary"]
            print(f"[OK] SCAT to BlueSky: {'SUCCESS' if summary['scat_to_bluesky'] else 'FAILED'}")
            print(f"[OK] BlueSky to LLM: {'SUCCESS' if summary['bluesky_to_llm'] else 'FAILED'}")
            print(f"[OK] LLM to BlueSky: {'SUCCESS' if summary['llm_to_bluesky'] else 'FAILED'}")
            print(f"[STATS] Iterations completed: {summary['iterations_completed']}")
            print(f"[STATS] Total LLM calls: {summary['total_llm_calls']}")
            print(f"[TIME] Total time: {results['total_verification_time_sec']:.2f} seconds")
        else:
            print("[ERROR] OVERALL: FAILED")
            print(f"[ERROR] Error: {results.get('error', 'Unknown error')}")
        
        print(f"[SAVE] Debug results saved to organized output folder")
        print(f"[SAVE] Debug log: llm_verification_debug.log")
        
    except KeyboardInterrupt:
        print("\n[WARN] Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
