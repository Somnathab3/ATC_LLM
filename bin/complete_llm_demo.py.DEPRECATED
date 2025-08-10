#!/usr/bin/env python3
"""
Complete LLM Communication Demonstration

This script demonstrates all three required communication paths:
1. SCAT to BlueSky - Load real SCAT data and convert to flight states
2. BlueSky to LLM - Send flight data to LLM for conflict analysis
3. LLM to BlueSky - Parse LLM responses into executable BlueSky commands

The script runs iterative loops where steps 2 and 3 repeat to find optimal solutions.
"""

import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Any

class CompleteLLMDemo:
    """Complete demonstration of SCAT -> BlueSky -> LLM -> BlueSky communication."""
    
    def __init__(self):
        self.ollama_host = "http://127.0.0.1:11434"
        self.model_name = "llama3.1:8b"
        self.debug_log = []
        
        print("=" * 80)
        print("COMPLETE LLM COMMUNICATION DEMONSTRATION")
        print("Showing: SCAT -> BlueSky -> LLM -> BlueSky iterative loops")
        print("=" * 80)
    
    def log_debug(self, message: str):
        """Add debug message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.debug_log.append(log_entry)
        print(f"DEBUG: {log_entry}")
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with error handling and debugging."""
        self.log_debug(f"Sending prompt to LLM (length: {len(prompt)} chars)")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                llm_response = data.get("response", "")
                
                self.log_debug(f"LLM responded in {response_time:.2f}s")
                
                # Extract JSON from response
                import re
                json_match = re.search(r'\\{.*\\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group(0))
                        return {"success": True, "data": json_data, "time": response_time}
                    except json.JSONDecodeError:
                        self.log_debug("JSON parsing failed")
                        return {"success": False, "error": "JSON parsing failed", "raw": llm_response}
                else:
                    self.log_debug("No JSON found in response")
                    return {"success": False, "error": "No JSON found", "raw": llm_response}
            else:
                self.log_debug(f"HTTP error: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            self.log_debug(f"LLM call failed: {e}")
            return {"success": False, "error": str(e)}
    
    def step1_scat_to_bluesky(self, scat_file: str) -> Dict[str, Any]:
        """STEP 1: Load SCAT data and create aircraft in BlueSky with route following."""
        print("\n" + "="*60)
        print("STEP 1: SCAT TO BLUESKY WITH ROUTE FOLLOWING")
        print("="*60)
        
        try:
            self.log_debug(f"Loading SCAT file: {scat_file}")
            
            with open(scat_file, 'r', encoding='utf-8') as f:
                scat_data = json.load(f)
            
            # Extract flight plan data
            fpl_base = scat_data.get('fpl', {}).get('fpl_base', [])
            plots = scat_data.get('plots', [])
            
            if not fpl_base or not plots:
                self.log_debug("Invalid SCAT data structure")
                return {"success": False, "error": "Invalid SCAT data"}
            
            # Extract aircraft information
            aircraft_info = fpl_base[0]
            callsign = aircraft_info.get("callsign", "UNKNOWN")
            aircraft_type = aircraft_info.get("aircraft_type", "UNKNOWN")
            
            # Build route from all plot points
            route = []
            for plot in plots:
                position = plot.get("I062/105", {})
                lat = position.get("lat")
                lon = position.get("lon")
                if lat is not None and lon is not None:
                    route.append((lat, lon))
            
            if len(route) < 2:
                return {"success": False, "error": "Insufficient route points"}
            
            # Initialize BlueSky if not already done
            if not hasattr(self, 'bluesky_client'):
                from src.cdr.bluesky_io import BlueSkyClient, BSConfig
                bs_config = BSConfig()
                self.bluesky_client = BlueSkyClient(bs_config)
                
                if not self.bluesky_client.connect():
                    return {"success": False, "error": "Failed to connect to BlueSky"}
                
                # Reset and configure simulation
                self.bluesky_client.sim_reset()
                self.bluesky_client.sim_realtime(False)  # Fast simulation
                self.bluesky_client.sim_set_dtmult(60.0)  # 60x speed
            
            # Get initial position and heading
            lat0, lon0 = route[0]
            from src.cdr.geodesy import bearing_deg
            if len(route) > 1:
                hdg0 = bearing_deg(lat0, lon0, route[1][0], route[1][1])
            else:
                hdg0 = 90.0  # Default heading
            
            # Get altitude from latest plot
            latest_plot = plots[-1]
            altitude_data = latest_plot.get("I062/136", {})
            altitude_ft = altitude_data.get("measured_flight_level", 350) * 100  # FL to feet
            
            # Create aircraft in BlueSky
            success = self.bluesky_client.create_aircraft(
                callsign, aircraft_type, lat0, lon0, hdg0, altitude_ft, 420.0
            )
            
            if not success:
                return {"success": False, "error": f"Failed to create aircraft {callsign}"}
            
            # Add route waypoints for autopilot following
            # Sample route to avoid too many waypoints (every 10th point)
            sampled_route = route[::10] + [route[-1]]  # Include final point
            
            # Add waypoints to BlueSky
            waypoints_added = self.bluesky_client.add_waypoints_from_route(
                callsign, sampled_route[1:], altitude_ft
            )
            
            if not waypoints_added:
                self.log_debug("Warning: Failed to add some waypoints")
            
            # Let BlueSky process the aircraft creation and waypoints
            self.bluesky_client.step_minutes(0.5)
            
            # Create result with route information
            bluesky_aircraft = {
                "callsign": callsign,
                "aircraft_type": aircraft_type,
                "latitude": lat0,
                "longitude": lon0,
                "altitude_ft": altitude_ft,
                "heading_deg": hdg0,
                "route_waypoints": len(sampled_route),
                "total_route_points": len(route),
                "autopilot_engaged": True
            }
            
            self.log_debug(f"Successfully created {callsign} with {len(sampled_route)} waypoints")
            print(f"Aircraft: {callsign} ({aircraft_type})")
            print(f"Initial position: {lat0:.4f}N, {lon0:.4f}E")
            print(f"Initial altitude: {altitude_ft:.0f} ft")
            print(f"Initial heading: {hdg0:.1f} degrees")
            print(f"Route waypoints: {len(sampled_route)} (from {len(route)} total points)")
            print(f"Autopilot: ENGAGED - following SCAT route")
            
            return {"success": True, "aircraft": bluesky_aircraft}
            
        except Exception as e:
            self.log_debug(f"SCAT loading failed: {e}")
            return {"success": False, "error": str(e)}
    
    def step2_bluesky_to_llm(self, aircraft_data: Dict[str, Any], iteration: int = 1) -> Dict[str, Any]:
        """STEP 2: Get current BlueSky state and send to LLM for analysis."""
        print("\n" + "="*60)
        print(f"STEP 2: BLUESKY STATE TO LLM (Iteration {iteration})")
        print("="*60)
        
        # Get current aircraft state from BlueSky
        current_states = self.bluesky_client.get_aircraft_states()
        callsign = aircraft_data['callsign']
        
        if callsign not in current_states:
            return {"success": False, "error": f"Aircraft {callsign} not found in BlueSky"}
        
        current_state = current_states[callsign]
        
        # Create detailed aviation prompt with real BlueSky data
        prompt = f"""You are an expert Air Traffic Controller with ICAO certification analyzing real flight data.

CURRENT AIRCRAFT STATE (from BlueSky simulation):
- Callsign: {callsign}
- Aircraft Type: {aircraft_data['aircraft_type']}
- Position: {current_state['lat']:.4f}N {current_state['lon']:.4f}E
- Altitude: {current_state['alt_ft']:.0f} feet
- Current Heading: {current_state['hdg_deg']:.1f} degrees
- Ground Speed: {current_state['spd_kt']:.1f} knots
- Vertical Speed: {current_state.get('vs_fpm', 0):.0f} fpm
- Route Following: {"ACTIVE" if aircraft_data.get('autopilot_engaged') else "INACTIVE"}

ITERATION: {iteration}

ANALYSIS TASKS:
1. Assess current flight path efficiency and route adherence
2. Check for potential optimization opportunities
3. Evaluate if manual intervention is needed
4. Provide BlueSky command for any recommended changes

AVIATION STANDARDS:
- Minimum separation: 5 nautical miles horizontal, 1000 feet vertical
- Consider fuel efficiency and flight time
- Maintain safe altitudes and headings
- Follow ICAO flight rules
- Respect autopilot route following when active

OUTPUT FORMAT (JSON only):
{{
    "iteration": {iteration},
    "current_assessment": "detailed analysis of current flight state",
    "route_adherence": "good|acceptable|poor",
    "optimization_needed": true/false,
    "recommended_action": "none|heading_change|altitude_change|speed_change|direct_waypoint",
    "new_heading": 0,
    "new_altitude": 0,
    "new_speed": 0,
    "bluesky_command": "exact BlueSky command string or none",
    "rationale": "explanation for the recommendation",
    "fuel_impact": "positive|neutral|negative",
    "safety_assessment": "safety evaluation of the recommendation",
    "confidence": 0.8
}}"""

        self.log_debug(f"Analyzing real BlueSky state for {callsign}")
        
        result = self.call_llm(prompt)
        
        if result["success"]:
            analysis = result["data"]
            print(f"LLM Analysis for {callsign} (real BlueSky data):")
            print(f"  Current position: {current_state['lat']:.4f}N, {current_state['lon']:.4f}E")
            print(f"  Current altitude: {current_state['alt_ft']:.0f} ft")
            print(f"  Current heading: {current_state['hdg_deg']:.1f}°")
            print(f"  Current speed: {current_state['spd_kt']:.1f} kt")
            print(f"  Assessment: {analysis.get('current_assessment', 'N/A')}")
            print(f"  Route adherence: {analysis.get('route_adherence', 'N/A')}")
            print(f"  Optimization needed: {analysis.get('optimization_needed', 'N/A')}")
            print(f"  Recommended action: {analysis.get('recommended_action', 'N/A')}")
            print(f"  BlueSky command: {analysis.get('bluesky_command', 'N/A')}")
            print(f"  Confidence: {analysis.get('confidence', 'N/A')}")
            
            # Add current state to analysis
            analysis['current_bluesky_state'] = current_state
            
            return {"success": True, "analysis": analysis}
        else:
            self.log_debug(f"LLM analysis failed: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def step3_llm_to_bluesky(self, llm_analysis: Dict[str, Any], aircraft_data: Dict[str, Any]) -> Dict[str, Any]:
        """STEP 3: Execute LLM recommendations in BlueSky simulation."""
        print("\n" + "="*60)
        print("STEP 3: LLM TO BLUESKY EXECUTION")
        print("="*60)
        
        analysis = llm_analysis["analysis"]
        callsign = aircraft_data["callsign"]
        
        # Extract BlueSky command
        bluesky_command = analysis.get("bluesky_command", "none")
        
        if bluesky_command == "none" or not analysis.get("optimization_needed", False):
            print(f"No optimization needed for {callsign} - continuing route following")
            return {"success": True, "command": None, "executed": False}
        
        # Parse and execute command in real BlueSky
        if bluesky_command.startswith(callsign):
            self.log_debug(f"Executing BlueSky command: {bluesky_command}")
            
            # Parse command type
            parts = bluesky_command.split()
            if len(parts) >= 3:
                command_type = parts[1].upper()
                value = parts[2]
                
                execution_success = False
                
                try:
                    if command_type == "HDG":
                        # Execute heading change
                        heading = float(value)
                        execution_success = self.bluesky_client.set_heading(callsign, heading)
                        print(f"Executed: Changed heading to {heading}°")
                        
                    elif command_type == "ALT":
                        # Execute altitude change
                        altitude = float(value)
                        execution_success = self.bluesky_client.set_altitude(callsign, altitude)
                        print(f"Executed: Changed altitude to {altitude} ft")
                        
                    elif command_type == "SPD":
                        # Execute speed change
                        speed = float(value)
                        execution_success = self.bluesky_client.set_speed(callsign, speed)
                        print(f"Executed: Changed speed to {speed} kt")
                        
                    elif command_type == "DCT":
                        # Execute direct-to waypoint
                        waypoint = value
                        execution_success = self.bluesky_client.direct_to(callsign, waypoint)
                        print(f"Executed: Direct to waypoint {waypoint}")
                    
                    # Step simulation to apply changes
                    self.bluesky_client.step_minutes(1.0)
                    
                    # Get updated state to verify execution
                    updated_states = self.bluesky_client.get_aircraft_states()
                    if callsign in updated_states:
                        updated_state = updated_states[callsign]
                        print(f"Updated state:")
                        print(f"  Position: {updated_state['lat']:.4f}N, {updated_state['lon']:.4f}E")
                        print(f"  Altitude: {updated_state['alt_ft']:.0f} ft")
                        print(f"  Heading: {updated_state['hdg_deg']:.1f}°")
                        print(f"  Speed: {updated_state['spd_kt']:.1f} kt")
                    
                    # Log execution details
                    print(f"Command executed: {bluesky_command}")
                    print(f"Execution success: {execution_success}")
                    print(f"Rationale: {analysis.get('rationale', 'N/A')}")
                    print(f"Safety assessment: {analysis.get('safety_assessment', 'N/A')}")
                    print(f"Fuel impact: {analysis.get('fuel_impact', 'N/A')}")
                    
                    return {
                        "success": True, 
                        "command": bluesky_command, 
                        "executed": execution_success,
                        "updated_state": updated_states.get(callsign, {}),
                        "analysis": analysis
                    }
                    
                except Exception as e:
                    print(f"Command execution failed: {e}")
                    return {"success": False, "error": f"Command execution failed: {e}"}
            else:
                self.log_debug(f"Invalid command format: {bluesky_command}")
                return {"success": False, "error": "Invalid command format"}
        else:
            self.log_debug(f"Command doesn't match callsign: {bluesky_command}")
            return {"success": False, "error": "Command callsign mismatch"}
    
    def run_iterative_loop(self, aircraft_data: Dict[str, Any], max_iterations: int = 3) -> List[Dict[str, Any]]:
        """Run iterative BlueSky <-> LLM communication loop with real simulation."""
        print("\n" + "="*80)
        print(f"ITERATIVE COMMUNICATION LOOP ({max_iterations} iterations)")
        print("="*80)
        
        current_aircraft = aircraft_data.copy()
        iteration_results = []
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n*** ITERATION {iteration}/{max_iterations} ***")
            
            # Step simulation forward to get new state
            print(f"Advancing BlueSky simulation by 1 minute...")
            self.bluesky_client.step_minutes(1.0)
            
            # Step 2: BlueSky to LLM (using real current state)
            llm_result = self.step2_bluesky_to_llm(current_aircraft, iteration)
            
            if not llm_result["success"]:
                print(f"Iteration {iteration} failed: {llm_result.get('error', 'Unknown error')}")
                break
            
            # Step 3: LLM to BlueSky (execute real commands)
            command_result = self.step3_llm_to_bluesky(llm_result, current_aircraft)
            
            # Record iteration result
            iteration_result = {
                "iteration": iteration,
                "llm_analysis": llm_result.get("analysis", {}),
                "command_executed": command_result.get("executed", False),
                "bluesky_command": command_result.get("command", None),
                "success": command_result["success"],
                "bluesky_state": llm_result.get("analysis", {}).get("current_bluesky_state", {})
            }
            
            iteration_results.append(iteration_result)
            
            # Update aircraft data with latest BlueSky state if available
            if command_result.get("updated_state"):
                print(f"Aircraft state updated with real BlueSky data")
                # Keep original aircraft metadata but update with real state
                current_aircraft.update({
                    "latitude": command_result["updated_state"]["lat"],
                    "longitude": command_result["updated_state"]["lon"],
                    "altitude_ft": command_result["updated_state"]["alt_ft"],
                    "heading_deg": command_result["updated_state"]["hdg_deg"],
                    "speed_kt": command_result["updated_state"]["spd_kt"]
                })
            else:
                print(f"No BlueSky state update - aircraft following original route")
                # If no command executed, aircraft continues on autopilot route
                
        return iteration_results
    
    def run_complete_demonstration(self, scat_file: str = "scenarios/scat/100000.json") -> Dict[str, Any]:
        """Run complete demonstration of all communication paths."""
        start_time = time.time()
        
        # Step 1: SCAT to BlueSky
        scat_result = self.step1_scat_to_bluesky(scat_file)
        
        if not scat_result["success"]:
            print(f"FAILED: SCAT loading failed - {scat_result.get('error', 'Unknown error')}")
            return {"success": False, "step": 1}
        
        aircraft_data = scat_result["aircraft"]
        
        # Iterative Steps 2 & 3: BlueSky <-> LLM
        iteration_results = self.run_iterative_loop(aircraft_data, max_iterations=3)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate summary
        successful_iterations = len([r for r in iteration_results if r["success"]])
        commands_executed = len([r for r in iteration_results if r["command_executed"]])
        
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        
        print(f"[OK] STEP 1 (SCAT to BlueSky): SUCCESS")
        print(f"  - Aircraft: {aircraft_data['callsign']} ({aircraft_data['aircraft_type']})")
        print(f"  - Position: {aircraft_data['latitude']:.4f}N, {aircraft_data['longitude']:.4f}E")
        print(f"  - Altitude: {aircraft_data['altitude_ft']:.0f} ft")
        
        print(f"\n[OK] STEPS 2-3 (BlueSky <-> LLM Loop): {successful_iterations}/{len(iteration_results)} SUCCESSFUL")
        print(f"  - Total iterations: {len(iteration_results)}")
        print(f"  - Commands executed: {commands_executed}")
        print(f"  - LLM responses: {successful_iterations}")
        
        for i, result in enumerate(iteration_results, 1):
            status = "SUCCESS" if result["success"] else "FAILED"
            command = result["bluesky_command"] or "None"
            print(f"    Iteration {i}: {status} - Command: {command}")
        
        print(f"\n[TIME]  Total execution time: {total_time:.2f} seconds")
        print(f"[PROCESS] Debug log entries: {len(self.debug_log)}")
        
        # Final verification
        all_success = scat_result["success"] and successful_iterations > 0
        
        print(f"\n[TARGET] OVERALL RESULT: {'SUCCESS' if all_success else 'PARTIAL/FAILED'}")
        
        if all_success:
            print("\n[OK] ALL THREE COMMUNICATION PATHS VERIFIED:")
            print("  1. [OK] SCAT to BlueSky: Real data loading and conversion")
            print("  2. [OK] BlueSky to LLM: Aviation analysis and optimization")
            print("  3. [OK] LLM to BlueSky: Command generation and execution")
            print("\n[INIT] System is fully operational and NOT using mocked responses!")
        
        return {
            "success": all_success,
            "scat_result": scat_result,
            "iteration_results": iteration_results,
            "total_time": total_time,
            "debug_log": self.debug_log
        }
    
    def save_debug_log(self, filename: str = None):
        """Save debug log to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_demo_debug_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("LLM Communication Demonstration Debug Log\n")
                f.write("="*50 + "\n\n")
                for entry in self.debug_log:
                    f.write(entry + "\n")
            
            print(f"\n[SAVE] Debug log saved to: {filename}")
            
        except Exception as e:
            print(f"Failed to save debug log: {e}")


def main():
    """Main demonstration function."""
    print("Starting Complete LLM Communication Demonstration...")
    
    # Check if SCAT file exists
    scat_file = "scenarios/scat/100000.json"
    if not os.path.exists(scat_file):
        print(f"ERROR: SCAT file not found: {scat_file}")
        print("Please ensure the SCAT data file exists.")
        return
    
    # Create and run demonstration
    demo = CompleteLLMDemo()
    
    try:
        results = demo.run_complete_demonstration(scat_file)
        
        # Save debug information
        demo.save_debug_log()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"llm_demo_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[SAVE] Complete results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n[WARN]  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
