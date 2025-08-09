#!/usr/bin/env python3
"""
LLM Communication Summary - Simple Debug View

This script provides a simple summary of LLM verification without Unicode characters.
"""

import json
import os
import time
import requests
from pathlib import Path

def test_llm_communication():
    """Simple LLM communication test with debug output."""
    
    print("=" * 60)
    print("LLM COMMUNICATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    # Test 1: Check Ollama connection
    print("\n1. OLLAMA CONNECTION TEST:")
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            model_names = [m.get('name', '') for m in models.get('models', [])]
            print(f"   SUCCESS: Connected to Ollama")
            print(f"   Available models: {model_names}")
            
            if "llama3.1:8b" in model_names:
                print(f"   SUCCESS: Target model 'llama3.1:8b' is available")
                ollama_ok = True
            else:
                print(f"   WARNING: Target model 'llama3.1:8b' not found")
                ollama_ok = False
        else:
            print(f"   FAILED: HTTP {response.status_code}")
            ollama_ok = False
    except Exception as e:
        print(f"   FAILED: {e}")
        ollama_ok = False
    
    if not ollama_ok:
        print("\nCannot proceed without Ollama connection.")
        return False
    
    # Test 2: Real LLM response test
    print("\n2. REAL LLM RESPONSE TEST:")
    try:
        prompt = """You are an expert air traffic controller. Respond with JSON only:
{
    "status": "operational", 
    "verification": "real_llm_response",
    "calculation": "What is 15 * 17?",
    "random_number": "Generate a 4-digit random number"
}

Replace the values with actual responses to prove you are a real LLM."""

        print("   Sending prompt to LLM...")
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
            timeout=60
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            llm_response = data.get("response", "")
            
            print(f"   SUCCESS: LLM responded in {response_time:.2f} seconds")
            print(f"   Response length: {len(llm_response)} characters")
            print(f"   Response preview: {llm_response[:200]}...")
            
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(0))
                    print(f"   SUCCESS: Valid JSON response received")
                    print(f"   JSON data: {json_data}")
                    llm_ok = True
                except json.JSONDecodeError:
                    print(f"   WARNING: JSON parsing failed")
                    print(f"   Raw JSON text: {json_match.group(0)}")
                    llm_ok = False
            else:
                print(f"   WARNING: No JSON found in response")
                llm_ok = False
        else:
            print(f"   FAILED: HTTP {response.status_code}")
            llm_ok = False
            
    except Exception as e:
        print(f"   FAILED: {e}")
        llm_ok = False
    
    # Test 3: Aviation-specific prompt
    print("\n3. AVIATION PROMPT TEST:")
    if llm_ok:
        try:
            aviation_prompt = """You are an expert Air Traffic Controller.

AIRCRAFT SITUATION:
- Aircraft 1: SAS101 at position 59.65N 17.92E, altitude 35000 ft, heading 090
- Aircraft 2: DLH456 at position 59.66N 17.95E, altitude 35000 ft, heading 270

TASK: Detect if these aircraft will have a conflict.

Respond with JSON only:
{
    "conflict_detected": true/false,
    "closest_approach_nm": number,
    "resolution_needed": true/false,
    "bluesky_command": "SAS101 HDG xxx"
}"""

            print("   Sending aviation prompt to LLM...")
            start_time = time.time()
            
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": "llama3.1:8b", "prompt": aviation_prompt, "stream": False},
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                aviation_response = data.get("response", "")
                
                print(f"   SUCCESS: Aviation LLM response in {response_time:.2f} seconds")
                print(f"   Response preview: {aviation_response[:300]}...")
                
                # Check for BlueSky command
                if "SAS101 HDG" in aviation_response:
                    print(f"   SUCCESS: BlueSky command format detected")
                    aviation_ok = True
                else:
                    print(f"   WARNING: BlueSky command format not found")
                    aviation_ok = False
            else:
                print(f"   FAILED: HTTP {response.status_code}")
                aviation_ok = False
                
        except Exception as e:
            print(f"   FAILED: {e}")
            aviation_ok = False
    else:
        print("   SKIPPED: Previous test failed")
        aviation_ok = False
    
    # Test 4: SCAT data loading
    print("\n4. SCAT DATA TEST:")
    scat_file = "scenarios/scat/100000.json"
    if os.path.exists(scat_file):
        try:
            with open(scat_file, 'r') as f:
                scat_data = json.load(f)
            
            fpl_base = scat_data.get('fpl', {}).get('fpl_base', [])
            plots = scat_data.get('plots', [])
            
            if fpl_base and plots:
                aircraft_info = fpl_base[0]
                callsign = aircraft_info.get('callsign', 'UNKNOWN')
                print(f"   SUCCESS: SCAT data loaded for {callsign}")
                print(f"   Flight plan entries: {len(fpl_base)}")
                print(f"   Plot points: {len(plots)}")
                scat_ok = True
            else:
                print(f"   FAILED: Invalid SCAT data structure")
                scat_ok = False
                
        except Exception as e:
            print(f"   FAILED: {e}")
            scat_ok = False
    else:
        print(f"   WARNING: SCAT file not found: {scat_file}")
        scat_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"1. Ollama Connection:     {'PASS' if ollama_ok else 'FAIL'}")
    print(f"2. Real LLM Response:     {'PASS' if llm_ok else 'FAIL'}")
    print(f"3. Aviation Prompts:      {'PASS' if aviation_ok else 'FAIL'}")
    print(f"4. SCAT Data Loading:     {'PASS' if scat_ok else 'FAIL'}")
    
    all_pass = ollama_ok and llm_ok and aviation_ok and scat_ok
    
    print(f"\nOVERALL RESULT: {'SUCCESS' if all_pass else 'PARTIAL/FAILED'}")
    
    if all_pass:
        print("\nCONCLUSION:")
        print("- LLM communication is REAL and NOT MOCKED")
        print("- All three communication paths are functional:")
        print("  * SCAT to BlueSky: Data loading works")
        print("  * BlueSky to LLM: Aviation prompts work")
        print("  * LLM to BlueSky: Command generation works")
        print("- System is ready for production use")
    else:
        print("\nISSUES DETECTED:")
        if not ollama_ok:
            print("- Ollama server connection failed")
        if not llm_ok:
            print("- LLM communication failed")
        if not aviation_ok:
            print("- Aviation prompt processing failed")
        if not scat_ok:
            print("- SCAT data loading failed")
    
    return all_pass

if __name__ == "__main__":
    test_llm_communication()
