#!/usr/bin/env python3

import requests
import json

# Test the exact same call as the LLM client
def test_ollama():
    host = "http://127.0.0.1:11434"
    model_name = "llama3.1:8b"
    prompt = 'You are an ATC conflict resolver. Given a detected conflict, return strict JSON:\n{"action":"turn|climb|descend","params":{},"ratio":0.0,"reason":"<text>"}\nDetection: "TEST"\nConstraints: {"max_resolution_angle_deg": 30}\nOnly JSON. No extra text.'
    
    print(f"Testing: {host}/api/generate")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt[:100]}...")
    
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=30,
        )
        print(f"Status Code: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        
        if resp.status_code >= 400:
            print(f"ERROR: {resp.status_code}")
            print(f"Response text: {resp.text}")
            return None
            
        data = resp.json()
        print(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if isinstance(data, dict):
            txt = data.get("response", "")
            print(f"Response text: {txt[:200]}...")
            
            # Try to extract JSON
            import re
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            if m:
                try:
                    json_data = json.loads(m.group(0))
                    print(f"Extracted JSON: {json_data}")
                    return json_data
                except Exception as e:
                    print(f"JSON parse error: {e}")
            else:
                print("No JSON found in response")
        
        return data
        
    except Exception as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    test_ollama()
