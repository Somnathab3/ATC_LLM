#!/usr/bin/env python3
"""
CDR Package Deduplication Report
Generated automatically during code audit
"""

DUPLICATE_ANALYSIS = {
    "similar_filenames": [
        ("llm_client.py", "enhanced_llm_client.py")
    ],
    
    "duplicate_functions": {
        "bearing_deg": ["geodesy.py", "monte_carlo_intruders.py"],
        "destination_point": ["geodesy.py (as destination_point_nm)", "monte_carlo_intruders.py"],
        # LLM client methods - multiple implementations across classes
        "ask_detect": ["LlamaClient", "LLMClient", "EnhancedLLMClient"],
        "ask_resolve": ["LlamaClient", "LLMClient", "EnhancedLLMClient"],
        "build_detect_prompt": ["LlamaClient", "LLMClient", "EnhancedLLMClient"],
        "build_resolve_prompt": ["LlamaClient", "LLMClient", "EnhancedLLMClient"],
    },
    
    "duplicate_classes": {
        "Config": "schemas.py (appears twice)",
        "MockResult": "llm_client.py (multiple versions)",
    },
    
    "canonicalization_plan": {
        "1_llm_clients": {
            "action": "merge_and_standardize",
            "target": "llm_client.py",
            "strategy": [
                "Keep LlamaClient as the primary implementation",
                "Merge EnhancedLLMClient features into LlamaClient",
                "Remove redundant LLMClient wrapper class",
                "Consolidate MockResult classes",
                "Maintain backward compatibility for tests"
            ]
        },
        
        "2_geodesy": {
            "action": "standardize_imports",
            "target": "geodesy.py",
            "strategy": [
                "Use geodesy.py as canonical source",
                "Update monte_carlo_intruders.py to import from geodesy",
                "Remove duplicate function implementations"
            ]
        },
        
        "3_schemas": {
            "action": "clean_duplicates",
            "target": "schemas.py",
            "strategy": [
                "Remove duplicate Config class definition",
                "Ensure single source of truth for all schema types"
            ]
        }
    }
}

def print_report():
    """Print the deduplication analysis report."""
    print("=== CDR PACKAGE DEDUPLICATION REPORT ===\n")
    
    print("SIMILAR FILENAMES:")
    for pair in DUPLICATE_ANALYSIS["similar_filenames"]:
        print(f"  {pair[0]} <-> {pair[1]}")
    print()
    
    print("DUPLICATE FUNCTIONS:")
    for func, locations in DUPLICATE_ANALYSIS["duplicate_functions"].items():
        print(f"  {func}: {locations}")
    print()
    
    print("DUPLICATE CLASSES:")
    for cls, location in DUPLICATE_ANALYSIS["duplicate_classes"].items():
        print(f"  {cls}: {location}")
    print()
    
    print("CANONICALIZATION PLAN:")
    for step, details in DUPLICATE_ANALYSIS["canonicalization_plan"].items():
        print(f"  {step}: {details['action']} -> {details['target']}")
        for strategy in details['strategy']:
            print(f"    - {strategy}")
        print()

if __name__ == "__main__":
    print_report()
