"""
Shared output utilities for ATC LLM system.

This module provides consistent output path generation using the new
Date_time_type folder structure across all components.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

def get_output_path(test_type: str, timestamp: Optional[str] = None, base_dir: str = "Output") -> Path:
    """
    Generate standardized output path with Date_time_type structure.
    
    Args:
        test_type: Type of test (batch, simulation, llm, verification, visualization, etc.)
        timestamp: Optional timestamp string (YYYYMMDD_HHMMSS format)
        base_dir: Base output directory (default: "Output")
    
    Returns:
        Path object for the organized output directory
        
    Example:
        >>> get_output_path("batch")
        PosixPath('Output/20250809_143022_batch')
        
        >>> get_output_path("simulation", "20250808_145935")
        PosixPath('Output/20250808_145935_simulation')
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract date and time components from timestamp
    if len(timestamp) >= 15:  # YYYYMMDD_HHMMSS format
        date_part = timestamp[:8]   # YYYYMMDD
        time_part = timestamp[9:15]  # HHMMSS
    elif len(timestamp) >= 13:     # YYYYMMDD_HHMM format
        date_part = timestamp[:8]   # YYYYMMDD
        time_part = timestamp[9:13] + "00"  # HHMMSS (add seconds)
    else:
        # Fallback - use provided timestamp as-is
        date_part = timestamp[:8] if len(timestamp) >= 8 else timestamp
        time_part = timestamp[8:] if len(timestamp) > 8 else "000000"
    
    # Create folder name: Date_time_type
    folder_name = f"{date_part}_{time_part}_{test_type}"
    
    # Create full path
    output_dir = Path(base_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def get_output_file_path(test_type: str, filename: str, timestamp: Optional[str] = None, base_dir: str = "Output") -> Path:
    """
    Generate full file path within organized output structure.
    
    Args:
        test_type: Type of test (batch, simulation, llm, etc.)
        filename: Name of the output file
        timestamp: Optional timestamp string
        base_dir: Base output directory
    
    Returns:
        Full path to the output file
        
    Example:
        >>> get_output_file_path("batch", "results.json")
        PosixPath('Output/20250809_143022_batch/results.json')
    """
    output_dir = get_output_path(test_type, timestamp, base_dir)
    return output_dir / filename

# Common test types for consistency
class TestTypes:
    """Standard test type constants."""
    BATCH = "batch"
    SIMULATION = "simulation"
    LLM = "llm"
    VERIFICATION = "verification"
    VISUALIZATION = "visualization"
    HEALTHCHECK = "healthcheck"
    COMPARE = "compare"
    DEMO = "demo"
