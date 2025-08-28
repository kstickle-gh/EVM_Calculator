# utils.py
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional

def load_component_csv(filepath: str) -> pd.DataFrame:
    """Load component data from CSV file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Component file not found: {filepath}")
    return pd.read_csv(filepath)

def get_available_components(component_type: str) -> List[str]:
    """Get list of available component files for a given type"""
    component_dir = f"./Components/{component_type}"
    if not os.path.exists(component_dir):
        return []
    return [f for f in os.listdir(component_dir) if f.endswith('.csv')]

def linear_interp_freq(freq_array: np.ndarray, param_array: np.ndarray, target_freq: float) -> float:
    """Linear interpolation for frequency-dependent parameters"""
    if target_freq <= freq_array[0]:
        return param_array[0]
    elif target_freq >= freq_array[-1]:
        return param_array[-1]
    else:
        return np.interp(target_freq, freq_array, param_array)

def db_to_linear(db_value: float) -> float:
    """Convert dB to linear scale"""
    return 10**(db_value/10)

def linear_to_db(linear_value: float) -> float:
    """Convert linear scale to dB"""
    return 10 * np.log10(linear_value)

def dbm_to_watts(dbm_value: float) -> float:
    """Convert dBm to watts"""
    return 10**((dbm_value - 30)/10)

def watts_to_dbm(watts_value: float) -> float:
    """Convert watts to dBm"""
    return 30 + 10 * np.log10(watts_value)

def check_component_libraries():
    """Check and display available component libraries"""
    
    component_types = ['Gain Blocks', 'Mixers', 'LO Blocks', 'ADCs']
    
    print("Component Library Status:")
    print("-" * 40)
    
    for comp_type in component_types:
        components = get_available_components(comp_type)
        print(f"{comp_type:15}: {len(components)} files found")
        
        if components:
            for comp in components[:3]:  # Show first 3 files
                print(f"  └─ {comp}")
            if len(components) > 3:
                print(f"  └─ ... and {len(components)-3} more")
        else:
            print(f"  └─ No files found in ./Components/{comp_type}/")
        print()
