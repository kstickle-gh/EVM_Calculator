# rf_components.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from utils import load_component_csv, linear_interp_freq, db_to_linear, linear_to_db

class DSA:
    """Digital Step Attenuator"""
    
    def __init__(self, attenuation: int = 0):
        self.attenuation = max(0, min(30, attenuation))  # Clamp to 0-30 dB
    
    def get_gain(self, freq: float) -> float:
        return -self.attenuation
    
    def get_nf(self, freq: float) -> float:
        return self.attenuation
    
    def get_iip3(self, freq: float) -> float:
        return 60.0  # dBm

class GainBlock:
    """Amplifier or gain block from CSV data"""
    
    def __init__(self, csv_path: str):
        self.data = load_component_csv(csv_path)
        self.freq_array = self.data['Freq (GHz)'].values
        self.gain_array = self.data['Gain (dB)'].values
        self.nf_array = self.data['NF (dB)'].values
        self.iip3_array = self.data['IIP3 (dBm)'].values
        
    def get_gain(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.gain_array, freq)
    
    def get_nf(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.nf_array, freq)
    
    def get_iip3(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.iip3_array, freq)

class Mixer:
    """Mixer from CSV data"""
    
    def __init__(self, csv_path: str):
        self.data = load_component_csv(csv_path)
        self.freq_array = self.data['Freq (GHz)'].values
        self.gain_array = self.data['Gain (dB)'].values
        self.nf_array = self.data['NF (dB)'].values
        self.iip3_array = self.data['IIP3 (dBm)'].values
        
    def get_gain(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.gain_array, freq)
    
    def get_nf(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.nf_array, freq)
    
    def get_iip3(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.iip3_array, freq)

class LOBlock:
    """Local Oscillator with phase noise data"""
    
    def __init__(self, csv_path: str):
        self.data = load_component_csv(csv_path)
        self.freq_array = self.data['Freq (GHz)'].values
        # Phase noise columns (assuming they're labeled as offset frequencies)
        self.pn_columns = [col for col in self.data.columns if col != 'Freq (GHz)']
        
    def get_phase_noise_profile(self, lo_freq: float) -> Dict[str, float]:
        """Get phase noise profile at specified LO frequency"""
        pn_profile = {}
        for col in self.pn_columns:
            pn_value = linear_interp_freq(
                self.freq_array, 
                self.data[col].values, 
                lo_freq
            )
            pn_profile[col] = pn_value
        return pn_profile
    
    def calculate_integrated_pn(self, lo_freq: float, scs_khz: float, bandwidth_mhz: float) -> float:
        """Calculate DSB integrated phase noise"""
        
        pn_profile = self.get_phase_noise_profile(lo_freq)
        
        # Convert column names to frequencies (assuming format like '1k', '10k', etc.)
        offset_freqs = []
        pn_values = []
        
        for col, pn_val in pn_profile.items():
            # Parse frequency from column name
            try:
                if col.endswith('k'):
                    freq_hz = float(col[:-1]) * 1e3
                elif col.endswith('M'):
                    freq_hz = float(col[:-1]) * 1e6
                elif col.endswith('K'):  # Handle uppercase K
                    freq_hz = float(col[:-1]) * 1e3
                else:
                    # Try to parse as plain number (assume Hz)
                    freq_hz = float(col)
                
                offset_freqs.append(freq_hz)
                pn_values.append(pn_val)
                
            except ValueError as e:
                # print(f"Debug: Could not parse column '{col}': {e}")
                continue
        
        if len(offset_freqs) == 0:
            print("Debug: No valid offset frequencies found")
            return -200  # Very low phase noise if no data
        
        # Sort by frequency
        sorted_indices = np.argsort(offset_freqs)
        offset_freqs = np.array(offset_freqs)[sorted_indices]
        pn_values = np.array(pn_values)[sorted_indices]
        
        # print(f"Debug: Sorted offset freqs: {offset_freqs}")
        # print(f"Debug: Sorted PN values: {pn_values}")
        
        # Integration limits
        f_low = scs_khz * 1e3 / 2  # Hz
        f_high = bandwidth_mhz * 1e6 / 2  # Hz
        
        # print(f"Debug: Integration limits: {f_low} Hz to {f_high} Hz")
        
        # Check if we have data in the integration range
        if f_high < offset_freqs[0] or f_low > offset_freqs[-1]:
            print(f"Debug: Integration range outside available data range")
            return -200
        
        # Create integration frequency array
        f_integration = np.logspace(np.log10(max(f_low, offset_freqs[0])), 
                                   np.log10(min(f_high, offset_freqs[-1])), 1000)
        
        # Interpolate phase noise over integration range
        pn_interp = np.interp(f_integration, offset_freqs, pn_values)

        # Convert to linear and integrate (DSB)
        pn_linear = db_to_linear(pn_interp)
        integrated_pn_linear = 2 * np.trapz(pn_linear, f_integration)
        
        # Convert back to dB
        result = linear_to_db(integrated_pn_linear)
        
        return result

class ADC:
    """ADC from CSV data"""
    
    def __init__(self, csv_path: str, is_iq: bool = False):
        self.data = load_component_csv(csv_path)
        self.freq_array = self.data['Freq (GHz)'].values
        self.adcfs_array = self.data['ADCFS (dBm)'].values
        self.nsd_array = self.data['NSD (dBFS)'].values
        self.iip3_array = self.data['IIP3 (dBm)'].values
        self.is_iq = is_iq
        
    def get_adcfs(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.adcfs_array, freq)
    
    def get_nsd(self, freq: float) -> float:
        nsd = linear_interp_freq(self.freq_array, self.nsd_array, freq)
        # 3dB improvement for IQ ADC
        return nsd - 3 if self.is_iq else nsd
    
    def get_iip3(self, freq: float) -> float:
        return linear_interp_freq(self.freq_array, self.iip3_array, freq)
    
    def get_nf(self, freq: float) -> float:
        """Calculate noise figure: NF = 174 + NSD + ADCFS"""
        return 174 + self.get_nsd(freq) + self.get_adcfs(freq)
    
    def get_gain(self, freq: float) -> float:
        """ADC has no gain"""
        return 0.0
