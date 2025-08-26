# evm_calculations.py
import numpy as np
from typing import List, Dict, Tuple, Optional
from rf_components import DSA, GainBlock, Mixer, LOBlock, ADC
from utils import db_to_linear, linear_to_db, dbm_to_watts, watts_to_dbm

class RFChain:
    """Represents the complete RF receiver chain"""
    
    def __init__(self):
        self.components = []
        self.component_names = []
        
    def add_component(self, component, name: str):
        """Add a component to the chain"""
        self.components.append(component)
        self.component_names.append(name)
    
    def clear_chain(self):
        """Clear all components from the chain"""
        self.components = []
        self.component_names = []

class SystemCalculator:
    """Main class for system-level EVM calculations"""
    
    def __init__(self):
        self.chain = RFChain()
        self.lo_block = None  # Store LO block separately
        
    def calculate_cascade_parameters(self, input_power_dbm: float, rf_freq: float, 
                                   if_freq: float, bandwidth_mhz: float, 
                                   par_im3_db: float = 7.0) -> Dict:
        """
        Calculate cascaded system parameters at each stage
        
        Returns dictionary with arrays for each parameter at each stage
        """
        n_stages = len(self.chain.components)
        
        # Initialize arrays for each parameter
        signal_power = np.zeros(n_stages + 1)  # +1 for input
        noise_power = np.zeros(n_stages + 1)
        distortion_power = np.zeros(n_stages + 1)
        ipn_dbc = np.zeros(n_stages + 1)
        evm_db = np.zeros(n_stages + 1)
        
        # Cascaded parameters for tracking
        cumulative_gain = 0.0
        cumulative_nf_linear = 0.0
        
        # Input conditions
        signal_power[0] = input_power_dbm
        noise_power[0] = -174 + linear_to_db(bandwidth_mhz * 1e6)  # Thermal noise floor
        distortion_power[0] = -200  # Very low initial distortion
        ipn_dbc[0] = 0  # Initialize to 0 (will be blank until mixer)
        
        # Calculate input EVM
        signal_linear = db_to_linear(signal_power[0])
        noise_linear = db_to_linear(noise_power[0])
        distortion_linear = db_to_linear(distortion_power[0])
        total_error_linear = noise_linear + distortion_linear  # No phase noise yet
        evm_db[0] = linear_to_db(total_error_linear / signal_linear)
        
        # Find mixer index
        mixer_index = -1
        lo_integrated_pn = 0
        
        for i, (component, name) in enumerate(zip(self.chain.components, self.chain.component_names)):
            # Check for actual Mixer component type
            if isinstance(component, Mixer):
                mixer_index = i
                break
        
        # Calculate LO integrated phase noise if LO block exists
        if self.lo_block is not None:
            try:
                lo_freq = rf_freq - if_freq  # LO frequency
                scs_khz = 15.0  # Use the actual SCS from system parameters
                lo_integrated_pn = self.lo_block.calculate_integrated_pn(lo_freq, scs_khz, bandwidth_mhz)
            except Exception as e:
                print(f"Debug: Error calculating LO PN: {e}")
                import traceback
                traceback.print_exc()
                lo_integrated_pn = 0
        else:
            print(f"Debug: No LO block available")
        
        # Track cascaded distortion - start with input distortion
        current_distortion_power = distortion_power[0]
        
        for i, component in enumerate(self.chain.components):
            # Determine operating frequency for this component
            if isinstance(component, Mixer):
                # Mixers always use RF frequency for their parameters
                operating_freq = rf_freq
            elif mixer_index >= 0 and i > mixer_index:
                # Components after mixer use IF frequency
                operating_freq = abs(if_freq) if if_freq != 0 else rf_freq
            else:
                # Components before mixer use RF frequency
                operating_freq = rf_freq
            
            # Get component parameters
            gain = component.get_gain(operating_freq)
            nf = component.get_nf(operating_freq)
            iip3 = component.get_iip3(operating_freq)
            
            # Update signal power
            signal_power[i + 1] = signal_power[i] + gain
            
            # Cascade noise figure calculation (Friis formula)
            if i == 0:
                cumulative_nf_linear = db_to_linear(nf)
            else:
                gain_product = db_to_linear(cumulative_gain)
                cumulative_nf_linear += (db_to_linear(nf) - 1) / gain_product
            
            # Update cumulative gain
            cumulative_gain += gain
            
            # Calculate noise power at this stage
            cumulative_nf_db = linear_to_db(cumulative_nf_linear)
            noise_power[i + 1] = -174 + linear_to_db(bandwidth_mhz * 1e6) + cumulative_nf_db + cumulative_gain
            
            # Handle distortion power calculation
            # First, apply gain/attenuation to existing distortion from previous stages
            if current_distortion_power > -150:  # Only if there's meaningful distortion
                current_distortion_power += gain
            else:
                current_distortion_power = -200
            
            # Check if this component generates new distortion
            # Only DSA is treated as ideal (no distortion generation)
            is_dsa = isinstance(component, DSA)
            
            if not is_dsa:
                # All CSV components (GainBlock, Mixer, ADC) generate distortion based on their IIP3
                input_power_to_stage = signal_power[i]  # Input power to this stage
                
                # IM3 power generated by this stage: P_IM3 = 3*P_in - 2*IIP3 + PAR_IM3
                im3_from_stage = 3 * input_power_to_stage - 2 * iip3 + par_im3_db
                
                # Apply gain of this stage to the generated IM3
                im3_at_stage_output = im3_from_stage + gain
                
                # Combine existing distortion with new distortion (power sum)
                if current_distortion_power > -150:
                    existing_distortion_linear = db_to_linear(current_distortion_power)
                else:
                    existing_distortion_linear = 0
                
                new_distortion_linear = db_to_linear(im3_at_stage_output)
                total_distortion_linear = existing_distortion_linear + new_distortion_linear
                current_distortion_power = linear_to_db(total_distortion_linear)
            
            # Set distortion power for this stage
            distortion_power[i + 1] = current_distortion_power
            
            # Calculate IPN (Integrated Phase Noise) - starts from mixer stage onwards
            # Phase noise gets mixed down to IF at the mixer stage
            if mixer_index >= 0 and i >= mixer_index and lo_integrated_pn != 0:
                ipn_dbc[i + 1] = lo_integrated_pn
            else:
                ipn_dbc[i + 1] = 0
            
            # Calculate EVM properly: EVM = 10*log10((P_error_total) / P_signal)
            signal_linear = db_to_linear(signal_power[i + 1])
            noise_linear = db_to_linear(noise_power[i + 1])
            
            # Handle distortion power (avoid very low values)
            if distortion_power[i + 1] > -150:
                distortion_linear = db_to_linear(distortion_power[i + 1])
            else:
                distortion_linear = 0
            
            # Handle phase noise (only from mixer stage onwards)
            if mixer_index >= 0 and i >= mixer_index and lo_integrated_pn > -150:
                # Phase noise power = signal power + IPN (in dBc)
                phase_noise_power_dbm = signal_power[i + 1] + lo_integrated_pn
                phase_noise_linear = db_to_linear(phase_noise_power_dbm)
            else:
                phase_noise_linear = 0
            
            # Total error power
            total_error_linear = noise_linear + distortion_linear + phase_noise_linear
            
            # EVM calculation
            if total_error_linear > 0 and signal_linear > 0:
                evm_db[i + 1] = linear_to_db(total_error_linear / signal_linear)
            else:
                evm_db[i + 1] = -200  # Very good EVM
        
        return {
            'signal_power': signal_power,
            'noise_power': noise_power,
            'distortion_power': distortion_power,
            'ipn_dbc': ipn_dbc,
            'evm_db': evm_db,
            'component_names': ['Input Signal'] + self.chain.component_names,
            'total_gain': cumulative_gain,
            'total_nf': linear_to_db(cumulative_nf_linear),
            'mixer_index': mixer_index,
            'lo_integrated_pn': lo_integrated_pn
        }