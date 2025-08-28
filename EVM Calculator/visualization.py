# visualization.py - Simplified with new signal chain table and improved layout
import matplotlib.pyplot
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from typing import Dict, List, Optional, Tuple
from rf_components import DSA, GainBlock, Mixer, LOBlock, ADC
from evm_calculations import SystemCalculator
from utils import get_available_components

class SignalChainAnalyzer:
    """Interactive signal chain analyzer with live parameter updates"""
    
    def __init__(self):
        self.system_calc = SystemCalculator()
        self.create_widgets()
        self.setup_layout()
        self.bind_events()
        
    def create_widgets(self):
        """Create all input widgets"""
        InitRF1 = 'att_3dB.csv'
        InitRF2 = 'None'
        InitRF3 = 'None'
        InitRF4 = 'None'
        InitCoarse1 = 'adl8113_15dB.csv'
        InitCoarse2 = 'adl8113_byp.csv'
        InitCoarse3 = 'adl8113_amp.csv'
        InitCoarse4 = 'None'
        InitRF5 = 'None'
        InitRF6 = 'adl8113_amp.csv'
        InitRF7 = 'att_5dB.csv'
        InitLO = 'LO_ideal.csv'
        InitMixer = 'HMC8193_TRF1305.csv'
        InitIF1 = 'att_1dB.csv'
        InitIF2 = 'None'
        InitIF3 = 'None'
        InitIF4 = 'None'
        InitIF5 = 'None'
        InitIF6 = 'None'
        InitADC = 'adc_XCZU65.csv'

        
        # System Parameters - wider description width, narrower input width
        style = {'description_width': '150px'}
        layout = {'width': '250px'}
        
        self.rf_freq = widgets.FloatText(value=8.0, description='RF Freq (GHz):', style=style, layout=layout)
        self.if_freq = widgets.FloatText(value=0.0, description='IF Freq (GHz):', style=style, layout=layout)
        self.bandwidth = widgets.FloatText(value=100.0, description='Bandwidth (MHz):', style=style, layout=layout)
        self.scs = widgets.FloatText(value=60.0, description='SCS (kHz):', style=style, layout=layout)
        self.par_im3 = widgets.FloatText(value=7.0, description='PAR IM3 (dB):', style=style, layout=layout)
        self.iq_adc = widgets.Checkbox(value=False, description='IQ ADC?', style=style)
        self.adc_backoff = widgets.FloatText(value=12.0, description='ADC Backoff (dB):', style=style, layout=layout)
        
        # Signal Chain Components - consistent styling
        chain_style = {'description_width': '120px'}
        chain_layout = {'width': '280px'}
        
        self.input_signal = widgets.FloatText(value=0.0, description='RF Input (dBm):', style=chain_style, layout=chain_layout)
        
        # Get available CSV files
        gain_files = ['None'] + get_available_components('Gain Blocks')
        mixer_files = ['None'] + get_available_components('Mixers')
        lo_files = ['None'] + get_available_components('LO Blocks')
        adc_files = ['None'] + get_available_components('ADCs')
        
        # Gain Blocks (4 regular + 4 coarse)
        self.rf_gain_blocks = []
        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF1, description='RF Gain 1:', style=chain_style, layout=chain_layout))
        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF2, description='RF Gain 2:', style=chain_style, layout=chain_layout))
        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF3, description='RF Gain 3:', style=chain_style, layout=chain_layout))
        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF4, description='RF Gain 4:', style=chain_style, layout=chain_layout))
        
        # Coarse Gain Blocks (radio button selection)
        self.coarse_gain_blocks = []
        self.coarse_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitCoarse1, description='Coarse 1:', style=chain_style, layout=chain_layout))
        self.coarse_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitCoarse2, description='Coarse 2:', style=chain_style, layout=chain_layout))
        self.coarse_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitCoarse3, description='Coarse 3:', style=chain_style, layout=chain_layout))
        self.coarse_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitCoarse4, description='Coarse 4:', style=chain_style, layout=chain_layout))
        self.coarse_gain_select = widgets.RadioButtons(options=[1, 2, 3, 4], value=1, description='Active Coarse Gain Block:',style=chain_style)
        
        # Additional components
        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF5, description='RF Gain 5:', style=chain_style, layout=chain_layout))

        #DSA block
        self.dsa = widgets.IntSlider(value=0, min=0, max=30, description='DSA (dB):',style=chain_style, layout=chain_layout)

        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF6, description='RF Gain 6:', style=chain_style, layout=chain_layout))
        self.rf_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitRF7, description='RF Gain 7:', style=chain_style, layout=chain_layout))
        
        self.lo_block = widgets.Dropdown(options=lo_files, value=InitLO, description='LO Block:', style=chain_style, layout=chain_layout)
        self.mixer_block = widgets.Dropdown(options=mixer_files, value=InitMixer, description='Mixer Block:', style=chain_style, layout=chain_layout)
        
        # Post-mixer gain blocks (6 blocks at IF frequency)
        self.if_gain_blocks = []
        self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitIF1, description='IF Gain 1:', style=chain_style, layout=chain_layout))
        self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitIF2, description='IF Gain 2:', style=chain_style, layout=chain_layout))
        self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitIF3, description='IF Gain 3:', style=chain_style, layout=chain_layout))
        self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitIF4, description='IF Gain 4:', style=chain_style, layout=chain_layout))
        self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitIF5, description='IF Gain 5:', style=chain_style, layout=chain_layout))
        self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value=InitIF6, description='IF Gain 6:', style=chain_style, layout=chain_layout))

        self.adc_block = widgets.Dropdown(options=adc_files, value=InitADC, description='ADC Block:', style=chain_style, layout=chain_layout)
        
        # Output display
        self.output_display = widgets.Output()
        
    def setup_layout(self):
        """Setup the widget layout"""
        
        # System parameters column - fixed width to prevent cutoff
        sys_params = widgets.VBox([
            widgets.HTML("<h3>System Parameters</h3>"),
            self.rf_freq, self.if_freq, self.bandwidth, self.scs,
            self.par_im3, self.iq_adc, self.adc_backoff
        ], layout={'width': '350px'})  # Fixed width for system parameters
        
        # Coarse gain selection section
        coarse_gain_section = widgets.VBox([
            self.coarse_gain_select,
            widgets.HTML("<b>--------------------------------</b>"),
            self.coarse_gain_blocks[0],
            self.coarse_gain_blocks[1], 
            self.coarse_gain_blocks[2],
            self.coarse_gain_blocks[3],
            widgets.HTML("<b>--------------------------------</b>")
        ])
        
        # Signal chain column
        signal_chain = widgets.VBox([
            widgets.HTML("<h3>Signal Chain</h3>"),
            widgets.HTML("<b>-- RF Components --</b>"),
            self.input_signal,
            self.rf_gain_blocks[0],
            self.rf_gain_blocks[1],
            self.rf_gain_blocks[2],
            self.rf_gain_blocks[3],
            coarse_gain_section,
            self.rf_gain_blocks[4],
            self.dsa,
            self.rf_gain_blocks[5],
            self.rf_gain_blocks[6],
            widgets.HTML("<b>-- LO + Mixer --</b>"),
            self.lo_block,
            self.mixer_block,
            widgets.HTML("<b>-- IF Components --</b>"),
            self.if_gain_blocks[0],
            self.if_gain_blocks[1],
            self.if_gain_blocks[2],
            self.if_gain_blocks[3],
            self.if_gain_blocks[4],
            self.if_gain_blocks[5],
            self.adc_block
        ], layout={'width': '350px'})  # Fixed width for signal chain
        
        # Output display with flexible width
        output_section = widgets.VBox([
            widgets.HTML("<h3>Signal Chain Analysis</h3>"),
            self.output_display
        ], layout={'flex': '1'})  # Takes remaining space
        
        # Main layout
        self.layout = widgets.HBox([
            sys_params,
            signal_chain,
            output_section
        ], layout={'width': '100%'})
        
    def bind_events(self):
        """Bind update events to all widgets"""
        
        # System parameters
        for widget in [self.rf_freq, self.if_freq, self.bandwidth, self.scs, self.par_im3, self.adc_backoff, self.input_signal]:
            widget.observe(self.update_analysis, names='value')
        
        self.iq_adc.observe(self.update_analysis, names='value')
        self.dsa.observe(self.update_analysis, names='value')
        self.coarse_gain_select.observe(self.update_analysis, names='value')
        
        # Component dropdowns
        all_dropdowns = (self.rf_gain_blocks + self.coarse_gain_blocks + [self.lo_block, self.mixer_block, self.adc_block] + self.if_gain_blocks)
        
        for dropdown in all_dropdowns:
            dropdown.observe(self.update_analysis, names='value')
    
    def build_signal_chain(self):
        """Build the signal chain from current widget values"""
        
        self.system_calc.chain.clear_chain()
        
        try:
            # Store LO block separately (not in main signal chain)
            lo_block = None
            if self.lo_block.value != 'None':
                lo_block = LOBlock(f'./Components/LO Blocks/{self.lo_block.value}')
            
            # Add initial first 4 gain blocks rf_gain_blocks[0-3]
            for i, dropdown in enumerate(self.rf_gain_blocks[:4]):
                if dropdown.value != 'None':
                    component = GainBlock(f'./Components/Gain Blocks/{dropdown.value}')
                    self.system_calc.chain.add_component(component, f'gain_block_{i+1}')
            
            # Add active coarse gain block
            # FIXED: Convert 1-based widget value to 0-based array index
            active_coarse = self.coarse_gain_select.value
            active_coarse_index = active_coarse - 1  # Convert to 0-based index
            
            if active_coarse_index < len(self.coarse_gain_blocks) and self.coarse_gain_blocks[active_coarse_index].value != 'None':
                component = GainBlock(f'./Components/Gain Blocks/{self.coarse_gain_blocks[active_coarse_index].value}')
                self.system_calc.chain.add_component(component, f'coarse_gain_{active_coarse}')
            
            # Add rf_gain_block[4]
            if self.rf_gain_blocks[4].value != 'None':
                component = GainBlock(f'./Components/Gain Blocks/{self.rf_gain_blocks[4].value}')
                self.system_calc.chain.add_component(component, 'gain_block_5')
            
            # Add DSA
            dsa_component = DSA(self.dsa.value)
            self.system_calc.chain.add_component(dsa_component, 'dsa')
            
            # FIXED: This loop was incorrect - should be rf_gain_blocks[5:7] not [4:6]
            for i, dropdown in enumerate(self.rf_gain_blocks[5:7], start=6):
                if dropdown.value != 'None':
                    component = GainBlock(f'./Components/Gain Blocks/{dropdown.value}')
                    self.system_calc.chain.add_component(component, f'gain_block_{i}')
            
            # Add mixer (this marks the transition to IF frequency)
            if self.mixer_block.value != 'None':
                mixer = Mixer(f'./Components/Mixers/{self.mixer_block.value}')
                self.system_calc.chain.add_component(mixer, 'mixer')
            
            # Add IF frequency gain blocks
            for i, dropdown in enumerate(self.if_gain_blocks):
                if dropdown.value != 'None':
                    component = GainBlock(f'./Components/Gain Blocks/{dropdown.value}')
                    self.system_calc.chain.add_component(component, f'if_gain_block_{i+1}')
            
            # Add ADC
            if self.adc_block.value != 'None':
                adc = ADC(f'./Components/ADCs/{self.adc_block.value}', self.iq_adc.value)
                self.system_calc.chain.add_component(adc, 'adc')
            
            # Store the LO block reference in the system calculator
            self.system_calc.lo_block = lo_block
            
            return True
            
        except Exception as e:
            print(f"Error building signal chain: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_analysis(self, change=None):
        """Update the signal chain analysis"""
        
        with self.output_display:
            clear_output(wait=True)
            
            if not self.build_signal_chain():
                print("Error building signal chain")
                return
            
            try:
                # Calculate cascade parameters
                results = self.system_calc.calculate_cascade_parameters(
                    self.input_signal.value,
                    self.rf_freq.value,
                    self.if_freq.value,
                    self.bandwidth.value,
                    self.par_im3.value
                )
                
                # Format IPN column - blank until mixer, then show values
                ipn_formatted = []
                mixer_index = results.get('mixer_index', -1)
                
                for i, ipn_val in enumerate(results['ipn_dbc']):
                    if mixer_index >= 0 and i > mixer_index:  # After mixer block
                        ipn_formatted.append(f"{ipn_val:.1f}" if ipn_val != 0 else "")
                    else:
                        ipn_formatted.append("")  # Blank before and at mixer
                
                # Create results table
                df = pd.DataFrame({
                    'Component': results['component_names'],
                    'Signal(dBm)': [f"{p:.1f}" for p in results['signal_power']],
                    'Noise(dBm)': [f"{p:.1f}" for p in results['noise_power']],
                    'Distortion(dBm)': [f"{p:.1f}" for p in results['distortion_power']],
                    'IPN(dBc)': ipn_formatted,
                    'EVM(dB)': [f"{p:.1f}" for p in results['evm_db']]
                })
                
                print("=" * 90)
                print(df.to_string(index=False))
                
            except Exception as e:
                print(f"Error in analysis: {e}")
                import traceback
                traceback.print_exc()
    
    def display(self):
        """Display the application"""
        display(self.layout)
        # Trigger initial calculation
        self.update_analysis()

def create_signal_chain_analyzer():
    """Create and return the signal chain analyzer application"""
    return SignalChainAnalyzer()
