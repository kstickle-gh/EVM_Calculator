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
        
        # System Parameters - wider description width, narrower input width
        style = {'description_width': '150px'}
        layout = {'width': '250px'}
        
        self.rf_freq = widgets.FloatText(value=8.0, description='RF Freq (GHz):', 
                                        style=style, layout=layout)
        self.if_freq = widgets.FloatText(value=0.5, description='IF Freq (GHz):', 
                                        style=style, layout=layout)
        self.bandwidth = widgets.FloatText(value=100.0, description='Bandwidth (MHz):', 
                                          style=style, layout=layout)
        self.scs = widgets.FloatText(value=15.0, description='SCS (kHz):', 
                                    style=style, layout=layout)
        self.par_im3 = widgets.FloatText(value=7.0, description='PAR IM3 (dB):', 
                                        style=style, layout=layout)
        self.iq_adc = widgets.Checkbox(value=False, description='IQ ADC?', 
                                      style=style)
        self.adc_backoff = widgets.FloatText(value=12.0, description='ADC Backoff (dB):', 
                                            style=style, layout=layout)
        
        # Signal Chain Components - consistent styling
        chain_style = {'description_width': '120px'}
        chain_layout = {'width': '250px'}
        
        self.input_signal = widgets.FloatText(value=0.0, description='RF Input (dBm):', 
                                             style=chain_style, layout=chain_layout)
        
        # Get available CSV files
        gain_files = ['None'] + get_available_components('Gain Blocks')
        mixer_files = ['None'] + get_available_components('Mixers')
        lo_files = ['None'] + get_available_components('LO Blocks')
        adc_files = ['None'] + get_available_components('ADCs')
        
        # Gain Blocks (4 regular + 4 coarse)
        self.gain_blocks = []
        for i in range(4):
            self.gain_blocks.append(widgets.Dropdown(options=gain_files, value='None',
                                                    description=f'Gain Block {i+1}:',
                                                    style=chain_style, layout=chain_layout))
        
        # Coarse Gain Blocks (radio button selection)
        self.coarse_gain_blocks = []
        for i in range(4):
            self.coarse_gain_blocks.append(widgets.Dropdown(options=gain_files, value='None',
                                                           description=f'Coarse {i}:',
                                                           style={'description_width': '120px'}, 
                                                           layout={'width': '250px'}))
        self.coarse_gain_select = widgets.RadioButtons(
            options=[0, 1, 2, 3], value=0, description='Active Coarse:',
            style={'description_width': '120px'}
        )
        
        # Additional components
        self.gain_block_5 = widgets.Dropdown(options=gain_files, value='None',
                                            description='Gain Block 5:',
                                            style=chain_style, layout=chain_layout)
        self.dsa = widgets.IntSlider(value=0, min=0, max=30, description='DSA (dB):',
                                    style=chain_style, layout=chain_layout)
        
        self.gain_blocks_6_7 = []
        for i in range(2):
            self.gain_blocks_6_7.append(widgets.Dropdown(options=gain_files, value='None',
                                                        description=f'Gain Block {i+6}:',
                                                        style=chain_style, layout=chain_layout))
        
        self.lo_block = widgets.Dropdown(options=lo_files, value='None',
                                        description='LO Block:',
                                        style=chain_style, layout=chain_layout)
        self.mixer_block = widgets.Dropdown(options=mixer_files, value='None',
                                           description='Mixer Block:',
                                           style=chain_style, layout=chain_layout)
        
        # Post-mixer gain blocks (6 blocks at IF frequency)
        self.if_gain_blocks = []
        for i in range(6):
            self.if_gain_blocks.append(widgets.Dropdown(options=gain_files, value='None',
                                                       description=f'IF Gain {i+1}:',
                                                       style=chain_style, layout=chain_layout))
        
        self.adc_block = widgets.Dropdown(options=adc_files, value='None',
                                         description='ADC Block:',
                                         style=chain_style, layout=chain_layout)
        
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
            widgets.HTML("<b>Coarse Gain Blocks</b>"),
            self.coarse_gain_select,
            self.coarse_gain_blocks[0],
            self.coarse_gain_blocks[1], 
            self.coarse_gain_blocks[2],
            self.coarse_gain_blocks[3]
        ])
        
        # Signal chain column
        signal_chain = widgets.VBox([
            widgets.HTML("<h3>Signal Chain</h3>"),
            self.input_signal,
            self.gain_blocks[0],
            self.gain_blocks[1],
            self.gain_blocks[2],
            self.gain_blocks[3],
            coarse_gain_section,
            self.gain_block_5,
            self.dsa,
            self.gain_blocks_6_7[0],
            self.gain_blocks_6_7[1],
            self.lo_block,
            self.mixer_block,
            widgets.HTML("<b>--- IF Frequency Components ---</b>"),
            self.if_gain_blocks[0],
            self.if_gain_blocks[1],
            self.if_gain_blocks[2],
            self.if_gain_blocks[3],
            self.if_gain_blocks[4],
            self.if_gain_blocks[5],
            self.adc_block
        ], layout={'width': '300px'})  # Fixed width for signal chain
        
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
        for widget in [self.rf_freq, self.if_freq, self.bandwidth, self.scs, 
                      self.par_im3, self.adc_backoff, self.input_signal]:
            widget.observe(self.update_analysis, names='value')
        
        self.iq_adc.observe(self.update_analysis, names='value')
        self.dsa.observe(self.update_analysis, names='value')
        self.coarse_gain_select.observe(self.update_analysis, names='value')
        
        # Component dropdowns
        all_dropdowns = (self.gain_blocks + self.coarse_gain_blocks + 
                        [self.gain_block_5] + self.gain_blocks_6_7 + 
                        [self.lo_block, self.mixer_block, self.adc_block] + 
                        self.if_gain_blocks)
        
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
            
            # Add initial gain blocks
            for i, dropdown in enumerate(self.gain_blocks):
                if dropdown.value != 'None':
                    component = GainBlock(f'./Components/Gain Blocks/{dropdown.value}')
                    self.system_calc.chain.add_component(component, f'gain_block_{i+1}')
            
            # Add active coarse gain block
            active_coarse = self.coarse_gain_select.value
            if self.coarse_gain_blocks[active_coarse].value != 'None':
                component = GainBlock(f'./Components/Gain Blocks/{self.coarse_gain_blocks[active_coarse].value}')
                self.system_calc.chain.add_component(component, f'coarse_gain_{active_coarse}')
            
            # Add gain block 5
            if self.gain_block_5.value != 'None':
                component = GainBlock(f'./Components/Gain Blocks/{self.gain_block_5.value}')
                self.system_calc.chain.add_component(component, 'gain_block_5')
            
            # Add DSA
            dsa_component = DSA(self.dsa.value)
            self.system_calc.chain.add_component(dsa_component, 'dsa')
            
            # Add gain blocks 6-7
            for i, dropdown in enumerate(self.gain_blocks_6_7):
                if dropdown.value != 'None':
                    component = GainBlock(f'./Components/Gain Blocks/{dropdown.value}')
                    self.system_calc.chain.add_component(component, f'gain_block_{i+6}')
            
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
                    'Signal Power (dBm)': [f"{p:.1f}" for p in results['signal_power']],
                    'Noise Power (dBm)': [f"{p:.1f}" for p in results['noise_power']],
                    'Distortion Power (dBm)': [f"{p:.1f}" for p in results['distortion_power']],
                    'IPN (dBc)': ipn_formatted,
                    'EVM (dB)': [f"{p:.1f}" for p in results['evm_db']]
                })
                
                print("Signal Chain Analysis:")
                print("=" * 100)
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
