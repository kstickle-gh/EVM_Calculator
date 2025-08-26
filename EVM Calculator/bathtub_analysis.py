# bathtub_analysis.py
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

def calculate_evm_bathtub_curves(analyzer):
    """Calculate EVM bathtub curves for all coarse gain states"""
    
    # Power sweep range
    power_range = np.linspace(-45, 30, 151)  # 0.5 dB steps
    
    # Store original coarse gain setting
    original_coarse_gain = analyzer.coarse_gain_select.value
    
    # Initialize results storage
    evm_curves = {}
    coarse_gain_labels = []
    
    print("Calculating EVM bathtub curves for all coarse gain states...")
    print("=" * 60)
    
    # Calculate curve for each coarse gain state
    for coarse_state in range(4):
        # Check if this coarse gain state has a selected component
        if analyzer.coarse_gain_blocks[coarse_state].value == 'None':
            print(f"Coarse Gain {coarse_state}: No component selected - skipping")
            continue
            
        # Set coarse gain state
        analyzer.coarse_gain_select.value = coarse_state
        
        # Get the component name for labeling
        component_name = analyzer.coarse_gain_blocks[coarse_state].value
        coarse_gain_labels.append(f"Coarse {coarse_state}: {component_name}")
        
        print(f"Calculating Coarse Gain {coarse_state}: {component_name}")
        
        # Initialize EVM array for this coarse gain state
        evm_total = np.zeros(len(power_range))
        
        # Calculate EVM for each input power level
        for i, input_power in enumerate(power_range):
            # Update input signal level
            analyzer.input_signal.value = input_power
            
            # Build signal chain with current settings
            if analyzer.build_signal_chain():
                # Calculate cascade parameters
                results = analyzer.system_calc.calculate_cascade_parameters(
                    input_power,
                    analyzer.rf_freq.value,
                    analyzer.if_freq.value,
                    analyzer.bandwidth.value,
                    analyzer.par_im3.value
                )
                
                # Get final EVM
                evm_total[i] = results['evm_db'][-1]
            else:
                evm_total[i] = 0  # Error case
        
        # Store results
        evm_curves[coarse_state] = {
            'power': power_range.copy(),
            'evm': evm_total.copy(),
            'label': coarse_gain_labels[-1]
        }
        
        print(f"  Completed: EVM range {np.min(evm_total[evm_total > -200]):.1f} to {np.max(evm_total[evm_total < 0]):.1f} dB")
    
    # Restore original settings
    analyzer.coarse_gain_select.value = original_coarse_gain
    analyzer.input_signal.value = 0.0  # Reset to default
    
    return evm_curves

def plot_evm_bathtub_curves(evm_curves, bandwidth_mhz):
    """Plot the EVM bathtub curves"""
    
    if len(evm_curves) == 0:
        print("No coarse gain states configured - cannot generate plots")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for different curves
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot each coarse gain curve
    for i, (coarse_state, data) in enumerate(evm_curves.items()):
        power = data['power']
        evm = data['evm']
        label = data['label']
        
        # Only plot valid EVM points (avoid very low values that indicate no signal)
        valid_mask = (evm > -200) & (evm < 10)
        
        if np.any(valid_mask):
            plt.plot(power[valid_mask], evm[valid_mask], 
                    color=colors[i % len(colors)], 
                    linewidth=2.5, 
                    label=label,
                    marker='o' if len(power[valid_mask]) < 50 else None,
                    markersize=3)
    
    # Formatting
    plt.xlabel('Reference Level (dBm)', fontsize=12, fontweight='bold')
    plt.ylabel(f'EVM (dB) Bandwidth {bandwidth_mhz:.0f} MHz', fontsize=12, fontweight='bold')
    plt.title('EVM Bathtub Curves - Coarse Gain Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(fontsize=10, loc='best')
    
    # Set axis limits
    plt.xlim(-45, 30)
    
    # Find reasonable EVM limits based on data
    all_evm_values = []
    for data in evm_curves.values():
        valid_evm = data['evm'][(data['evm'] > -200) & (data['evm'] < 10)]
        if len(valid_evm) > 0:
            all_evm_values.extend(valid_evm)
    
    if len(all_evm_values) > 0:
        evm_min = max(np.min(all_evm_values) - 5, -80)  # Don't go below -80 dB
        evm_max = min(np.max(all_evm_values) + 5, 5)    # Don't go above +5 dB
        plt.ylim(evm_min, evm_max)
    else:
        plt.ylim(-60, 0)
    
    # Add some reference lines
    plt.axhline(y=-30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=-40, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.show()

def generate_bathtub_analysis(analyzer):
    """Main function to generate bathtub curve analysis"""
    
    if analyzer is None:
        print("Error: Signal chain analyzer not provided.")
        return
    
    print("EVM Bathtub Curve Analysis")
    print("=" * 50)
    print(f"System Settings:")
    print(f"  RF Frequency: {analyzer.rf_freq.value} GHz")
    print(f"  IF Frequency: {analyzer.if_freq.value} GHz") 
    print(f"  Bandwidth: {analyzer.bandwidth.value} MHz")
    print(f"  SCS: {analyzer.scs.value} kHz")
    print(f"  PAR IM3: {analyzer.par_im3.value} dB")
    print(f"  Input Power Sweep: -45 to +30 dBm")
    print()
    
    # Calculate curves
    evm_curves = calculate_evm_bathtub_curves(analyzer)
    
    if len(evm_curves) > 0:
        print(f"\nGenerated {len(evm_curves)} EVM curves")
        print("Plotting results...")
        
        # Plot the curves
        plot_evm_bathtub_curves(evm_curves, analyzer.bandwidth.value)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 40)
        for coarse_state, data in evm_curves.items():
            valid_evm = data['evm'][(data['evm'] > -200) & (data['evm'] < 0)]
            if len(valid_evm) > 0:
                best_evm = np.min(valid_evm)
                print(f"{data['label']}: Best EVM = {best_evm:.1f} dB")
            else:
                print(f"{data['label']}: No valid EVM data")
    else:
        print("No coarse gain states configured. Please select components for coarse gain blocks.")

def create_bathtub_analysis_widget(analyzer):
    """Create an interactive widget for bathtub analysis"""
    import ipywidgets as widgets
    
    # Create button to run analysis
    run_button = widgets.Button(
        description='Generate EVM Bathtub Curves',
        button_style='primary',
        layout={'width': '250px', 'height': '40px'}
    )
    
    # Output area for results
    output = widgets.Output()
    
    def on_button_click(b):
        with output:
            clear_output(wait=True)
            generate_bathtub_analysis(analyzer)
    
    run_button.on_click(on_button_click)
    
    # Layout
    widget_layout = widgets.VBox([
        widgets.HTML("<h3>EVM Bathtub Curve Analysis</h3>"),
        widgets.HTML("Generate EVM vs input power curves for all configured coarse gain states."),
        run_button,
        output
    ])
    
    return widget_layout