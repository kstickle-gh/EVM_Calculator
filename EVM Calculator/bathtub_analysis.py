# bathtub_analysis.py
import matplotlib.pyplot as plt
import numpy as np

def calculate_evm_bathtub_curves(analyzer, pmin=-45, pmax=30, dsa_step=5):
    """Calculate EVM bathtub curves with configurable parameters"""
    
    # Configurable power sweep
    power_range = np.linspace(pmin, pmax, int((pmax - pmin) + 1))  # 1 dB steps
    
    # Store original settings
    original_coarse_gain = analyzer.coarse_gain_select.value
    original_dsa = analyzer.dsa.value
    
    # Initialize results storage
    evm_curves = {}
    
    # Calculate curve for each coarse gain state
    for coarse_state in analyzer.coarse_gain_select.options:
        # Convert to 0-based index for accessing the coarse_gain_blocks array
        coarse_index = coarse_state - 1
        
        if analyzer.coarse_gain_blocks[coarse_index].value == 'None':
            continue
            
        analyzer.coarse_gain_select.value = coarse_state
        component_name = analyzer.coarse_gain_blocks[coarse_index].value
        
        print(f"Calculating Coarse Gain {coarse_state}: {component_name}")
        
        evm_curves[coarse_state] = {}
        
        # Step the DSA with configurable step size
        for dsa_setting in range(0, 31, dsa_step):
            analyzer.dsa.value = dsa_setting
            
            valid_powers = []
            valid_evms = []
            
            # Early termination flags
            consecutive_clips = 0
            max_consecutive_clips = 2  # Stop after 2 consecutive clips
            
            for i, input_power in enumerate(power_range):
                analyzer.input_signal.value = input_power
                
                if analyzer.build_signal_chain():
                    results = analyzer.system_calc.calculate_cascade_parameters(
                        input_power,
                        analyzer.rf_freq.value,
                        analyzer.if_freq.value,
                        analyzer.bandwidth.value,
                        analyzer.par_im3.value
                    )
                    
                    adc_power_dbm = results['total_gain'] + input_power
                    
                    if adc_power_dbm + analyzer.adc_backoff.value <= results['adcfs'][-1]:
                        valid_powers.append(input_power)
                        valid_evms.append(results['evm_db'][-1])
                        consecutive_clips = 0  # Reset counter
                    else:
                        consecutive_clips += 1
                        if consecutive_clips >= max_consecutive_clips:
                            break  # Stop this DSA setting
                else:
                    consecutive_clips += 1
                    if consecutive_clips >= max_consecutive_clips:
                        break
            
            # Store results
            evm_curves[coarse_state][dsa_setting] = {
                'power': np.array(valid_powers),
                'evm': np.array(valid_evms),
                'label': f"Coarse {coarse_state}: {component_name}",
                'dsa': dsa_setting
            }
    
    # Restore settings
    analyzer.coarse_gain_select.value = original_coarse_gain
    analyzer.dsa.value = original_dsa
    analyzer.input_signal.value = 0.0
    
    return evm_curves

def plot_evm_bathtub_curves(evm_curves, bandwidth_mhz, pmin, pmax, dsa_step):
    """Plot the EVM bathtub curves with DSA stepping and minimum EVM envelope"""
    
    if len(evm_curves) == 0:
        print("No coarse gain states configured - cannot generate plots")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Color palette for different curves
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Collect all data points for minimum EVM calculation
    all_data_points = []
    
    # Plot each coarse gain state
    for i, (coarse_state, dsa_data) in enumerate(evm_curves.items()):
        color = colors[i % len(colors)]
        
        # Plot each DSA setting for this coarse gain state
        for dsa_setting, data in dsa_data.items():
            power = data['power']
            evm = data['evm']
            label = data['label']
            
            # Only plot if we have valid data
            if len(power) > 0 and len(evm) > 0:
                # Filter out any remaining invalid EVM points
                valid_mask = (evm > -200) & (evm < 10) & ~np.isnan(evm)
                
                if np.any(valid_mask):
                    # Store valid data points for minimum EVM calculation
                    for p, e in zip(power[valid_mask], evm[valid_mask]):
                        all_data_points.append((p, e, coarse_state, dsa_setting))
                    
                    # Determine line style and label
                    if dsa_setting == 0:
                        # Solid line for 0dB DSA with label
                        linestyle = '-'
                        linewidth = 2.5
                        plot_label = label
                        alpha = 1.0
                    else:
                        # Dotted line for other DSA settings, no label
                        linestyle = ':'
                        linewidth = 1.5
                        plot_label = None
                        alpha = 0.7
                    
                    plt.plot(power[valid_mask], evm[valid_mask], 
                            color=color,
                            linestyle=linestyle,
                            linewidth=linewidth,
                            label=plot_label,
                            alpha=alpha)
                    
                    # Add clip point marker only for 0dB DSA (solid lines)
                    if dsa_setting == 0 and np.sum(valid_mask) > 0:
                        last_idx = np.where(valid_mask)[0][-1]
                        plt.plot(power[last_idx], evm[last_idx], 
                                marker='s', markersize=8, 
                                color=color, 
                                markerfacecolor='white',
                                markeredgewidth=2)
    
    # Calculate and plot minimum EVM envelope
    if all_data_points:
        # Convert to numpy array for easier processing
        all_data = np.array(all_data_points, dtype=object)
        powers = np.array([point[0] for point in all_data_points])
        evms = np.array([point[1] for point in all_data_points])
        
        # Create power grid for interpolation
        power_min = np.min(powers)
        power_max = np.max(powers)
        power_grid = np.linspace(power_min, power_max, 200)  # Fine grid for smooth curve
        
        min_evm_envelope = []
        optimal_settings = []  # Track which coarse/DSA gives minimum EVM
        
        for target_power in power_grid:
            # Find all data points within a small window around target power
            power_tolerance = 0.5  # dB tolerance for power matching
            nearby_mask = np.abs(powers - target_power) <= power_tolerance
            
            if np.any(nearby_mask):
                nearby_evms = evms[nearby_mask]
                nearby_indices = np.where(nearby_mask)[0]
                
                # Find minimum EVM and its corresponding settings
                min_idx = np.argmin(nearby_evms)
                min_evm = nearby_evms[min_idx]
                original_idx = nearby_indices[min_idx]
                
                min_evm_envelope.append(min_evm)
                optimal_settings.append({
                    'power': target_power,
                    'evm': min_evm,
                    'coarse_state': all_data_points[original_idx][2],
                    'dsa_setting': all_data_points[original_idx][3]
                })
            else:
                min_evm_envelope.append(np.nan)
                optimal_settings.append(None)
        
        # Remove NaN values for plotting
        valid_envelope_mask = ~np.isnan(min_evm_envelope)
        if np.any(valid_envelope_mask):
            plt.plot(power_grid[valid_envelope_mask], 
                    np.array(min_evm_envelope)[valid_envelope_mask],
                    color='black',
                    linewidth=3,
                    linestyle='-',
                    label='Minimum EVM Envelope',
                    zorder=10)  # Plot on top
            
            # Add markers at key points of the envelope
            envelope_powers = power_grid[valid_envelope_mask]
            envelope_evms = np.array(min_evm_envelope)[valid_envelope_mask]
            
            # Sample every 10th point for markers to avoid clutter
            marker_indices = np.arange(0, len(envelope_powers), 10)
            plt.plot(envelope_powers[marker_indices], 
                    envelope_evms[marker_indices],
                    'ko', markersize=4, zorder=11)
    
    # Formatting
    plt.xlabel('Reference Level (dBm)', fontsize=12, fontweight='bold')
    plt.ylabel(f'EVM (dB) Bandwidth {bandwidth_mhz:.0f} MHz', fontsize=12, fontweight='bold')
    plt.title(f'EVM Bathtub Curves - Coarse Gain with DSA Stepping (Step: {dsa_step}dB)\n(Solid=0dB DSA, Dotted={dsa_step}-30dB DSA, Black=Min EVM Envelope)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Clean legend with coarse gain states and minimum envelope
    plt.legend(fontsize=10, loc='best')
    
    # Set axis limits based on input parameters
    plt.xlim(pmin, pmax)
    
    # Find reasonable EVM limits based on all data
    all_evm_values = []
    for dsa_data in evm_curves.values():
        for data in dsa_data.values():
            if len(data['evm']) > 0:
                valid_evm = data['evm'][(data['evm'] > -200) & (data['evm'] < 10)]
                if len(valid_evm) > 0:
                    all_evm_values.extend(valid_evm)
    
    evm_min = -70  # Don't go below -70 dB
    evm_max = -20   # Don't go above -20 dB
    plt.ylim(evm_min, evm_max)
    
    # Add some reference lines
    plt.axhline(y=-30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=-40, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.show()
    
    # Print information about the minimum EVM envelope
    if all_data_points and 'optimal_settings' in locals():
        print("\nMinimum EVM Envelope Analysis:")
        print("-" * 50)
        
        # Find the overall best EVM point
        valid_optimal = [opt for opt in optimal_settings if opt is not None]
        if valid_optimal:
            best_point = min(valid_optimal, key=lambda x: x['evm'])
            print(f"Overall Best EVM: {best_point['evm']:.1f} dB at {best_point['power']:.1f} dBm")
            print(f"  Optimal Settings: Coarse {best_point['coarse_state']}, DSA {best_point['dsa_setting']} dB")
            
            # Show EVM improvement
            print(f"\nEVM Envelope Statistics:")
            envelope_evms_valid = [opt['evm'] for opt in valid_optimal]
            print(f"  Best EVM: {min(envelope_evms_valid):.1f} dB")
            print(f"  Worst EVM: {max(envelope_evms_valid):.1f} dB")
            print(f"  EVM Range: {max(envelope_evms_valid) - min(envelope_evms_valid):.1f} dB")

def generate_bathtub_analysis(analyzer, pmin=-45, pmax=30, dsa_step=5):
    """Main function to generate bathtub curve analysis with configurable parameters"""
    
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
    print(f"  Input Power Sweep: {pmin} to {pmax} dBm")
    print(f"  DSA Step Size: {dsa_step} dB")
    print()
    
    # Calculate curves
    evm_curves = calculate_evm_bathtub_curves(analyzer, pmin, pmax, dsa_step)
    
    if len(evm_curves) > 0:
        # Plot the curves
        plot_evm_bathtub_curves(evm_curves, analyzer.bandwidth.value, pmin, pmax, dsa_step)
        
        # Print summary statistics
        print("\nSummary Statistics (0dB DSA baseline):")
        print("-" * 40)
        for coarse_state, dsa_data in evm_curves.items():
            # Get the 0dB DSA data for summary statistics
            if 0 in dsa_data:
                baseline_data = dsa_data[0]  # 0dB DSA setting
                valid_evm = baseline_data['evm'][(baseline_data['evm'] > -200) & (baseline_data['evm'] < 0)]
                if len(valid_evm) > 0:
                    best_evm = np.min(valid_evm)
                    power_range = baseline_data['power']
                    if len(power_range) > 0:
                        power_min = np.min(power_range)
                        power_max = np.max(power_range)
                        print(f"{baseline_data['label']}: Best EVM = {best_evm:.1f} dB, Power Range = {power_min:.1f} to {power_max:.1f} dBm")
                    else:
                        print(f"{baseline_data['label']}: Best EVM = {best_evm:.1f} dB, No valid power range")
                else:
                    print(f"{baseline_data['label']}: No valid EVM data")
            else:
                print(f"Coarse {coarse_state}: No baseline (0dB DSA) data available")
