# multi_frequency_envelope.py - Multi-Frequency EVM Envelope Analysis
import matplotlib.pyplot as plt
import numpy as np

def calculate_evm_envelope_for_frequency(analyzer, rf_freq, if_freq, pmin=-45, pmax=30, dsa_step=5):
    """Calculate minimum EVM envelope for a specific RF/IF frequency combination"""
    
    # Configurable power sweep
    power_range = np.linspace(pmin, pmax, int((pmax - pmin) + 1))  # 1 dB steps
    
    # Store original settings
    original_rf_freq = analyzer.rf_freq.value
    original_if_freq = analyzer.if_freq.value
    original_coarse_gain = analyzer.coarse_gain_select.value
    original_dsa = analyzer.dsa.value
    original_input_power = analyzer.input_signal.value
    
    # Set the frequencies
    analyzer.rf_freq.value = rf_freq
    analyzer.if_freq.value = if_freq
    
    # Collect all data points for minimum EVM calculation
    all_data_points = []
    
    print(f"  Calculating RF={rf_freq}GHz, IF={if_freq}GHz...")
    
    # Calculate for each coarse gain state
    for coarse_state in analyzer.coarse_gain_select.options:
        # Convert to 0-based index for accessing the coarse_gain_blocks array
        coarse_index = coarse_state - 1
        
        if coarse_index >= len(analyzer.coarse_gain_blocks) or analyzer.coarse_gain_blocks[coarse_index].value == 'None':
            continue
            
        analyzer.coarse_gain_select.value = coarse_state
        
        # Step the DSA with configurable step size
        for dsa_setting in range(0, 31, dsa_step):
            analyzer.dsa.value = dsa_setting
            
            # Early termination flags
            consecutive_clips = 0
            max_consecutive_clips = 2
            
            for input_power in power_range:
                analyzer.input_signal.value = input_power
                
                if analyzer.build_signal_chain():
                    try:
                        results = analyzer.system_calc.calculate_cascade_parameters(
                            input_power,
                            analyzer.rf_freq.value,
                            analyzer.if_freq.value,
                            analyzer.bandwidth.value,
                            analyzer.par_im3.value
                        )
                        
                        adc_power_dbm = results['total_gain'] + input_power
                        
                        if adc_power_dbm + analyzer.adc_backoff.value <= results['adcfs'][-1]:
                            evm_db = results['evm_db'][-1]
                            if evm_db > -200 and evm_db < 10 and not np.isnan(evm_db):
                                all_data_points.append((input_power, evm_db, coarse_state, dsa_setting))
                            consecutive_clips = 0
                        else:
                            consecutive_clips += 1
                            if consecutive_clips >= max_consecutive_clips:
                                break
                    except Exception:
                        consecutive_clips += 1
                        if consecutive_clips >= max_consecutive_clips:
                            break
                else:
                    consecutive_clips += 1
                    if consecutive_clips >= max_consecutive_clips:
                        break
    
    # Calculate minimum EVM envelope
    envelope_powers = []
    envelope_evms = []
    
    if all_data_points:
        # Convert to numpy arrays
        powers = np.array([point[0] for point in all_data_points])
        evms = np.array([point[1] for point in all_data_points])
        
        # Create power grid for envelope
        power_min = np.min(powers)
        power_max = np.max(powers)
        power_grid = np.linspace(power_min, power_max, 150)
        
        for target_power in power_grid:
            # Find all data points within tolerance
            power_tolerance = 0.5  # dB tolerance
            nearby_mask = np.abs(powers - target_power) <= power_tolerance
            
            if np.any(nearby_mask):
                nearby_evms = evms[nearby_mask]
                min_evm = np.min(nearby_evms)
                envelope_powers.append(target_power)
                envelope_evms.append(min_evm)
    
    # Restore original settings
    analyzer.rf_freq.value = original_rf_freq
    analyzer.if_freq.value = original_if_freq
    analyzer.coarse_gain_select.value = original_coarse_gain
    analyzer.dsa.value = original_dsa
    analyzer.input_signal.value = original_input_power
    
    return np.array(envelope_powers), np.array(envelope_evms)

def calculate_multi_frequency_envelopes(analyzer, frequency_combinations, pmin=-45, pmax=30, dsa_step=5):
    """Calculate EVM envelopes for multiple RF/IF frequency combinations"""

    print("=" * 70)
    
    envelopes = {}
    
    for i, (rf_freq, if_freq) in enumerate(frequency_combinations):
        try:
            powers, evms = calculate_evm_envelope_for_frequency(analyzer, rf_freq, if_freq, pmin, pmax, dsa_step)
            
            if len(powers) > 0 and len(evms) > 0:
                envelopes[f"RF{rf_freq}GHz_IF{if_freq}GHz"] = {
                    'powers': powers,
                    'evms': evms,
                    'rf_freq': rf_freq,
                    'if_freq': if_freq,
                    'label': f"RF={rf_freq}GHz, IF={if_freq}GHz"
                }
                # print(f"    ✓ Completed: {len(powers)} points")
            else:
                print(f"    ✗ No valid data")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # print(f"\nSuccessfully calculated {len(envelopes)} envelope curves")
    return envelopes

def plot_multi_frequency_envelopes(envelopes, bandwidth_mhz, pmin, pmax, dsa_step):
    """Plot multiple EVM envelopes on the same plot"""
    
    if len(envelopes) == 0:
        print("No envelope data to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Color palette - use a colormap for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(envelopes)))
    
    # Plot each envelope
    for i, (key, data) in enumerate(envelopes.items()):
        powers = data['powers']
        evms = data['evms']
        label = data['label']
        
        plt.plot(powers, evms,
                color=colors[i],
                linewidth=2.5,
                marker='o',
                markersize=3,
                label=label,
                alpha=0.8)
    
    # Formatting
    plt.xlabel('Reference Level (dBm)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Minimum EVM (dB) Bandwidth {bandwidth_mhz:.0f} MHz', fontsize=12, fontweight='bold')
    plt.title(f'Minimum EVM Envelopes - Multiple RF/IF Frequency Combinations\n(Power: {pmin} to {pmax} dBm, DSA Step: {dsa_step} dB)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits based on input parameters
    plt.xlim(pmin, pmax)
    
    # Find reasonable EVM limits
    all_evm_values = []
    for data in envelopes.values():
        all_evm_values.extend(data['evms'])
    
    if all_evm_values:
        evm_min = max(min(all_evm_values) - 2, -70)  # Don't go below -70 dB
        evm_max = min(max(all_evm_values) + 2, -15)   # Don't go above -15 dB
        plt.ylim(evm_min, evm_max)
    
    # Add reference lines
    plt.axhline(y=-30, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='-30 dB ref')
    plt.axhline(y=-40, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='-40 dB ref')
    
    # Legend
    plt.legend(fontsize=9, loc='best', ncol=2)
    
    plt.tight_layout()
    plt.show()

def analyze_envelope_performance(envelopes):
    """Analyze and compare envelope performance across frequencies"""
    
    print("\nMulti-Frequency Envelope Performance Analysis:")
    print("=" * 60)
    
    # Performance metrics for each envelope
    performance_data = []
    
    for key, data in envelopes.items():
        evms = data['evms']
        powers = data['powers']
        rf_freq = data['rf_freq']
        if_freq = data['if_freq']
        
        if len(evms) > 0:
            best_evm = np.min(evms)
            worst_evm = np.max(evms)
            evm_range = worst_evm - best_evm
            
            # Find power range where EVM < -30 dB
            good_evm_mask = evms <= -30
            if np.any(good_evm_mask):
                good_power_range = np.max(powers[good_evm_mask]) - np.min(powers[good_evm_mask])
            else:
                good_power_range = 0
            
            performance_data.append({
                'rf_freq': rf_freq,
                'if_freq': if_freq,
                'best_evm': best_evm,
                'worst_evm': worst_evm,
                'evm_range': evm_range,
                'power_span': np.max(powers) - np.min(powers),
                'good_power_range': good_power_range,
                'label': data['label']
            })
    
    # Sort by RF frequency for organized display
    performance_data.sort(key=lambda x: x['rf_freq'])
    
    print(f"{'Frequency Combination':<25} {'Best EVM':<10} {'EVM Range':<12} {'Power Span':<12} {'>-30dB Range':<12}")
    print("-" * 75)
    
    for perf in performance_data:
        print(f"{perf['label']:<25} {perf['best_evm']:>7.1f} dB {perf['evm_range']:>9.1f} dB "
              f"{perf['power_span']:>9.1f} dB {perf['good_power_range']:>9.1f} dB")
    
    # Find best overall performance
    if performance_data:
        best_overall = min(performance_data, key=lambda x: x['best_evm'])
        widest_range = max(performance_data, key=lambda x: x['good_power_range'])
        
        print(f"\nBest Overall EVM: {best_overall['label']} ({best_overall['best_evm']:.1f} dB)")
        print(f"Widest >-30dB Range: {widest_range['label']} ({widest_range['good_power_range']:.1f} dB span)")

def display_multi_frequency_envelope(analyzer, frequency_combinations, pmin=-45, pmax=30, dsa_step=5):
    """Main function to generate and display multi-frequency envelope analysis"""
    
    if analyzer is None:
        print("Error: Signal chain analyzer not provided.")
        return
    
    print("Multi-Frequency EVM Envelope Analysis")
    print("=" * 50)
    print(f"System Settings:")
    print(f"  Bandwidth: {analyzer.bandwidth.value} MHz")
    print(f"  SCS: {analyzer.scs.value} kHz")
    print(f"  PAR IM3: {analyzer.par_im3.value} dB")
    print(f"  Input Power Sweep: {pmin} to {pmax} dBm")
    print(f"  DSA Step Size: {dsa_step} dB")
    print()
    
    # Calculate envelopes
    envelopes = calculate_multi_frequency_envelopes(analyzer, frequency_combinations, pmin, pmax, dsa_step)
    
    if len(envelopes) > 0:

        # Plot the envelopes
        plot_multi_frequency_envelopes(envelopes, analyzer.bandwidth.value, pmin, pmax, dsa_step)
        
        # Analyze performance
        analyze_envelope_performance(envelopes)
                
    else:
        print("No envelope data generated. Check system configuration.")