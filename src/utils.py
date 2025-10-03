"""
Utility functions for RNN working memory experiments.

This module contains helper functions for visualization, analysis, and
experimental setup for working memory capacity studies.

Authors: Vivian Kang, Jameel Saeb, Chetanya Singh, Ah-Young Moon
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_eigenvalue_spectrum(weight_matrix, title="Weight Matrix Analysis"):
    """
    Analyze and visualize the eigenvalue spectrum of a weight matrix.
    
    Args:
        weight_matrix (np.ndarray): Square weight matrix to analyze
        title (str): Title for the plots
    """
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(weight_matrix)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot weight matrix sample
    show_count = min(50, weight_matrix.shape[0])  # Show subset for readability
    im = axes[0].imshow(weight_matrix[:show_count, :show_count], 
                       cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Sample from Weight Matrix')
    axes[0].set_xlabel('Presynaptic Neuron')
    axes[0].set_ylabel('Postsynaptic Neuron')
    plt.colorbar(im, ax=axes[0])
    
    # Plot eigenvalue spectrum
    axes[1].scatter(np.real(eigenvalues), np.imag(eigenvalues), 
                   alpha=0.6, s=20)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.5, 
                label='Unit Circle')
    
    axes[1].set_title('Eigenvalue Spectrum')
    axes[1].set_xlabel('Real Component')
    axes[1].set_ylabel('Imaginary Component')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    max_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"Maximum eigenvalue magnitude: {max_eigenvalue:.3f}")
    print(f"Number of eigenvalues > 1: {np.sum(np.abs(eigenvalues) > 1)}")


def visualize_network_activity(firing_rates, input_stream, model_params, 
                              title="Network Activity"):
    """
    Visualize RNN activity patterns and input sequence.
    
    Args:
        firing_rates (np.ndarray): Neural activity over time (N, time_steps)
        input_stream (np.ndarray): Input sequence over time
        model_params (dict): Model parameters for timing conversion
        title (str): Plot title
    """
    # Create time axis
    simulation_time = (np.arange(len(input_stream)) * model_params['dt'] - 
                      model_params['burnIn'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot input sequence
    axes[0].plot(simulation_time, input_stream, 'b-', linewidth=2)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Input Value')
    axes[0].set_title('Input Sequence')
    axes[0].grid(True, alpha=0.3)
    
    # Plot network activity heatmap
    extents = [simulation_time[0], simulation_time[-1], 
               0, model_params['N']]
    im = axes[1].imshow(firing_rates, aspect='auto', extent=extents, 
                       cmap='viridis', interpolation='nearest')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Neuron Index')
    axes[1].set_title('Network Firing Rates')
    plt.colorbar(im, ax=axes[1], label='Firing Rate')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_input_encoding_example(input_onehot, input_stream, model_params,
                               title="Input Encoding Example"):
    """
    Visualize an example of input encoding scheme.
    
    Args:
        input_onehot (np.ndarray): One-hot encoded input matrix
        input_stream (np.ndarray): Integer input sequence  
        model_params (dict): Model parameters
        title (str): Plot title
    """
    # Skip burn-in period for visualization
    burn_in_steps = int(model_params['burnIn'] / model_params['dt'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot integer sequence
    time_axis = (np.arange(len(input_stream) - burn_in_steps) * 
                model_params['dt'])
    axes[0].plot(time_axis, input_stream[burn_in_steps:], 'o-')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Input Value')
    axes[0].set_title('Input Sequence (Integer Values)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot one-hot encoding
    im = axes[1].imshow(input_onehot[:, burn_in_steps:], 
                       aspect='auto', cmap='binary')
    axes[1].set_xlabel('Time (timesteps)')
    axes[1].set_ylabel('Input Channel')
    axes[1].set_title('One-Hot Encoding')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_model_parameters(results_dict, parameter_name, 
                           metric_name="Test Accuracy"):
    """
    Compare experimental results across different parameter values.
    
    Args:
        results_dict (dict): Dictionary mapping parameter values to results
        parameter_name (str): Name of the parameter being varied
        metric_name (str): Name of the metric being plotted
    """
    param_values = sorted(results_dict.keys())
    
    # If results contain multiple delays, plot separate lines
    if isinstance(list(results_dict.values())[0], dict):
        plt.figure(figsize=(10, 6))
        
        # Get all delays from first result
        delays = sorted(list(results_dict.values())[0].keys())
        
        for delay in delays:
            metric_values = [results_dict[param][delay] for param in param_values]
            plt.plot(param_values, metric_values, 'o-', 
                    label=f'Delay = {delay}', linewidth=2, markersize=6)
        
        plt.xlabel(parameter_name)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs {parameter_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    else:
        # Single metric per parameter value
        metric_values = [results_dict[param] for param in param_values]
        
        plt.figure(figsize=(8, 6))
        plt.plot(param_values, metric_values, 'o-', 
                linewidth=2, markersize=8)
        plt.xlabel(parameter_name)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs {parameter_name}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_experiment_results(results, filename, description=""):
    """
    Save experimental results to a file.
    
    Args:
        results (dict): Results dictionary to save
        filename (str): Output filename (without extension)
        description (str): Optional description of the experiment
    """
    import json
    import datetime
    
    # Add metadata
    save_data = {
        'results': results,
        'description': description,
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment_type': 'rnn_working_memory'
    }
    
    # Save as JSON
    with open(f"{filename}.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"Results saved to {filename}.json")


def load_experiment_results(filename):
    """
    Load experimental results from a file.
    
    Args:
        filename (str): Input filename (with or without .json extension)
        
    Returns:
        dict: Loaded results dictionary
    """
    import json
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


def calculate_memory_capacity_metric(accuracies, threshold=0.5):
    """
    Calculate a memory capacity metric based on decoder accuracies.
    
    Args:
        accuracies (dict): Dictionary mapping delays to accuracies
        threshold (float): Minimum accuracy threshold for "successful" memory
        
    Returns:
        int: Maximum delay with accuracy above threshold
    """
    delays = sorted(accuracies.keys())
    
    for delay in reversed(delays):
        if accuracies[delay] >= threshold:
            return delay
    
    return 0  # No delay meets threshold


def summarize_experiment_results(results_dict, experiment_name="Experiment"):
    """
    Print a summary of experimental results.
    
    Args:
        results_dict (dict): Dictionary of experimental results
        experiment_name (str): Name of the experiment
    """
    print(f"\n=== {experiment_name} Summary ===")
    
    for condition, results in results_dict.items():
        print(f"\nCondition: {condition}")
        
        if isinstance(results, dict):
            # Multiple delays
            best_delay = max(results.keys(), key=lambda d: results[d])
            worst_delay = min(results.keys(), key=lambda d: results[d])
            
            print(f"  Best performance: {results[best_delay]:.3f} (delay={best_delay})")
            print(f"  Worst performance: {results[worst_delay]:.3f} (delay={worst_delay})")
            
            # Memory capacity
            capacity = calculate_memory_capacity_metric(results)
            print(f"  Memory capacity: {capacity} timesteps")
            
        else:
            # Single value
            print(f"  Result: {results:.3f}")
    
    print("=" * (len(experiment_name) + 16))
