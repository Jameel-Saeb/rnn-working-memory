"""
Experiment 2: Effect of Recurrent Gain Parameter (g)

This script investigates how the gain parameter 'g' affects working memory
performance. The gain controls the strength of recurrent connections and
determines network dynamics.

Research Question: What is the optimal balance between stability and chaos
for memory retention?
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import rnn_model
import decoder
import utils


def run_gain_experiment(gain_values=[0.5, 1.0, 1.5, 2.0], 
                       delays=[5, 10, 15, 20],
                       sequence_length=2000, num_epochs=8):
    """
    Test decoder performance across different gain values.
    
    Args:
        gain_values (list): List of gain values to test
        delays (list): Time delays to evaluate
        sequence_length (int): Length of training sequences
        num_epochs (int): Training epochs per condition
        
    Returns:
        dict: Results mapping gain values to accuracy dictionaries
    """
    print("=== Experiment 2: Recurrent Gain Parameter Study ===")
    print(f"Testing gain values: {gain_values}")
    print(f"Delays evaluated: {delays}")
    
    results = {}
    
    for g in gain_values:
        print(f"\nTesting gain g = {g}")
        
        # Create model with specific gain
        model = rnn_model.create_default_model()
        model['g'] = g
        model = rnn_model.initialize_weights(model)
        
        # Evaluate decoder performance
        _, test_accuracies = decoder.evaluate_multiple_delays(
            model, delays, sequence_length, num_epochs
        )
        
        results[g] = test_accuracies
        
        # Print summary for this gain
        avg_accuracy = sum(test_accuracies.values()) / len(test_accuracies)
        print(f"  Average accuracy: {avg_accuracy:.4f}")
    
    # Visualize results
    utils.compare_model_parameters(results, "Gain Parameter (g)", "Test Accuracy")
    
    # Save results
    experiment_data = {
        'results': results,
        'experiment_parameters': {
            'gain_values': gain_values,
            'delays': delays,
            'sequence_length': sequence_length,
            'num_epochs': num_epochs
        }
    }
    
    utils.save_experiment_results(
        experiment_data,
        "../results/experiment2_gain_study", 
        "Gain parameter experiment testing stability vs chaos trade-off"
    )
    
    return results


def analyze_gain_effects(results):
    """
    Analyze the effects of different gain values on memory performance.
    
    Args:
        results (dict): Results from gain experiment
    """
    print("\n=== Gain Parameter Analysis ===")
    
    for gain, accuracies in results.items():
        print(f"\nGain g = {gain}:")
        
        # Calculate metrics
        max_acc = max(accuracies.values())
        min_acc = min(accuracies.values())
        avg_acc = sum(accuracies.values()) / len(accuracies)
        
        # Memory capacity (delays with >50% accuracy)
        good_delays = [d for d, acc in accuracies.items() if acc > 0.5]
        capacity = max(good_delays) if good_delays else 0
        
        print(f"  Max accuracy: {max_acc:.3f}")
        print(f"  Min accuracy: {min_acc:.3f}")  
        print(f"  Avg accuracy: {avg_acc:.3f}")
        print(f"  Memory capacity: {capacity} timesteps")
        
        # Stability assessment
        if gain < 1.0:
            print("  Regime: Subcritical (stable but weak)")
        elif gain == 1.0:
            print("  Regime: Critical (balanced)")
        else:
            print("  Regime: Supercritical (chaotic)")


if __name__ == "__main__":
    # Run the experiment
    results = run_gain_experiment()
    
    # Analyze results
    analyze_gain_effects(results)
    
    # Overall summary
    utils.summarize_experiment_results(results, "Gain Parameter Study")
