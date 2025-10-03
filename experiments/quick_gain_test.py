#!/usr/bin/env python3
"""
Quick Test of Experiment 2: Effect of Recurrent Gain Parameter (g)

A shortened version of the gain experiment for quick testing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import rnn_model
import decoder
import utils


def run_quick_gain_test():
    """Run a quick test of the gain parameter experiment."""
    print("=== Quick Gain Parameter Test ===")
    
    # Reduced parameters for faster testing
    gain_values = [0.8, 1.0, 1.2]  # Just 3 values
    delays = [5, 10, 15]  # Fewer delays
    sequence_length = 1000  # Shorter sequences
    num_epochs = 5  # Fewer epochs
    
    print(f"Testing gain values: {gain_values}")
    print(f"Delays: {delays}")
    print(f"Sequence length: {sequence_length}")
    print(f"Epochs: {num_epochs}")
    
    results = {}
    
    for g in gain_values:
        print(f"\n🔧 Testing gain g = {g}")
        
        # Create model with specific gain
        model = rnn_model.create_default_model()
        model['g'] = g
        model = rnn_model.initialize_weights(model)
        
        # Evaluate decoder performance (reduced parameters)
        _, test_accuracies = decoder.evaluate_multiple_delays(
            model, delays, sequence_length, num_epochs
        )
        
        results[g] = test_accuracies
        
        # Print summary for this gain
        avg_accuracy = sum(test_accuracies.values()) / len(test_accuracies)
        print(f"  ✅ Average accuracy: {avg_accuracy:.4f}")
        
        # Quick analysis
        best_delay = max(test_accuracies.keys(), key=lambda d: test_accuracies[d])
        worst_delay = min(test_accuracies.keys(), key=lambda d: test_accuracies[d])
        print(f"  📈 Best at delay {best_delay}: {test_accuracies[best_delay]:.3f}")
        print(f"  📉 Worst at delay {worst_delay}: {test_accuracies[worst_delay]:.3f}")
    
    # Quick comparison
    print(f"\n=== Quick Results Summary ===")
    for gain, accuracies in results.items():
        avg_acc = sum(accuracies.values()) / len(accuracies)
        print(f"Gain {gain}: Average = {avg_acc:.3f}")
    
    return results


if __name__ == "__main__":
    results = run_quick_gain_test()
    print("\n✅ Quick gain test completed!")
