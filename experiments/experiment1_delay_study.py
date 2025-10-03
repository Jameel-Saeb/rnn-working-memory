"""
Experiment 1: Decoder Performance Across Different Time Delays

This script evaluates how decoder accuracy changes as a function of the temporal
delay between the network state and the target input to be decoded.

Research Question: How does working memory capacity degrade over time?
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import rnn_model
import decoder 
import utils


def run_delay_experiment(delays=[3, 5, 10, 15, 20], sequence_length=3000, 
                        num_epochs=10, save_results=True):
    """
    Run the time delay experiment.
    
    Args:
        delays (list): List of delay values to test
        sequence_length (int): Length of training sequences
        num_epochs (int): Training epochs per decoder
        save_results (bool): Whether to save results to file
        
    Returns:
        tuple: (loss_curves, test_accuracies)
    """
    print("=== Experiment 1: Decoder Performance vs Time Delay ===")
    print(f"Testing delays: {delays}")
    print(f"Sequence length: {sequence_length}")
    print(f"Training epochs: {num_epochs}")
    
    # Create and initialize model
    model = rnn_model.create_default_model()
    model = rnn_model.initialize_weights(model)
    
    # Run experiment
    loss_curves, test_accuracies = decoder.evaluate_multiple_delays(
        model, delays, sequence_length, num_epochs
    )
    
    # Print results summary
    print("\n=== Results Summary ===")
    for delay in sorted(test_accuracies.keys()):
        print(f"Delay = {delay:2d}: Test Accuracy = {test_accuracies[delay]:.4f}")
    
    # Visualize results
    decoder.plot_training_curves(loss_curves, "Training Loss Curves - Time Delay Experiment")
    decoder.plot_accuracy_vs_delay(test_accuracies, "Memory Decay Over Time")
    
    # Save results
    if save_results:
        results = {
            'test_accuracies': test_accuracies,
            'loss_curves': loss_curves,
            'model_parameters': model,
            'experiment_parameters': {
                'delays': delays,
                'sequence_length': sequence_length,
                'num_epochs': num_epochs
            }
        }
        utils.save_experiment_results(
            results, 
            "../results/experiment1_delay_study",
            "Time delay experiment measuring memory decay"
        )
    
    return loss_curves, test_accuracies


if __name__ == "__main__":
    # Run the experiment with default parameters
    loss_curves, accuracies = run_delay_experiment()
    
    # Print final summary
    utils.summarize_experiment_results(
        {'delay_experiment': accuracies}, 
        "Time Delay Study"
    )
