"""
RNN Working Memory Package

A comprehensive toolkit for studying working memory capacity in recurrent neural networks.
Based on the Neuromatch Academy framework and Barak & Tsodyks (2014) theoretical work.

This package provides:
- RNN models for working memory simulation
- Decoder networks for temporal decoding
- Experimental utilities and visualization tools
- Predefined experiments for parameter exploration

Authors: Vivian Kang, Jameel Saeb, Chetanya Singh, Ah-Young Moon
"""

from .rnn_model import (
    create_default_model,
    initialize_weights, 
    step,
    make_input,
    generate_training_data,
    simulate_network
)

from .decoder import (
    WorkingMemoryDecoder,
    train_decoder,
    evaluate_multiple_delays,
    plot_training_curves,
    plot_accuracy_vs_delay
)

from .utils import (
    analyze_eigenvalue_spectrum,
    visualize_network_activity,
    plot_input_encoding_example,
    compare_model_parameters,
    save_experiment_results,
    load_experiment_results,
    calculate_memory_capacity_metric,
    summarize_experiment_results
)

__version__ = "1.0.0"
__all__ = [
    # RNN model functions
    'create_default_model',
    'initialize_weights',
    'step', 
    'make_input',
    'generate_training_data',
    'simulate_network',
    
    # Decoder functions
    'WorkingMemoryDecoder',
    'train_decoder',
    'evaluate_multiple_delays', 
    'plot_training_curves',
    'plot_accuracy_vs_delay',
    
    # Utility functions
    'analyze_eigenvalue_spectrum',
    'visualize_network_activity',
    'plot_input_encoding_example',
    'compare_model_parameters',
    'save_experiment_results',
    'load_experiment_results',
    'calculate_memory_capacity_metric',
    'summarize_experiment_results'
]
