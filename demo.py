"""
Quick Start Demo for RNN Working Memory Project

This script demonstrates the basic usage of the RNN working memory toolkit
with a simple example that can be run immediately after setup.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt

try:
    import rnn_model
    import utils
    print("✅ Successfully imported RNN working memory modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've run 'python setup.py' first")
    sys.exit(1)


def quick_demo():
    """Run a quick demonstration of the RNN working memory system."""
    print("🧠 RNN Working Memory Quick Demo")
    print("=" * 35)
    
    # Step 1: Create and initialize model
    print("1. Creating RNN model...")
    model = rnn_model.create_default_model()
    model = rnn_model.initialize_weights(model)
    print(f"   Network size: {model['N']} neurons")
    print(f"   Input channels: {model['nIn']}")
    print(f"   Gain parameter: {model['g']}")
    
    # Step 2: Generate input sequence
    print("\n2. Generating input sequence...")
    input_onehot, input_stream = rnn_model.make_input(sequence_length=20, model=model)
    print(f"   Sequence length: {len(input_stream)} timesteps")
    print(f"   Number of inputs: {np.sum(input_stream > 0)}")
    
    # Step 3: Simulate network
    print("\n3. Simulating network dynamics...")
    firing_rates = rnn_model.simulate_network(model, input_onehot)
    print(f"   Final network activity range: [{firing_rates[:, -1].min():.3f}, {firing_rates[:, -1].max():.3f}]")
    
    # Step 4: Visualize results
    print("\n4. Generating visualizations...")
    
    # Plot network activity
    utils.visualize_network_activity(firing_rates, input_stream, model, 
                             title="Quick Demo: RNN Activity")
    
    # Analyze weight matrix
    utils.analyze_eigenvalue_spectrum(model['J'], 
                              title="Quick Demo: Recurrent Weight Matrix")
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("- Explore experiments/experiment1_delay_study.py for memory analysis")
    print("- Check notebooks/ for interactive tutorials")
    print("- See README.md for comprehensive documentation")


def memory_capacity_preview():
    """Show a preview of memory capacity analysis."""
    print("\n🔍 Memory Capacity Preview")
    print("=" * 26)
    
    # Test with different gain values
    gains = [0.8, 1.0, 1.2]
    
    for g in gains:
        print(f"\nTesting gain g = {g}...")
        
        # Create model with specific gain
        model = rnn_model.create_default_model()
        model['g'] = g
        model = rnn_model.initialize_weights(model)
        
        # Simulate short sequence
        input_onehot, _ = rnn_model.make_input(10, model)
        firing_rates = rnn_model.simulate_network(model, input_onehot)
        
        # Measure activity statistics
        final_activity = firing_rates[:, -1]
        mean_activity = np.mean(final_activity)
        std_activity = np.std(final_activity)
        max_eigenval = np.max(np.abs(np.linalg.eigvals(model['J'])))
        
        print(f"   Mean activity: {mean_activity:.3f}")
        print(f"   Activity std:  {std_activity:.3f}")
        print(f"   Max eigenval:  {max_eigenval:.3f}")
        
        # Simple stability assessment
        if max_eigenval > 1.1:
            print("   Status: ⚠️  Potentially unstable (chaotic)")
        elif max_eigenval < 0.9:
            print("   Status: 📉 Stable but weak")
        else:
            print("   Status: ✅ Balanced dynamics")


if __name__ == "__main__":
    try:
        # Run the demo
        quick_demo()
        
        # Show memory capacity preview
        memory_capacity_preview()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)
