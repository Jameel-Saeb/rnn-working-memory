#!/usr/bin/env python3
"""
Comprehensive Test Suite for RNN Working Memory Project

This script tests all major components to ensure everything is working properly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🔍 Testing imports...")
    
    try:
        import rnn_model
        import decoder
        import utils
        print("✅ New modular imports working")
    except ImportError as e:
        print(f"❌ Modular import error: {e}")
        return False
    
    try:
        from Generate_td import model, step, make_input
        from Decoder import Decoder
        print("✅ Legacy imports working")
    except ImportError as e:
        print(f"❌ Legacy import error: {e}")
        return False
    
    return True


def test_rnn_model():
    """Test RNN model functionality."""
    print("\n🧠 Testing RNN model...")
    
    try:
        import rnn_model
        
        # Test model creation
        model = rnn_model.create_default_model()
        model = rnn_model.initialize_weights(model)
        print(f"✅ Model created: {model['N']} neurons, g={model['g']}")
        
        # Test input generation
        onehot, stream = rnn_model.make_input(10, model)
        print(f"✅ Input generated: shape {onehot.shape}")
        
        # Test simulation
        firing_rates = rnn_model.simulate_network(model, onehot)
        print(f"✅ Simulation completed: activity range [{firing_rates.min():.3f}, {firing_rates.max():.3f}]")
        
        # Test training data generation
        X, Y = rnn_model.generate_training_data(model, 100, delay=5)
        print(f"✅ Training data: {X.shape[0]} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ RNN model test failed: {e}")
        return False


def test_decoder():
    """Test decoder functionality."""
    print("\n🤖 Testing decoder...")
    
    try:
        import rnn_model
        import decoder
        
        # Create model and data
        model = rnn_model.create_default_model()
        model = rnn_model.initialize_weights(model)
        X, Y = rnn_model.generate_training_data(model, 200, delay=5)
        
        # Test decoder training
        dec, accuracy, losses = decoder.train_decoder(
            X, Y, model, delay=5, num_epochs=2, verbose=False
        )
        print(f"✅ Decoder trained: accuracy = {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decoder test failed: {e}")
        return False


def test_experiments():
    """Test experiment scripts."""
    print("\n🧪 Testing experiments...")
    
    try:
        # Test by running a minimal version
        import rnn_model
        import decoder
        
        model = rnn_model.create_default_model()
        model = rnn_model.initialize_weights(model)
        
        # Test multiple delays (minimal)
        delays = [3, 5]
        loss_curves, accuracies = decoder.evaluate_multiple_delays(
            model, delays, sequence_length=100, num_epochs=1
        )
        
        print(f"✅ Multi-delay test: {len(accuracies)} delays tested")
        for delay, acc in accuracies.items():
            print(f"   Delay {delay}: {acc:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment test failed: {e}")
        return False


def test_legacy_compatibility():
    """Test that legacy code still works."""
    print("\n📜 Testing legacy compatibility...")
    
    try:
        from Generate_td import model, step, make_input, generate_trainingdata
        from Decoder import Decoder
        
        # Test legacy RNN
        onehot, stream = make_input(5, model)
        firing_rates = np.zeros(model['N'])
        for t in range(10):
            firing_rates = step(firing_rates, onehot[:, min(t, onehot.shape[1]-1)], model)
        
        print(f"✅ Legacy RNN: final activity range [{firing_rates.min():.3f}, {firing_rates.max():.3f}]")
        
        # Test legacy training data generation
        X, Y = generate_trainingdata(model, 50, 3)
        print(f"✅ Legacy training data: {X.shape[0]} samples")
        
        # Test legacy decoder
        import torch
        decoder_net = Decoder(model['N'], 32, model['nIn'])
        test_input = torch.FloatTensor(X[:5])
        output = decoder_net(test_input)
        print(f"✅ Legacy decoder: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        return False


def test_visualization():
    """Test visualization utilities."""
    print("\n📊 Testing visualization...")
    
    try:
        import rnn_model
        import utils
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        model = rnn_model.create_default_model()
        model = rnn_model.initialize_weights(model)
        
        # Test eigenvalue analysis (just computation, no display)
        eigenvalues = np.linalg.eigvals(model['J'])
        max_eigenval = np.max(np.abs(eigenvalues))
        print(f"✅ Eigenvalue analysis: max magnitude = {max_eigenval:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests."""
    print("🚀 RNN Working Memory Project - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("RNN Model Test", test_rnn_model),
        ("Decoder Test", test_decoder),
        ("Experiment Test", test_experiments),
        ("Legacy Compatibility Test", test_legacy_compatibility),
        ("Visualization Test", test_visualization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"📊 OVERALL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Project is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
