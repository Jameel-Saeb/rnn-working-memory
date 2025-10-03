"""
RNN Working Memory Model

This module implements a recurrent neural network model for studying working memory
capacity in neural systems. Based on the Neuromatch Academy RNN framework and
inspired by Barak and Tsodyks (2014).

Authors: Vivian Kang, Jameel Saeb, Chetanya Singh, Ah-Young Moon
"""

import math
import numpy as np
from matplotlib import pyplot as plt


def create_default_model():
    """
    Create a default RNN model with standard parameters for working memory tasks.
    
    Returns:
        dict: Model parameters dictionary containing network and input properties
    """
    model = {}
    
    # Recurrent pool properties
    model['N'] = 1000  # number of neurons in the recurrent pool
    model['g'] = 1.0  # gain parameter controlling recurrent connection strength
    model['sp'] = 0.25  # sparsity: fraction of non-zero connections in recurrent matrix
    model['tau'] = 20  # neural membrane time constant in milliseconds
    model['dt'] = 0.1  # simulation timestep in milliseconds
    model['nonlin'] = lambda x: np.tanh(x)  # activation function for recurrent units
    
    # Input layer properties
    model['nIn'] = 20  # number of input channels (for one-hot encoding)
    model['gIn'] = 10.0  # gain parameter for input connections
    model['spIn'] = 0.05  # sparsity of input-to-recurrent connections
    model['burnIn'] = 10  # burn-in time before inputs start (ms)
    model['durIn'] = 1  # duration each input pulse is active (ms)
    model['ISI'] = 0  # inter-stimulus interval between inputs (ms)
    model['nonlinIn'] = lambda x: x  # linear activation for input layer
    
    return model


def initialize_weights(model):
    """
    Initialize the recurrent and input weight matrices for the RNN.
    
    Args:
        model (dict): Model parameters dictionary
        
    Returns:
        dict: Updated model with weight matrices J (recurrent) and Jin (input)
    """
    # Create recurrent weight matrix J
    # Random Gaussian weights with sparse connectivity
    randMat = np.random.normal(0, 1, size=(model['N'], model['N']))
    spMat = np.random.uniform(0, 1, size=(model['N'], model['N'])) <= model['sp']
    
    # Normalize by sqrt(N*sparsity) to maintain eigenvalue spectrum scaling
    model['J'] = np.multiply(randMat, spMat) * model['g'] / math.sqrt(model['N'] * model['sp'])
    
    # Create input weight matrix Jin
    randMatIn = np.random.normal(0, 1, size=(model['N'], model['nIn']))
    spMatIn = np.random.uniform(0, 1, size=(model['N'], model['nIn'])) <= model['spIn']
    
    # Normalize input weights similarly
    model['Jin'] = np.multiply(randMatIn, spMatIn) * model['gIn'] / math.sqrt(model['nIn'] * model['spIn'])
    
    return model


def step(firing_rates, input_layer, model):
    """
    Perform one simulation step using Euler's method.
    
    Updates neural firing rates based on recurrent dynamics and external input.
    
    Args:
        firing_rates (np.ndarray): Current firing rates of recurrent neurons (N,)
        input_layer (np.ndarray): Current input values (nIn,)
        model (dict): Model parameters dictionary
        
    Returns:
        np.ndarray: Updated firing rates after one timestep
    """
    # Exponential decay factor for membrane dynamics
    timestep = math.exp(-model['dt'] / model['tau'])
    
    # Compute total input: recurrent + external
    vIn = (np.matmul(model['J'], firing_rates) + 
           np.matmul(model['Jin'], model['nonlinIn'](input_layer)))
    
    # Update firing rates using first-order dynamics
    updated_rates = model['nonlin'](vIn + (firing_rates - vIn) * timestep)
    
    return updated_rates


def make_input(sequence_length, model):
    """
    Generate a sequence of inputs for the RNN simulation.
    
    Creates a random sequence of one-hot encoded inputs with specified timing.
    
    Args:
        sequence_length (int): Number of input pulses to generate
        model (dict): Model parameters dictionary
        
    Returns:
        tuple: (onehot, input_stream)
            - onehot (np.ndarray): One-hot encoded input matrix (nIn, total_time)
            - input_stream (np.ndarray): Integer sequence of input identities
    """
    # Initialize with burn-in period (no input)
    input_stream = [0] * int(model['burnIn'] / model['dt'])
    
    # Generate sequence of random inputs
    for i in range(sequence_length):
        # Random input identity (1 to nIn)
        val = np.random.randint(0, model['nIn']) + 1
        
        # Add inter-stimulus interval
        for t in range(int(model['ISI'] / model['dt'])):
            input_stream.append(0.0)
            
        # Add input pulse duration
        for t in range(int(model['durIn'] / model['dt'])):
            input_stream.append(val)
    
    input_stream = np.array(input_stream)
    
    # Convert to one-hot encoding
    onehot = np.zeros((model['nIn'] + 1, input_stream.size))
    onehot[input_stream.astype(int), np.arange(input_stream.size)] = 1.0
    onehot = onehot[1:, :]  # Remove the "no input" row
    
    return onehot, input_stream


def generate_training_data(model, sequence_length, delay):
    """
    Generate training data pairs for decoder training.
    
    Creates (network_state, past_input) pairs where the network state at time t
    is paired with the input that was active at time (t - delay).
    
    Args:
        model (dict): Model parameters dictionary
        sequence_length (int): Number of input pulses in the sequence
        delay (int): Number of timesteps to look back for target labels
        
    Returns:
        tuple: (X, Y)
            - X (np.ndarray): Network states at time t (n_samples, N)
            - Y (np.ndarray): Input identities at time (t - delay) (n_samples,)
    """
    # Generate input sequence and simulate network
    input_onehot, input_stream = make_input(sequence_length, model)
    
    states = []  # Store network states
    inputs = []  # Store corresponding input identities
    
    # Initialize network with zero firing rates
    firing_rates = np.zeros(model['N'])
    
    # Simulate network dynamics
    for t in range(input_onehot.shape[1]):
        input_t = input_onehot[:, t]
        firing_rates = step(firing_rates, input_t, model)
        
        # Record state when an input is present
        if np.sum(input_t) > 0:
            input_id = np.argmax(input_t)  # Get active input identity
            states.append(firing_rates.copy())
            inputs.append(input_id)
    
    # Convert to arrays and align for delay
    X = np.array(states)
    Y = np.array(inputs)
    
    # Create training pairs: state at t, input at (t - delay)
    X = X[delay:]  # Network states from time delay onwards
    Y = Y[:-delay]  # Input labels from time 0 to (end - delay)
    
    return X, Y


def simulate_network(model, input_sequence):
    """
    Simulate the RNN with a given input sequence.
    
    Args:
        model (dict): Model parameters dictionary
        input_sequence (np.ndarray): Input sequence (nIn, time_steps)
        
    Returns:
        np.ndarray: Firing rates over time (N, time_steps)
    """
    time_steps = input_sequence.shape[1]
    firing_rates = np.zeros((model['N'], time_steps))
    
    # Initialize with small random activity
    firing_rates[:, 0] = np.random.uniform(0, 0.1, size=(model['N']))
    
    # Simulate dynamics
    for t in range(time_steps - 1):
        firing_rates[:, t + 1] = step(firing_rates[:, t], input_sequence[:, t], model)
    
    return firing_rates
