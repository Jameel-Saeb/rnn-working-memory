# Legacy Training Data Generation
#
# This is the original implementation used for generating RNN training data.
# For the improved, well-documented version, see src/rnn_model.py
#
# This file is kept for reproducibility of original results.

import math
from matplotlib import pyplot as plt
import numpy as np

# RNN Model Configuration
# Set up the recurrent neural network parameters in a dictionary for easy access
model = {}

# Recurrent neural network parameters
model['N'] = 1000  # number of neurons in the recurrent pool
model['g'] = 1  # gain parameter: controls strength of recurrent connections
model['sp'] = 0.25  # sparsity: fraction of non-zero connections in recurrent matrix
model['tau'] = 20  # neural membrane time constant in milliseconds  
model['dt'] = 0.1  # simulation timestep in milliseconds
model['nonlin'] = lambda x: np.tanh(x)  # activation function for recurrent units

# Input layer parameters
# Note: Using one-hot encoding where each input activates exactly one neuron
# Alternative encoding strategies could potentially improve memory capacity
model['nIn'] = 20  # number of input channels (size of input vocabulary)
model['gIn'] = 10.0  # gain parameter for input-to-network connections
model['spIn'] = 0.05  # sparsity of input-to-recurrent connections (5% connected)
model['burnIn'] = 10  # burn-in time before inputs start (milliseconds)
model['durIn'] = 1  # duration each input pulse is active (milliseconds)
model['ISI'] = 0  # inter-stimulus interval between inputs (milliseconds)
model['nonlinIn'] = lambda x: x  # linear activation for input layer (no distortion)

# Create the synaptic weight matrix.
# Normalizing weights by sqrt(N*sparsity) keeps the eigenvalue spectrum
# invariant to the size of the population N.
randMat  = np.random.normal(0, 1, size=(model['N'], model['N']))
spMat  = np.random.uniform(0, 1, size=(model['N'], model['N'])) <= model['sp']
model['J'] = np.multiply(randMat, spMat) * model['g'] / math.sqrt(model['N'] * model['sp'])

# Create the input weight matrix.
randMatIn = np.random.normal(0, 1, size=(model['N'], model['nIn']))
spMatIn = np.random.uniform(0, 1, size=(model['N'], model['nIn'])) <= model['spIn']
model['Jin'] = np.multiply(randMatIn, spMatIn) * model['gIn'] / math.sqrt(model['nIn'] * model['spIn'])

def step(firing_rates, input_layer, model):
  # The simulation function. We use Euler's method to simulate the evolution of
  # model neuron firing rates given the input_layer firing rates.

  timestep = math.exp(-model['dt']/model['tau'])
  vIn = np.matmul(model['J'], firing_rates) \
        + np.matmul(model['Jin'], model['nonlinIn'](input_layer))
  updated_rates = model['nonlin'](vIn + (firing_rates - vIn) * timestep)

  return updated_rates


def make_input(sequence_length, model):
  # Generates a sequence of inputs according to the parameters in model. Returns
  # the sequence both as a one-hot encoding and as a sequence of integer values.

  input_stream = [0] * int(model['burnIn']/model['dt'])

  for i in range(sequence_length):
    val = np.random.randint(0, model['nIn']) + 1
    for t in range(int(model['ISI']/model['dt'])):
      input_stream.append(0.0)
    for t in range(int(model['durIn']/model['dt'])):
      input_stream.append(val)

  input_stream = np.array(input_stream)

  onehot = np.zeros((model['nIn'] + 1, input_stream.size))
  onehot[input_stream, np.arange(input_stream.size)] = 1.0
  onehot = onehot[1:, :]

  return onehot, input_stream


def generate_trainingdata(model, sequence_length, delay):
    input_onehot, input_stream = make_input(sequence_length, model)
    States = []  
    Inputs = []  
    delay = delay 

    firing_rates = np.zeros(model['N'])
    for t in range(input_onehot.shape[1]):
        input_t = input_onehot[:, t]
        firing_rates = step(firing_rates, input_t, model)
        
        if np.sum(input_t) > 0:
            input_id = np.argmax(input_t)  
            States.append(firing_rates.copy())  
            Inputs.append(input_id)
            
    X = np.array(States)
    Y = np.array(Inputs)
    X = X[delay:]  # Match delays
    Y = Y[:-delay]  
    return X, Y




