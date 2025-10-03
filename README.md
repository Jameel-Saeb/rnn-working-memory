# RNN Working Memory Capacity Study

A comprehensive computational neuroscience project investigating the working memory capacity of recurrent neural networks (RNNs). This work extends the Neuromatch Academy RNN framework and builds upon theoretical foundations from Barak and Tsodyks (2014).

## Authors

- Vivian Kang
- Jameel Saeb
- Chetanya Singh
- Ah-Young Moon

## Project Overview

This project explores how recurrent neural networks can model working memory - the brain's ability to temporarily hold and manipulate information. We investigate the factors that influence memory capacity and retention in artificial neural systems.

### Research Questions

1. **Memory Decay**: How does decoder accuracy change with increasing time delays?
2. **Network Dynamics**: How does the recurrent gain parameter (g) affect memory performance?
3. **Input Connectivity**: How does the sparsity of input connections influence memory capacity?
4. **Interference Effects**: How do continuous input streams affect memory retention?
5. **Encoding Strategies**: Which input representation schemes optimize memory performance?

## Project Structure

```
NEUR680FINAL/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── rnn_model.py             # RNN implementation and simulation
│   ├── decoder.py               # Neural decoder for temporal prediction
│   └── utils.py                 # Visualization and analysis utilities
├── experiments/                  # Experimental scripts
│   ├── experiment1_delay_study.py      # Time delay experiments
│   └── experiment2_gain_study.py       # Gain parameter experiments
├── notebooks/                   # Jupyter notebooks
│   └── RNN_working_memory.ipynb # Original exploration notebook
├── results/                     # Results and figures
│   ├── *.png                   # Generated plots and figures
│   └── *.json                  # Saved experimental results
├── Final_project.qmd           # Quarto presentation source
├── Final_project.html          # Compiled presentation
└── README.md                   # This file
```

## Installation and Setup

### Prerequisites

```bash
# Required Python packages
numpy
matplotlib
torch
jupyter
```

### Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install numpy matplotlib torch jupyter
```

3. Add the src directory to your Python path or run experiments from the project root.

## Core Components

### RNN Model (`src/rnn_model.py`)

Implements a sparse recurrent neural network with the following features:

- **1000 neurons** in the recurrent pool
- **Sparse connectivity** (25% non-zero connections)
- **One-hot input encoding** (20 possible input states)
- **Configurable dynamics** through gain parameter

Key functions:

- `create_default_model()`: Initialize model with standard parameters
- `generate_training_data()`: Create decoder training datasets
- `simulate_network()`: Run network simulations

### Decoder Network (`src/decoder.py`)

PyTorch-based feedforward neural network that learns to predict past inputs from current network states:

- **Input**: Current RNN state (1000 neurons)
- **Output**: Predicted input identity from X timesteps ago
- **Architecture**: Feedforward with ReLU activation

Key functions:

- `WorkingMemoryDecoder`: Main decoder class
- `train_decoder()`: Training pipeline with validation
- `evaluate_multiple_delays()`: Batch evaluation across delays

### Utilities (`src/utils.py`)

Analysis and visualization tools:

- **Eigenvalue analysis**: Examine network stability
- **Activity visualization**: Plot neural dynamics
- **Result comparison**: Compare across parameters
- **Data management**: Save/load experimental results

## Running Experiments

### Experiment 1: Time Delay Study

Evaluates memory decay over increasing time delays:

```bash
cd experiments
python3 experiment1_delay_study.py
```

**Expected Results**:

- Short delays (5 steps): ~93% accuracy
- Medium delays (10-15 steps): ~77% accuracy
- Long delays (20 steps): ~66% accuracy

### Experiment 2: Gain Parameter Study

Tests the effect of recurrent connection strength:

```bash
cd experiments
python3 experiment2_gain_study.py
```

**Expected Results**:

- g = 0.5: Weak dynamics, poor memory
- g = 1.0: Optimal balance
- g > 1.5: Chaotic, unstable memory

## Key Findings

### 1. Memory Degrades Over Time

Decoder accuracy decreases systematically with longer time delays, demonstrating fundamental limits of working memory capacity.

### 2. Critical Dynamics Optimize Memory

Networks perform best at the "edge of chaos" (g ≈ 1.0), balancing stability with dynamic richness.

### 3. Sparse Input Connectivity Is Optimal

Input sparsity around 5-10% provides the best trade-off between signal strength and interference.

### 4. Continuous Inputs Cause Interference

New inputs overwrite previous memory traces, limiting capacity under ongoing stimulation.

### 5. Subset-Based Encoding Outperforms Magnitude-Based

Using different neurons for different inputs (one-hot) preserves memory better than varying spike rates.

## Biological Relevance

Our findings parallel key properties of biological working memory:

- **Capacity Limitations**: Like human working memory, RNN capacity is limited and degrades over time
- **Prefrontal Dynamics**: The optimal gain regime mirrors neural dynamics in prefrontal cortex
- **Interference Effects**: New information interferes with stored memories, as observed in psychology
- **Network Criticality**: Biological neural networks also operate near critical points

## Applications

This research has implications for:

### Neuroscience

- **Computational models** of working memory disorders
- **Understanding** neural dynamics in memory tasks
- **Predictions** for experimental interventions

### Machine Learning

- **Improved RNN architectures** for sequential tasks
- **Memory-augmented networks** for language and planning
- **Robust temporal processing** in AI systems

## Future Directions

1. **Extended Parameter Exploration**: Investigate additional network properties like heterogeneous time constants
2. **Biologically Realistic Mechanisms**: Incorporate synaptic plasticity and inhibitory circuits
3. **Real-World Applications**: Test on language processing and sensory prediction tasks
4. **Scaling Analysis**: Examine performance in larger networks and longer sequences
5. **Multi-Item Memory**: Extend to simultaneous storage of multiple items

## Usage Examples

### Basic RNN Simulation

```python
from src.rnn_model import create_default_model, initialize_weights, simulate_network, make_input

# Create and initialize model
model = create_default_model()
model = initialize_weights(model)

# Generate input sequence
input_onehot, input_stream = make_input(10, model)

# Simulate network
firing_rates = simulate_network(model, input_onehot)

# Visualize results
from src.utils import visualize_network_activity
visualize_network_activity(firing_rates, input_stream, model)
```

### Train a Decoder

```python
from src.rnn_model import generate_training_data
from src.decoder import train_decoder

# Generate training data
X_train, Y_train = generate_training_data(model, sequence_length=1000, delay=10)

# Train decoder
decoder, accuracy, losses = train_decoder(X_train, Y_train, model, delay=10)
print(f"Test accuracy: {accuracy:.3f}")
```

### Run Parameter Comparison

```python
from src.decoder import evaluate_multiple_delays
from src.utils import compare_model_parameters

# Test multiple gain values
gains = [0.5, 1.0, 1.5, 2.0]
results = {}

for g in gains:
    model['g'] = g
    model = initialize_weights(model)
    _, accuracies = evaluate_multiple_delays(model, delays=[5, 10, 15, 20])
    results[g] = accuracies

# Compare results
compare_model_parameters(results, "Gain Parameter", "Test Accuracy")
```

## Contributing

This project was developed as part of a computational neuroscience course. For questions or suggestions:

1. Check existing documentation and code comments
2. Review the experiment scripts for usage examples
3. Refer to the original Neuromatch Academy materials for additional context

## References

1. Barak, O., & Tsodyks, M. (2014). Working models of working memory. _Current Opinion in Neurobiology_, 25, 20-24.

2. Neuromatch Academy. (2023). Working memory capacity of recurrent neural network models. _Computational Neuroscience Course Materials_.

3. Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. _Neuron_, 63(4), 544-557.

## License

This project is developed for educational purposes as part of coursework in computational neuroscience.

---

_For detailed methodology and results, see the compiled presentation in `Final_project.html` or the source Quarto document `Final_project.qmd`._
