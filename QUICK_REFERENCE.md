# RNN Working Memory Project - Quick Reference

## 🚀 Quick Start

1. **Setup**: `python setup.py`
2. **Demo**: `python demo.py`
3. **Experiments**: `cd experiments && python experiment1_delay_study.py`

## 📁 Project Structure

```
NEUR680FINAL/
├── 📄 README.md                    # Main documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Setup script
├── 📄 demo.py                      # Quick demo
├── 📄 config.ini                   # Configuration
├── 📄 CHANGELOG.md                 # Version history
│
├── 📂 src/                         # Core source code
│   ├── __init__.py                 # Package initialization
│   ├── rnn_model.py               # RNN implementation
│   ├── decoder.py                 # Neural decoder
│   └── utils.py                   # Utilities & visualization
│
├── 📂 experiments/                 # Experiment scripts
│   ├── experiment1_delay_study.py # Time delay analysis
│   └── experiment2_gain_study.py  # Gain parameter study
│
├── 📂 notebooks/                   # Jupyter notebooks
│   └── RNN_working_memory.ipynb   # Original exploration
│
├── 📂 results/                     # Results & figures
│   └── *.png                      # Generated plots
│
├── 📂 Final_project_files/         # Presentation assets
│
└── 📄 Legacy files                 # Original implementations
    ├── Decoder.py                  # (preserved for reproducibility)
    ├── Generate_td.py              # (preserved for reproducibility)
    └── test_input_neuron_counts.py # (preserved for reproducibility)
```

## 🔬 Core Components

### RNN Model (`src/rnn_model.py`)

- `create_default_model()` - Initialize model parameters
- `initialize_weights()` - Create weight matrices
- `simulate_network()` - Run network simulation
- `generate_training_data()` - Create decoder training data

### Decoder (`src/decoder.py`)

- `WorkingMemoryDecoder` - PyTorch decoder class
- `train_decoder()` - Training pipeline
- `evaluate_multiple_delays()` - Batch evaluation

### Utilities (`src/utils.py`)

- `visualize_network_activity()` - Plot neural dynamics
- `analyze_eigenvalue_spectrum()` - Stability analysis
- `compare_model_parameters()` - Parameter comparison

## 🧪 Running Experiments

```bash
# Time delay study
cd experiments
python experiment1_delay_study.py

# Gain parameter study
python experiment2_gain_study.py
```

## 📊 Key Results

- **Memory Decay**: Accuracy drops from 93% (5 steps) to 66% (20 steps)
- **Optimal Gain**: g ≈ 1.0 balances stability and dynamics
- **Input Sparsity**: 5-10% connectivity is optimal
- **Encoding**: Subset-based > magnitude-based representation

## 🛠 Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup validation
python setup.py

# Quick demonstration
python demo.py
```

## 📚 References

1. Barak & Tsodyks (2014) - Working models of working memory
2. Neuromatch Academy - RNN working memory framework
3. Sussillo & Abbott (2009) - Coherent patterns from chaos
