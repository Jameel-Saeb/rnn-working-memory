# RNN Working Memory Capacity Study

> A computational neuroscience investigation of working memory dynamics in recurrent neural networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🧠 Overview

This project explores how recurrent neural networks model **working memory** - the brain's ability to temporarily hold and manipulate information. Through computational modeling, we investigate memory capacity limits, temporal dynamics, and the neural mechanisms underlying cognitive performance.

**Key Features:**

- 🔬 Biologically-inspired 1000-neuron RNN with sparse connectivity
- 🧮 PyTorch-based temporal decoder for memory analysis
- 📊 Comprehensive experimental framework with automated testing
- 📈 Publication-quality visualizations and results
- 🔄 Full reproducibility with organized codebase

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/rnn-working-memory.git
cd rnn-working-memory

# Install dependencies
pip install -r requirements.txt

# Run setup and validation
python setup.py

# Quick demonstration
python demo.py

# Run experiments
cd experiments
python experiment1_delay_study.py
```

## 📊 Key Results

| Metric                   | Finding                                          |
| ------------------------ | ------------------------------------------------ |
| **Memory Decay**         | 95% accuracy (3 steps) → 69% accuracy (20 steps) |
| **Optimal Dynamics**     | Best performance at recurrent gain g ≈ 1.0       |
| **Critical Point**       | Networks operate at "edge of chaos" for memory   |
| **Biological Relevance** | Matches prefrontal cortex dynamics               |

## 🏗️ Project Structure

```
├── src/                    # Core implementation
│   ├── rnn_model.py       # 1000-neuron sparse RNN
│   ├── decoder.py         # PyTorch temporal decoder
│   └── utils.py           # Analysis & visualization
├── experiments/           # Research experiments
├── notebooks/            # Interactive exploration
├── results/              # Generated figures & data
└── tests/               # Comprehensive test suite
```

## 🔬 Scientific Impact

This work contributes to understanding:

- **Computational principles** of biological working memory
- **Network dynamics** at critical points in neural systems
- **Memory interference** and capacity limitations
- **Applications** to AI systems requiring temporal processing

## 👥 Authors

**Team Members:**

- [Vivian Kang](https://github.com/viviankang)
- [Jameel Saeb](https://github.com/jameelsaeb)
- [Chetanya Singh](https://github.com/chetanyasingh)
- [Ah-Young Moon](https://github.com/ahyoungmoon)

_Developed as part of NEUR680 Computational Neuroscience_

## 📚 Documentation

- 📖 [Complete Documentation](README.md) - Full project details
- ⚡ [Quick Reference](QUICK_REFERENCE.md) - Essential commands
- 📋 [Change Log](CHANGELOG.md) - Version history
- 🧪 [Experiments Guide](experiments/) - Research protocols

## 🎯 For Employers

This project demonstrates:

✅ **Research Skills** - Novel computational neuroscience investigation  
✅ **Technical Proficiency** - Python, PyTorch, NumPy, scientific computing  
✅ **Software Engineering** - Clean architecture, testing, documentation  
✅ **Data Science** - Experimental design, statistical analysis, visualization  
✅ **Collaboration** - Team-based research project with clear contributions  
✅ **Communication** - Publication-quality results and presentations

**Technologies:** Python • PyTorch • NumPy • Matplotlib • Jupyter • Git

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Citation

If you use this work in your research, please cite:

```bibtex
@misc{kang2024rnn,
  title={RNN Working Memory Capacity Study},
  author={Kang, Vivian and Saeb, Jameel and Singh, Chetanya and Moon, Ah-Young},
  year={2024},
  institution={NEUR680 Computational Neuroscience},
  url={https://github.com/yourusername/rnn-working-memory}
}
```

---

_🧠 Exploring the computational foundations of cognition through neural network modeling_
