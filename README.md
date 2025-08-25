# Master-cho-mamba 🚀

**Mamba-based Offline Reinforcement Learning Agent for D4RL Environments**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

Master-cho-mamba is a state-of-the-art offline reinforcement learning agent that leverages **Mamba (State Space Model)** architecture for sequence modeling in reinforcement learning tasks. Built specifically for D4RL (Deep Data-Driven Reinforcement Learning) benchmark environments, it combines the efficiency of Mamba with advanced offline RL techniques.

## ✨ Key Features

- **🦎 Mamba Architecture**: State Space Model for efficient sequence modeling
- **🎯 Offline RL**: Behavior Cloning + Q-learning hybrid approach
- **🌍 D4RL Support**: Full compatibility with D4RL benchmark environments
- **📊 Auxiliary Tasks**: State prediction and reward estimation for better representation learning
- **⚡ Efficient Training**: Optimized for large-scale offline datasets

## 🏗️ Architecture

```
Input: (batch_size, context_length, state_dim)
    ↓
Mamba Encoder (RoPE + Mamba2 + Attention Pooling)
    ↓
Fixed-dim Embedding: (batch_size, d_model)
    ↓
Q-Network + Action Selector
    ↓
Action: (batch_size, action_dim)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SChoish/Master-cho-mamba.git
cd Master-cho-mamba

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train on Hopper-medium-v2
python trainer.py --env hopper --dataset_type medium --context_length 5

# Train on Walker2d-expert-v2
python trainer.py --env walker2d --dataset_type expert --context_length 10

# Custom hyperparameters
python trainer.py \
    --env halfcheetah \
    --dataset_type medium-expert \
    --context_length 15 \
    --d_model 512 \
    --n_layers 8 \
    --batch_size 512
```

### Evaluation

```bash
# Evaluate trained model
python utils.py --checkpoint outputs/best_checkpoint.pt --env hopper-medium-v2
```

## 🔧 Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_length` | 5 | Sequence length for trajectory encoding |
| `d_model` | 256 | Mamba model dimension |
| `n_layers` | 6 | Number of Mamba layers |
| `alpha` | 2.5 | BC vs Q-learning weight |
| `beta_s` | 1.0 | State prediction loss weight |
| `beta_r` | 1.0 | Reward prediction loss weight |

### Supported Environments

- **Hopper**: `hopper-{random,medium,medium-replay,medium-expert,expert}-v2`
- **Walker2d**: `walker2d-{random,medium,medium-replay,medium-expert,expert}-v2`
- **HalfCheetah**: `halfcheetah-{random,medium,medium-replay,medium-expert,expert}-v2`
- **Ant**: `ant-{random,medium,medium-replay,medium-expert,expert}-v2`

## 📊 Results

*Results will be updated as training progresses. Performance metrics and D4RL scores will be added once we achieve competitive results.*

## 🛠️ Project Structure

```
Master-cho-mamba/
├── agent.py              # Bamba agent implementation
├── model.py              # Mamba encoder and Q-networks
├── trainer.py            # Training loop and utilities
├── utils.py              # Evaluation and utility functions
├── d4rl_utils/          # D4RL dataset handling
│   ├── dataset_downloader.py
│   └── trajectory_dataset.py
├── outputs/              # Training outputs (gitignored)
├── datasets/             # D4RL datasets (gitignored)
└── paper/                # Research materials (gitignored)
```

## 🔬 Technical Details

### Mamba Encoder
- **RoPE**: Rotary Position Embedding for positional information
- **Mamba2**: State Space Model for sequence modeling
- **Attention Pooling**: Weighted aggregation of sequence representations

### Training Strategy
- **Hybrid Learning**: Behavior Cloning + Q-learning
- **Auxiliary Tasks**: State and reward prediction for representation learning
- **Adaptive Weighting**: Dynamic balance between BC and RL objectives

### Optimization
- **AdamW**: Weight decay for regularization
- **Gradient Clipping**: Stable training
- **Warmup Schedulers**: Learning rate scheduling

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{master-cho-mamba2024,
  title={Master-cho-mamba: Mamba-based Offline Reinforcement Learning},
  author={SChoish},
  year={2024},
  url={https://github.com/SChoish/Master-cho-mamba}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mamba**: State Space Models for sequence modeling
- **D4RL**: Deep Data-Driven Reinforcement Learning benchmark
- **PyTorch**: Deep learning framework
- **MuJoCo**: Physics simulation environment

---

**Made with ❤️ by SChoish**

*For questions and discussions, please open an issue on GitHub.*
