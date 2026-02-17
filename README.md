# GNN-RNN Decoder for the Rotated Surface Code

A machine learning decoder for quantum error correction (QEC) on the rotated surface code, combining a Graph Neural Network (GNN) with a Gated Recurrent Unit (GRU). The GNN embeds per-round syndrome graphs into fixed-size vectors, which the GRU processes sequentially to predict whether a logical error has occurred.

Based on the master's thesis *"Sequential Graph-Based Decoding of the Surface Code"* (included as `Thesis.pdf`).

## How It Works

1. **Circuit simulation** — [Stim](https://github.com/quantumlib/Stim) generates noisy surface code circuits and samples syndrome data.
2. **Graph construction** — Each QEC round's syndrome (detector outcomes) is converted into a graph where nodes represent triggered detectors and edges connect nearby nodes (k-nearest neighbors).
3. **GNN embedding** — A stack of `GraphConv` layers embeds each round's syndrome graph into a fixed-size vector.
4. **Temporal processing** — The GRU reads the sequence of graph embeddings and outputs a hidden state that summarises the full error history.
5. **Decoding** — A linear head maps the final hidden state to a probability of logical failure.

The decoder is trained end-to-end with binary cross-entropy loss on Stim-simulated data.

## Repository Structure

```
.
├── args.py              # Args dataclass — all hyperparameters
├── data.py              # Stim circuit generation, syndrome sampling, graph construction
├── gru_decoder.py       # GNN + GRU model with train/test methods
├── mwpm.py              # MWPM baseline decoder (via PyMatching)
├── utils.py             # Graph convolution layer, grouping, logging utilities
├── requirements.txt     # Python dependencies
├── scripts/
│   ├── train_nn.py      # Training script
│   ├── test_nn.py       # Evaluation script (GNN-RNN vs MWPM)
│   └── load_nn.py       # Load pretrained models for d=3, 5, 7
├── models/
│   ├── distance3.pt     # Pretrained model, d=3
│   ├── distance5.pt     # Pretrained model, d=5
│   └── distance7.pt     # Pretrained model, d=7
├── figures/             # Performance plots
└── Thesis.pdf           # Master's thesis
```

## Setup

```bash
# Clone the repository
git clone https://github.com/LangeMoritz/GNN-RNN-surface-code-IBM_bachelor_project_2026.git
cd GNN-RNN-surface-code-IBM_bachelor_project_2026

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes a `--find-links` line for PyG with CUDA 12.1. If you are running on CPU only, comment out the CUDA line and uncomment the CPU line in `requirements.txt`.

## Usage

### Training

Edit the `Args` in `scripts/train_nn.py` to set the desired distance, error rates, number of rounds, and model size, then run:

```bash
python scripts/train_nn.py
```

Key parameters in `args.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `distance` | Code distance (d) | 5 |
| `error_rates` | Physical error rates to train on | [0.001 .. 0.005] |
| `t` | Number of QEC rounds | [99] |
| `dt` | Chunk size for sliding window | 2 |
| `sliding` | Use sliding window over rounds | True |
| `embedding_features` | GNN layer sizes | [5, 32, 64, 128, 256] |
| `hidden_size` | GRU hidden dimension | 128 |
| `n_layers` | Number of GRU layers | 4 |
| `batch_size` | Batch size | 2048 |
| `n_epochs` | Training epochs | 600 |

### Testing

Evaluate a pretrained model against MWPM:

```bash
python scripts/test_nn.py
```

This loads a pretrained model, runs inference on freshly sampled syndromes, and prints accuracy alongside the MWPM baseline.

### Loading Pretrained Models

`scripts/load_nn.py` shows how to load the pretrained d=3, d=5, and d=7 models:

```python
from gru_decoder import GRUDecoder
from data import Args
import torch

args = Args(
    distance=3,
    embedding_features=[5, 32, 64, 128, 256],
    hidden_size=128,
    n_layers=4,
)
decoder = GRUDecoder(args)
decoder.load_state_dict(torch.load("./models/distance3.pt", weights_only=True))
```

## Dependencies

- Python 3.10+
- [Stim](https://github.com/quantumlib/Stim) — stabilizer circuit simulator
- [PyTorch](https://pytorch.org/) — neural network framework
- [PyTorch Geometric](https://pyg.org/) — graph neural networks
- [PyMatching](https://github.com/oscarhiggott/PyMatching) — MWPM decoder baseline
