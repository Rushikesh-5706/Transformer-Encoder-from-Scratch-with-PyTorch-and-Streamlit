# Transformer Encoder from Scratch with PyTorch and Streamlit

A complete implementation of the Transformer encoder architecture — attention, positional encoding, feed-forward network, and layered stack — written entirely in PyTorch primitives, trained on SST-2 sentiment classification, and served through a Streamlit interpretability dashboard.

---

## Overview

This project builds the Transformer encoder from the paper "Attention Is All You Need" (Vaswani et al. 2017) without delegating any attention computation to `torch.nn.MultiheadAttention` or any higher-level abstraction. Every matrix multiplication, scaling operation, softmax, and head-split is written explicitly, which makes the implementation suitable as a reference for understanding what the attention mechanism actually computes, rather than treating it as a black box.

The motivating reason for this approach is interpretability. When attention weights are computed inside `torch.nn.MultiheadAttention`, they are accessible only through hooks or by wrapping the module, and the internal computation is opaque by default. Writing `scaled_dot_product_attention` as a standalone function means the weights are always returned as first-class tensors, ready for downstream visualisation without any instrumentation overhead.

The Streamlit application takes those returned weights and provides three views: raw attention heatmaps per layer and head, attention entropy trajectories across training epochs, and token-level attribution scores aggregated from all layers and heads. These are not retrofitted analytics — they are built into the forward pass from the start.

---

## Architecture

```
Input Tokens
     │
     ▼
Embedding Layer (vocab_size × d_model=128)
     │
     ▼
Positional Encoding (sinusoidal or learned)
     │
     ▼
┌─────────────────────────────────────────┐
│           Encoder Layer × 2             │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │      LayerNorm (pre-norm)       │    │
│  │           │                     │    │
│  │      MultiHeadAttention (×4)    │    │
│  │      W_q  W_k  W_v projections  │    │
│  │           │                     │    │
│  │   scaled_dot_product_attention  │    │
│  │   Q·Kᵀ / √d_k → softmax → ·V    │    │
│  │           │                     │    │
│  │      W_o output projection      │    │
│  │           │                     │    │
│  │      Residual Connection        │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │      LayerNorm (pre-norm)       │    │
│  │      FFN: Linear(128→512)→ReLU  │    │
│  │           Linear(512→128)       │    │
│  │      Residual Connection        │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
     │
     ▼
Final LayerNorm
     │
CLS Token Representation (position 0)
     │
     ▼
Linear Classifier → Logits → Sentiment (Positive / Negative)


Training Artifacts:
  models/final_model.pth       ──► app.py (model weights)
  models/vocab.pt              ──► app.py (word2idx map)
  logs/training_metrics.csv    ──► Entropy Dashboard tab
  snapshots/epoch_1_weights.pt ──► inspection / debugging
  snapshots/final_epoch_weights.pt ──► inspection / debugging
  verification/*.json          ──► shape correctness proofs
```

---

## Model Configuration

| Parameter     | Value                |
|---------------|----------------------|
| d_model       | 128                  |
| num_heads     | 4                    |
| num_layers    | 2                    |
| d_ff          | 512                  |
| dropout       | 0.1                  |
| max_seq_len   | 128                  |
| dataset       | SST-2 (HuggingFace)  |
| optimizer     | Adam with warmup     |
| warmup_steps  | 400                  |
| grad_clip     | 1.0 (L2 norm)        |

---

## Setup and Installation

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
git clone https://github.com/Rushikesh-5706/Transformer-Encoder-from-Scratch-with-PyTorch-and-Streamlit.git
cd Transformer-Encoder-from-Scratch-with-PyTorch-and-Streamlit/transformer-encoder-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and adjust if needed:

```bash
cp .env.example .env
```

The defaults in `.env.example` are CPU-compatible. If a GPU is available, set `DEVICE=cuda` in `.env`.

---

## Running the Training Pipeline

```bash
source .venv/bin/activate
python train.py
```

Training on SST-2 at `BATCH_SIZE=32` with `NUM_EPOCHS=15` takes approximately 25–45 minutes on a modern CPU. On an NVIDIA GPU it runs in 3–5 minutes. The script produces:

- `models/final_model.pth` — final model weights
- `models/vocab.pt` — vocabulary dictionary required by app.py
- `logs/training_metrics.csv` — per-epoch attention entropy for all layers and heads
- `snapshots/epoch_1_weights.pt` — attention weight tensors from validation, saved after epoch 1
- `snapshots/final_epoch_weights.pt` — attention weight tensors from the final validation pass

---

## Running the Streamlit App

**Without Docker:**

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8501
```

Navigate to `http://localhost:8501` in a browser.

**With Docker Compose:**

```bash
docker-compose up --build -d
```

Wait for the container to become healthy (approximately 90 seconds on first start while dependencies load), then access `http://localhost:8501`.

---

## Docker Usage

Build and run the container:

```bash
docker-compose up --build -d
# Check health status
docker ps
# Access the dashboard
# http://localhost:8501
```

Stop and remove:

```bash
docker-compose down
```

Pull and run the pre-built image from Docker Hub:

```bash
docker pull rushi5706/transformer-encoder-scratch:latest
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/snapshots:/app/snapshots \
  -v $(pwd)/verification:/app/verification \
  rushi5706/transformer-encoder-scratch:latest
```

---

## Project Structure

```
transformer-encoder-scratch/
├── app.py                          Streamlit interpretability dashboard
├── model.py                        Transformer encoder implementation
├── train.py                        Training pipeline with entropy logging
├── generate_verification.py        Shape verification script
├── docker-compose.yml              Docker Compose configuration
├── Dockerfile                      Container build instructions
├── .dockerignore                   Docker build exclusions
├── .gitignore                      Git exclusions
├── .env.example                    Environment variable template
├── requirements.txt                Pinned Python dependencies
├── README.md                       This file
├── logs/
│   └── training_metrics.csv        Per-epoch attention entropy (generated)
├── models/
│   ├── final_model.pth             Trained model weights (generated)
│   └── vocab.pt                    Vocabulary mapping (generated)
├── snapshots/
│   ├── epoch_1_weights.pt          Attention weights at epoch 1 (generated)
│   └── final_epoch_weights.pt      Attention weights at final epoch (generated)
├── verification/
│   ├── attention_output.json       MultiHeadAttention shape proof
│   └── encodings_output.json       PositionalEncoding shape proof
└── reports/
    └── attention_head_biography.md Per-head training behaviour analysis
```

---

## Verification Artifacts

`verification/attention_output.json` proves that `MultiHeadAttention(d_model=128, num_heads=4)` produces tensors of the correct shapes given input `(1, 10, 128)`:

```json
{
  "input_shape": [1, 10, 128],
  "output_shape": [1, 10, 128],
  "attention_weights_shape": [1, 4, 10, 10]
}
```

`verification/encodings_output.json` proves that both positional encoding variants produce the expected output shape for input `(1, 20, 128)`:

```json
{
  "sinusoidal_encoding_shape": [1, 20, 128],
  "learned_encoding_shape": [1, 20, 128]
}
```

To regenerate both files:

```bash
python generate_verification.py
```

---

## Interpretability Features

| Dashboard Tab       | What It Shows                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| Attention Heatmap   | Raw softmax attention weights for a selected layer and head on a user sentence |
| Entropy Dashboard   | Per-head entropy at the final epoch; per-epoch trajectory for any layer+head  |
| Token Attribution   | Aggregated attention from CLS query across all layers and heads, per token     |

The heatmap shows which positions each head attends to when processing a user-supplied sentence. The entropy dashboard reveals whether heads have specialised (low entropy) or remain diffuse (high entropy) by the end of training. Token attribution gives a coarse attribution signal for which words most influenced the model's classification decision.

---

## Implementation Notes

### Why not torch.nn.MultiheadAttention?

`torch.nn.MultiheadAttention` in PyTorch:
- Returns attention weights only when `need_weights=True`, and even then, they are averaged across heads by default
- Does not expose the intermediate scores before softmax
- Makes it harder to implement custom masking schemes
- Is opaque when debugging numerical issues

This project implements `scaled_dot_product_attention` as a standalone function:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

Step by step:
1. Compute raw dot-product similarities between queries and keys: `Q @ Kᵀ`
2. Scale by `1/√d_k` to prevent the dot products from growing too large for meaningful softmax gradients
3. Optionally mask out positions (for padding) by filling with -1e9 before softmax, which drives those positions to near-zero attention weight
4. Apply softmax across the key dimension to produce a proper probability distribution
5. Mix the value vectors using those probabilities: the attended output is a weighted sum of values

`MultiHeadAttention` calls this function once for all heads (implemented in parallel via the batch dimension after reshaping), then projects the concatenated output back to `d_model`.
