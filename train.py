"""
Training pipeline for the Transformer Encoder on SST-2 sentiment classification.

Loads the SST-2 dataset from HuggingFace datasets, builds a whitespace-tokenised
vocabulary, trains using Adam with a linear warmup + inverse-sqrt decay schedule,
clips gradients at norm 1.0, logs attention entropy per layer and head to CSV,
saves weight snapshots, and stores the final model and vocabulary for use in app.py.
"""

import csv
import logging
import math
import os

# Redirect HuggingFace cache to a project-local directory to avoid system
# permission restrictions that can occur when ~/.cache/huggingface is not writable.
os.environ.setdefault('HF_HOME', os.path.join(os.path.dirname(__file__), '.hf_cache'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from model import TransformerEncoder

# ---------------------------------------------------------------------------
# Logging configuration — use the standard library logger, never bare print()
# in training loops so that log levels can be controlled at runtime.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters — read from environment so Docker/CI can override without
# touching source code.
# ---------------------------------------------------------------------------
DEVICE = torch.device(os.environ.get("DEVICE", "cpu"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 15))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
MAX_VOCAB_SIZE = int(os.environ.get("MAX_VOCAB_SIZE", 10000))

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
MAX_SEQ_LEN = 128
NUM_CLASSES = 2
WARMUP_STEPS = 400
GRAD_CLIP_NORM = 1.0
SNAPSHOT_EPOCHS = {1}


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocab(texts, max_size):
    """
    Build a word-to-index mapping from a list of raw text strings.

    Tokenisation is kept intentionally simple (whitespace split + lowercase)
    to avoid introducing a tokeniser dependency that would complicate the
    Docker image. Special tokens follow the BERT convention so that app.py
    can prepend [CLS] in the same way train.py does.

    Args:
        texts: Iterable of raw sentence strings.
        max_size: Maximum vocabulary size including special tokens.

    Returns:
        word2idx: Dict mapping token string to integer index.
    """
    counter = Counter()
    for text in texts:
        for token in text.lower().split():
            counter[token] += 1

    word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
    for token, _ in counter.most_common(max_size - 4):
        word2idx[token] = len(word2idx)

    return word2idx


def tokenize_and_encode(text, word2idx, max_len):
    """
    Convert a raw sentence into a fixed-length integer sequence.

    Prepends the CLS token, maps words to indices, truncates to max_len,
    and zero-pads shorter sequences. This function is the canonical tokeniser
    for both training and inference — app.py copies this exact logic so that
    inference does not depend on importing train.py.

    Args:
        text: Raw input sentence string.
        word2idx: Vocabulary mapping from build_vocab().
        max_len: Target sequence length including the CLS token.

    Returns:
        List of integer token indices of length exactly max_len.
    """
    tokens = text.lower().split()
    cls_idx = word2idx.get('<CLS>', 2)
    unk_idx = word2idx.get('<UNK>', 1)
    pad_idx = word2idx.get('<PAD>', 0)

    encoded = [cls_idx]
    for token in tokens[: max_len - 1]:
        encoded.append(word2idx.get(token, unk_idx))

    # Pad to max_len
    encoded += [pad_idx] * (max_len - len(encoded))
    return encoded


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class SSTDataset(Dataset):
    """
    Wraps the HuggingFace SST-2 dataset split into a PyTorch Dataset.

    SST-2 uses the column names 'sentence' (text) and 'label' (0 or 1),
    unlike generic datasets that use 'text'. This class handles that naming
    explicitly rather than relying on a generic field name.

    Args:
        data: HuggingFace dataset split (e.g. load_dataset('sst2')['train']).
        word2idx: Vocabulary mapping from build_vocab().
        max_len: Maximum sequence length including the CLS prefix.
    """

    def __init__(self, data, word2idx, max_len):
        """Store dataset split, vocabulary, and maximum sequence length."""
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        """Return number of examples in this split."""
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenise and encode one example, returning (input_ids tensor, label tensor)."""
        item = self.data[idx]
        # SST-2 uses 'sentence', not 'text'
        text = item['sentence']
        label = item['label']
        input_ids = tokenize_and_encode(text, self.word2idx, self.max_len)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def get_warmup_scheduler(optimizer, warmup_steps):
    """
    Build a LambdaLR scheduler implementing the Transformer paper's schedule.

    Learning rate increases linearly from 0 to the base rate over warmup_steps,
    then decays proportionally to 1/sqrt(step). This schedule is from Section 5.3
    of Vaswani et al. and helps stabilise training in the early phases when
    gradient estimates are noisy.

    Args:
        optimizer: The Adam optimizer to attach the schedule to.
        warmup_steps: Number of steps for the linear warm-up phase.

    Returns:
        A torch.optim.lr_scheduler.LambdaLR scheduler.
    """
    def lr_lambda(current_step):
        """Return the learning rate multiplier for the given step."""
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return 1.0 / math.sqrt(current_step + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def compute_attention_entropy(attn_weights):
    """
    Compute per-head attention entropy averaged over batch and query positions.

    Entropy measures how diffuse or focused a head's attention distribution is.
    A head with low entropy after training has specialised to attend sharply to
    specific positions; high entropy indicates it still spreads attention broadly.
    Tracking this over training epochs reveals how and when heads specialise.

    Args:
        attn_weights: Tensor of shape (batch, num_heads, seq_q, seq_k).

    Returns:
        List of float entropy values, one per head в shape dimension 1.
    """
    # attn_weights: (batch, num_heads, seq, seq)
    # Entropy per position: - sum(p * log(p + eps))
    entropy_per_position = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1)
    # Average over batch and query positions -> (num_heads,)
    mean_entropy = entropy_per_position.mean(dim=(0, 2))
    return mean_entropy.tolist()


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logger.info("Loading SST-2 dataset from HuggingFace...")
    dataset = load_dataset('sst2')
    train_data = dataset['train']
    val_data = dataset['validation']

    logger.info(f"Train examples: {len(train_data)}  |  Validation examples: {len(val_data)}")

    # Build vocabulary from training sentences only (no peeking at validation)
    logger.info(f"Building vocabulary (max size {MAX_VOCAB_SIZE})...")
    train_texts = [item['sentence'] for item in train_data]
    word2idx = build_vocab(train_texts, MAX_VOCAB_SIZE)
    logger.info(f"Vocabulary size: {len(word2idx)}")

    # Save vocabulary so app.py can reconstruct it without importing train.py
    os.makedirs('models', exist_ok=True)
    torch.save({'word2idx': word2idx}, 'models/vocab.pt')
    logger.info("Vocabulary saved to models/vocab.pt")

    # Datasets and loaders
    train_dataset = SSTDataset(train_data, word2idx, MAX_SEQ_LEN)
    val_dataset = SSTDataset(val_data, word2idx, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = TransformerEncoder(
        vocab_size=len(word2idx),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        encoding_type='sinusoidal',
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_warmup_scheduler(optimizer, WARMUP_STEPS)

    # CSV logging setup — keep file open across the full training loop
    os.makedirs('logs', exist_ok=True)
    os.makedirs('snapshots', exist_ok=True)

    csv_path = 'logs/training_metrics.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'layer', 'head', 'attention_entropy'])

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        last_batch_attn_weights = None

        for batch_input_ids, batch_labels in train_loader:
            batch_input_ids = batch_input_ids.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits, all_attn_weights = model(batch_input_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * batch_labels.size(0)
            correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            total += batch_labels.size(0)
            last_batch_attn_weights = [w.detach().cpu() for w in all_attn_weights]

        train_loss = total_loss / total
        train_acc = correct / total

        # -------------------------------------------------------------------
        # Validation pass
        # -------------------------------------------------------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_attn_weights = None

        with torch.no_grad():
            for batch_input_ids, batch_labels in val_loader:
                batch_input_ids = batch_input_ids.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                logits, all_attn_weights = model(batch_input_ids)
                val_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
                val_total += batch_labels.size(0)
                if val_attn_weights is None:
                    val_attn_weights = [w.detach().cpu() for w in all_attn_weights]

        val_acc = val_correct / val_total

        logger.info(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

        # -------------------------------------------------------------------
        # Log attention entropy for every layer and head
        # -------------------------------------------------------------------
        weights_for_entropy = last_batch_attn_weights if last_batch_attn_weights else val_attn_weights
        for layer_idx, layer_weights in enumerate(weights_for_entropy):
            head_entropies = compute_attention_entropy(layer_weights)
            for head_idx, entropy_value in enumerate(head_entropies):
                csv_writer.writerow([epoch, layer_idx, head_idx, round(entropy_value, 6)])

        csv_file.flush()

        # -------------------------------------------------------------------
        # Epoch snapshots — save attention weights from the first validation batch
        # -------------------------------------------------------------------
        if epoch in SNAPSHOT_EPOCHS:
            snapshot_path = f'snapshots/epoch_{epoch}_weights.pt'
            torch.save(val_attn_weights, snapshot_path)
            logger.info(f"Saved attention snapshot: {snapshot_path}")

    # Save final epoch snapshot regardless of epoch number
    torch.save(val_attn_weights, 'snapshots/final_epoch_weights.pt')
    logger.info("Saved final epoch attention snapshot: snapshots/final_epoch_weights.pt")

    csv_file.close()
    logger.info("Training metrics CSV closed.")

    # Save final model weights
    torch.save(model.state_dict(), 'models/final_model.pth')
    logger.info("Model saved to models/final_model.pth")
    logger.info("Training complete.")
