"""
Transformer Encoder implementation from scratch using PyTorch primitives.

This module implements every component of the Transformer encoder architecture
as described in "Attention Is All You Need" (Vaswani et al. 2017), without
delegating any attention computation to torch.nn.MultiheadAttention. The goal
is full visibility into the mechanism so that downstream interpretability tools
can access attention weights at every layer and head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention for a batch of query, key, value tensors.

    This is the core attention operation from Vaswani et al. Section 3.2.1.
    The scaling factor 1/sqrt(d_k) prevents the dot products from growing large
    in magnitude when d_k is large, which would push softmax into regions with
    extremely small gradients.

    Args:
        Q: Query tensor of shape (batch, num_heads, seq_q, d_k)
        K: Key tensor of shape (batch, num_heads, seq_k, d_k)
        V: Value tensor of shape (batch, num_heads, seq_v, d_v) where seq_v == seq_k
        mask: Optional boolean mask of shape (batch, 1, seq_q, seq_k) or compatible.
              Positions where mask == 0 are filled with -inf before softmax.

    Returns:
        output: Attended values, shape (batch, num_heads, seq_q, d_v)
        attn_weights: Softmax attention distribution, shape (batch, num_heads, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module implemented without torch.nn.MultiheadAttention.

    Projects Q, K, V into num_heads separate subspaces, applies scaled dot-product
    attention independently in each, then concatenates and projects back to d_model.
    This allows the model to jointly attend to information from different representation
    subspaces at different positions.

    Args:
        d_model: Dimensionality of the model's embedding space.
        num_heads: Number of parallel attention heads. d_model must be divisible by num_heads.
    """

    def __init__(self, d_model, num_heads):
        """Initialise projection layers and store head configuration."""
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Separate linear projections for queries, keys, values, and output.
        # No bias follows the original paper convention; it also reduces overfitting
        # on smaller datasets.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        """
        Compute multi-head attention.

        Args:
            Q: Query tensor, shape (batch, seq_q, d_model)
            K: Key tensor, shape (batch, seq_k, d_model)
            V: Value tensor, shape (batch, seq_v, d_model), seq_v == seq_k
            mask: Optional attention mask, shape broadcastable to (batch, num_heads, seq_q, seq_k)

        Returns:
            output: Projected attended values, shape (batch, seq_q, d_model)
            attn_weights: Attention distributions, shape (batch, num_heads, seq_q, seq_k)
        """
        batch_size = Q.size(0)

        # Linear projections into the full d_model space
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split d_model into num_heads separate d_head subspaces.
        # (batch, seq, d_model) -> (batch, seq, num_heads, d_head) -> (batch, num_heads, seq, d_head)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Compute attention in all heads simultaneously
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Recombine heads: (batch, num_heads, seq, d_head) -> (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final output projection
        output = self.W_o(attn_out)

        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    Injects positional information into token embeddings.

    Supports two encoding strategies:
    - 'sinusoidal': Deterministic, fixed encoding using sine/cosine functions.
      Generalises to sequence lengths beyond those seen during training.
    - 'learned': An nn.Embedding table trained end-to-end with the model.
      Can express more complex positional patterns but is tied to max_len.

    Args:
        d_model: Embedding dimensionality — must match the token embedding dimension.
        max_len: Maximum sequence length to support.
        encoding_type: Either 'sinusoidal' or 'learned'.
    """

    def __init__(self, d_model, max_len, encoding_type='sinusoidal'):
        """Build or register the positional encoding table based on encoding_type."""
        super().__init__()
        self.encoding_type = encoding_type

        if encoding_type == 'sinusoidal':
            # Build a (1, max_len, d_model) encoding matrix and register it as a
            # non-trainable buffer so it moves to the correct device automatically.
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            # Frequency terms: 10000^(2i/d_model) in the denominator
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)

        elif encoding_type == 'learned':
            self.pe = nn.Embedding(max_len, d_model)

        else:
            raise ValueError(f"encoding_type must be 'sinusoidal' or 'learned', got '{encoding_type}'")

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.

        Args:
            x: Token embeddings, shape (batch, seq_len, d_model)

        Returns:
            x with positional information added, same shape as input.
        """
        if self.encoding_type == 'sinusoidal':
            return x + self.pe[:, :x.size(1), :]
        else:
            positions = torch.arange(0, x.size(1), device=x.device)
            return x + self.pe(positions)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network applied identically at each sequence position.

    Consists of two linear transformations with a ReLU activation in between,
    following the formulation from Vaswani et al. Section 3.3:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    The inner dimension d_ff is typically 4x d_model, providing a large
    context-free transformation capacity between attention layers.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Inner layer dimensionality.
        dropout: Dropout rate applied after the activation.
    """

    def __init__(self, d_model, d_ff, dropout):
        """Initialise the two linear layers and dropout."""
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply the feed-forward transformation.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Transformed tensor, shape (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer.

    Implements the pre-norm variant (LayerNorm before the sub-layer rather than
    after), which provides more stable gradients during deep network training
    compared to the original post-norm formulation in Vaswani et al.

    Each layer applies:
    1. LayerNorm -> Multi-head self-attention -> residual add
    2. LayerNorm -> Feed-forward network -> residual add

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout probability applied to sub-layer outputs before residual addition.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """Initialise attention, FFN, layer norms, and dropout for one encoder layer."""
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply one encoder layer with pre-norm residual connections.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Optional attention mask.

        Returns:
            x: Updated representation, shape (batch, seq_len, d_model)
            attn_weights: Attention distributions from this layer's self-attention.
        """
        # Sub-layer 1: self-attention with pre-norm and residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)

        # Sub-layer 2: feed-forward with pre-norm and residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Full Transformer encoder for sequence classification.

    Stacks num_layers encoder layers on top of token + positional embeddings.
    Uses the [CLS] token representation (position 0) for classification, following
    the BERT convention. The first token must be the CLS token index (2) in the
    vocabulary built by train.py.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding and model dimensionality.
        num_heads: Number of attention heads per encoder layer.
        num_layers: Number of stacked encoder layers.
        d_ff: Feed-forward inner dimensionality.
        max_len: Maximum supported sequence length.
        dropout: Dropout probability throughout the model.
        num_classes: Number of output classes for classification.
        encoding_type: Positional encoding strategy — 'sinusoidal' or 'learned'.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_len,
        dropout,
        num_classes,
        encoding_type='sinusoidal',
    ):
        """Initialise embedding, positional encoding, encoder layers, and classifier."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, encoding_type)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, mask=None):
        """
        Encode a batch of token sequences and produce classification logits.

        Args:
            src: Integer token indices, shape (batch, seq_len). The first token
                 in every sequence must be the CLS index for classification to work.
            mask: Optional attention mask, shape broadcastable to
                  (batch, num_heads, seq_len, seq_len).

        Returns:
            logits: Class scores, shape (batch, num_classes)
            all_attn_weights: List of length num_layers, each element a tensor
                              of shape (batch, num_heads, seq_len, seq_len).
                              Used for interpretability visualisations downstream.
        """
        # Embed tokens and add positional information
        x = self.dropout(self.pos_encoding(self.embedding(src)))

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        x = self.norm(x)

        # Extract the CLS token representation at position 0 for classification
        cls_repr = x[:, 0, :]
        logits = self.classifier(cls_repr)

        return logits, all_attn_weights
