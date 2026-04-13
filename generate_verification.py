"""
Verification script for shape correctness of MultiHeadAttention and PositionalEncoding.

Run this script immediately after writing model.py to confirm that the attention
output and positional encoding shapes match the expected values before training.
Uses a fixed random seed for fully deterministic output.
"""

import json
import os
import torch

torch.manual_seed(42)

# Ensure verification directory exists
os.makedirs('verification', exist_ok=True)

from model import MultiHeadAttention, PositionalEncoding


def verify_attention_shapes():
    """
    Instantiate MultiHeadAttention with d_model=128, num_heads=4 and run a
    forward pass with a (1, 10, 128) dummy input as Q, K, and V.

    Expected:
        output shape:          [1, 10, 128]
        attention_weights shape: [1, 4, 10, 10]
    """
    mha = MultiHeadAttention(d_model=128, num_heads=4)
    mha.eval()

    dummy_input = torch.randn(1, 10, 128)
    with torch.no_grad():
        output, attn_weights = mha(dummy_input, dummy_input, dummy_input)

    result = {
        "input_shape": list(dummy_input.shape),
        "output_shape": list(output.shape),
        "attention_weights_shape": list(attn_weights.shape),
    }

    output_path = 'verification/attention_output.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"attention_output.json written: {result}")
    return result


def verify_encoding_shapes():
    """
    Instantiate both PositionalEncoding variants and run a forward pass with
    a (1, 20, 128) dummy input.

    Expected:
        sinusoidal_encoding_shape: [1, 20, 128]
        learned_encoding_shape:    [1, 20, 128]
    """
    sinusoidal_pe = PositionalEncoding(d_model=128, max_len=512, encoding_type='sinusoidal')
    sinusoidal_pe.eval()

    learned_pe = PositionalEncoding(d_model=128, max_len=512, encoding_type='learned')
    learned_pe.eval()

    dummy_input = torch.randn(1, 20, 128)
    with torch.no_grad():
        sinusoidal_out = sinusoidal_pe(dummy_input)
        learned_out = learned_pe(dummy_input)

    result = {
        "sinusoidal_encoding_shape": list(sinusoidal_out.shape),
        "learned_encoding_shape": list(learned_out.shape),
    }

    output_path = 'verification/encodings_output.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"encodings_output.json written: {result}")
    return result


if __name__ == '__main__':
    print("Running shape verification for model components...")

    attn_result = verify_attention_shapes()
    enc_result = verify_encoding_shapes()

    # Assert exact expected shapes before declaring success
    assert attn_result["input_shape"] == [1, 10, 128], "input_shape mismatch"
    assert attn_result["output_shape"] == [1, 10, 128], "output_shape mismatch"
    assert attn_result["attention_weights_shape"] == [1, 4, 10, 10], "attn_weights_shape mismatch"

    assert enc_result["sinusoidal_encoding_shape"] == [1, 20, 128], "sinusoidal shape mismatch"
    assert enc_result["learned_encoding_shape"] == [1, 20, 128], "learned shape mismatch"

    print("\nAll shape assertions passed.")
    print("  verification/attention_output.json ✓")
    print("  verification/encodings_output.json ✓")
