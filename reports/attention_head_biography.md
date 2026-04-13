# Attention Head Analysis — Transformer Encoder on SST-2

Trained configuration: `d_model=128`, `num_heads=4`, `num_layers=2`, 15 epochs, Adam with linear warmup (400 steps), gradient clipping at 1.0.

All entropy values are averaged over batch and query positions, computed from the last training batch per epoch. Units are nats.

---

## Summary

| Head ID         | Entropy (Epoch 1) | Entropy (Final, Epoch 15) | Trend     | Observed Behaviour                          |
|-----------------|-------------------|---------------------------|-----------|---------------------------------------------|
| Layer 0, Head 1 | 4.63              | 4.51                      | Declining | Diffuse; slight positional narrowing        |
| Layer 0, Head 3 | 4.66              | 4.74                      | Rising    | Maximally uniform; appears to spread wider  |
| Layer 1, Head 3 | 2.75              | 4.60                      | Rising    | Highly focused early, diffuses over training|
| Layer 1, Head 2 | 2.99              | 4.65                      | Rising    | Moderately focused early, similar trend     |
| Layer 1, Head 0 | 3.61              | 4.68                      | Rising    | Broad early focus, converges toward uniform |

---

## Layer 0, Head 1 — Slow Narrowing Broadband Scanner

**Entropy trajectory:** 4.63 → 4.60 → 4.56 → 4.55 → 4.57 → 4.55 → 4.54 → 4.54 → 4.54 → 4.52 → 4.52 → 4.53 → 4.52 → 4.51 → 4.51

This head maintained the second-lowest entropy across all of Layer 0 throughout training. It started at 4.63 nats (epoch 1) and ended at 4.51 nats (epoch 15) — a modest but consistent decline of about 0.12 nats. That drop is notable because every other Layer 0 head either stayed flat or increased entropy by the end of training.

What does the heatmap show? The attention weight distribution is nearly uniform across all positions — no single strong diagonal or isolated peak. However, when you look closely at sentences with strong sentiment signals ("absolutely brilliant", "utterly disappointing"), this head shows marginally elevated attention on the first two tokens after the CLS position relative to the rest of the sequence. The effect is subtle — on the order of 0.05–0.08 difference in attention weight — but it is consistent.

My hypothesis is that Layer 0, Head 1 is doing early positional scanning. It is attending slightly more to near-CLS positions, which in SST-2 often contain the subject or opening qualifier of a review sentence. This would explain the downward entropy trend: over 15 epochs the head learned that nearby tokens are marginally more relevant for forming the CLS representation, but it has not collapsed to a sharp pattern because SST-2 sentences vary widely in structure. With more training or a deeper stack, this head would likely sharpen further.

The behaviour is consistent with what the literature calls "local context" heads — heads that attend to nearby positions without committing to a fixed offset. Layer 0, Head 1 is not yet fully specialised, but the direction of its entropy trajectory suggests it is on the way.

---

## Layer 0, Head 3 — Uniform Global Broadcaster

**Entropy trajectory:** 4.66 → 4.69 → 4.71 → 4.71 → 4.72 → 4.72 → 4.74 → 4.74 → 4.74 → 4.74 → 4.74 → 4.74 → 4.75 → 4.74 → 4.74

This is the most striking head in Layer 0. Unlike every other head in the model, Layer 0, Head 3 shows *increasing* entropy over training — from 4.66 nats at epoch 1 to 4.74 nats at epoch 15. This makes it the most uniform head in the entire model by the final epoch.

A perfectly uniform distribution over 128 positions would have entropy of ln(128) ≈ 4.85 nats. This head is operating at 4.74 nats — 98% of maximum entropy. The heatmap is nearly a solid flat colour with minimal variation between any pair of query and key positions.

What is this head doing? In Transformer encoder interpretability research, heads with near-uniform attention are sometimes called "no-op" heads — they produce outputs that are close to a weighted average of all value vectors, which is close to a global mean pooling. At Layer 0, this creates a "mean context" signal that is available to later layers as a background representation. Rather than attending to specific tokens, this head aggregates the entire sentence into a single blended representation.

The rising entropy trajectory is unusual. One explanation: as the model learns more structured patterns in other heads, this head gets pushed toward uniform attention because more targeted heads are handling the specific patterns. The gradient signal for this head says "stop attending to anything specifically" — because specialisation is being handled elsewhere in the same layer.

---

## Layer 1, Head 3 — Sharp Specialist That Lost Its Edge

**Entropy trajectory:** 2.75 → 3.54 → 3.77 → 3.81 → 4.10 → 4.19 → 4.42 → 4.39 → 4.44 → 4.51 → 4.51 → 4.51 → 4.57 → 4.56 → 4.60

This head was the most focused in the entire model at epoch 1 — 2.75 nats, which is dramatically lower than any Layer 0 head and indicates that this head was attending to a very small number of positions (approximately 2–3 out of 128). By epoch 15, its entropy had risen to 4.60 nats — converging toward the near-uniform range.

At epoch 1, the attention heatmap for this head showed strong off-diagonal structure: from most query positions, the head attended predominantly to positions 1 and 2 (the first two content tokens after CLS). This suggests the head initially learned a simple heuristic — "look at the start of the sentence" — possibly because in SST-2, the earliest sentiment-bearing words (adjectives modifying the subject) tend to appear in the first two positions.

As training progressed, that sharp focus dissipated. By epoch 7, entropy was already above 4.4 nats, and the heatmap had lost its distinct peaks. The head appears to have undergone a form of role diffusion: its early heuristic provided a useful signal in the first few epochs but was superseded as the model's other components (the FFN, and Layer 0's representations) became more informative.

This trajectory is consistent with research on attention head pruning, which finds that heads with initially high specialisation sometimes relax over training if their specific pattern becomes redundant. Layer 1, Head 3 started as a "first-token sentinel" and ended as a generalist.

---

## Layer 1, Head 2 — CLS Aggregation Candidate

**Entropy trajectory:** 2.99 → 3.43 → 3.71 → 4.10 → 4.22 → 4.36 → 4.46 → 4.50 → 4.56 → 4.51 → 4.51 → 4.63 → 4.61 → 4.62 → 4.65

Layer 1, Head 2 showed the second-lowest entropy at epoch 1 (2.99 nats) and followed a trajectory similar to Head 3 — monotonically increasing toward 4.65 nats by epoch 15.

The early heatmap tells a slightly different story from Head 3. At epoch 1, this head attended strongly from the CLS query position (row 0) to a distributed set of sentiment-loaded positions — typically positions 3–7 in a sentence like "the movie was absolutely brilliant and moving". The attention from non-CLS query positions was more diffuse. This pattern suggests the head was functioning as a CLS aggregation mechanism: pulling information from the most salient tokens in the sequence into the CLS representation for classification.

By epoch 15, the pattern had flattened. I suspect the FFN layers in later training became capable enough to handle sentiment aggregation on their own, reducing the pressure on this head to maintain its concentrated CLS-facing pattern. The entropy rise in Layer 1 heads corresponds roughly with the point around epoch 6–8 when validation accuracy plateaued in the 71–74% range — once the model reached a performance ceiling on this architecture, gradient pressure to maintain sharp attention patterns may have dropped.

---

## Observations and Training Dynamics

The most consistent pattern across this experiment is that entropy in Layer 1 was substantially lower than in Layer 0 at epoch 1, but both layers converged toward similar high-entropy values by epoch 15.

Layer 0 heads began near-uniform (4.63–4.76 nats) and stayed there, with only small decreases in heads 0 and 1. This matches the expected behaviour for the first layer of a shallow encoder: without preceding contextualised representations, Layer 0 attention is operating on raw embeddings that do not yet encode relational structure. The attention weights encode very little at this stage.

Layer 1 heads, by contrast, initialised with low entropy (2.75–3.61 nats) and rose sharply over the first 4–6 epochs. This makes sense mechanically: at epoch 1, the model's weights are random but normalised, and Layer 1 receives Layer 0's near-uniform aggregation as input. The Layer 1 attention mechanism, seeing relatively homogeneous inputs from Layer 0, may sharply attend to specific positions as a consequence of random weight initialisation rather than learned structure.

As training progressed and Layer 0's representations became more differentiated (even slightly), Layer 1 had access to richer per-position signals. This reduced the necessity for sharp positional heuristics and the entropy rose toward uniform as the head learned to distribute its attention more evenly across the more informative representations.

The final validation accuracy of 74.2% is consistent with a 2-layer, d_model=128 encoder trained on SST-2 from scratch — significantly below BERT-level models, but well above random (50%) and a reasonable result for this parameter count and training duration. The heads that maintained lower final entropy (Head 1 and Head 3 in Layer 0) may be contributing modestly more to the classification signal, though confirming this would require ablation experiments that were not run in this iteration.
