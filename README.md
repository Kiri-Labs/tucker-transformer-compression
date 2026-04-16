# Tucker Transformer Compression

**Status:** Experiment 01 Complete | 4× compression achieved

Tucker decomposition for transformer MLP layer compression. From tensor analysis (physics/mathematics).

## Results

| Metric | Value |
|--------|-------|
| Model | distilgpt2 (82M params) |
| MLP Compression | **4.01×** (108 MB → 27 MB) |
| Layers Compressed | 12 (6 blocks × 2 projections) |
| Baseline Perplexity | 231.67 |
| Compressed Perplexity | 10,347.96 |
| Quality Impact | **4367% degradation** — unacceptable without fine-tuning |

## Key Finding

Tucker/SVD compression works mechanically but causes massive perplexity increase at 4×. Without post-compression fine-tuning, this approach is unsuitable for LLMs.

## Repository

```
src/tucker_compress.py      # Truncated SVD implementation
results/01-tucker-baseline.json  # Full results
.github/workflows/run.yml   # CI runner
```

## Next Direction

Try lower compression ratios (2× instead of 4×) or add fine-tuning step after compression.

---

Kiri Research Labs
