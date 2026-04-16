# Tucker Transformer Compression

**Goal:** Compress transformer MLP layers using Tucker decomposition from tensor analysis.

**Platform:** GitHub Actions (CPU, 7GB RAM, 6hr limit)

## Why Tucker Decomposition

From tensor analysis (physics/mathematics) — represents tensors with core tensor + factor matrices. Better than SVD for preserving multi-way structure.

**Hypothesis:** 2-4x compression on MLP weights with minimal accuracy loss.

## Current Experiment

| # | Name | Status | Date |
|---|------|--------|------|
| 01 | Baseline Tucker on distilgpt2 | **DESIGNED** | 2026-04-16 |

## Structure

```
src/tucker_compress.py      # Experiment code
.github/workflows/run.yml    # CI runner
results/                     # Auto-committed results
```

## Results

*[Pending first run]*

---

Kiri Research Labs
