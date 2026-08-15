# LoRA vs DoRA: Mechanism to Experiment

This repo has two parts:

- `from-scratch/` — minimal NumPy/PyTorch implementations of LoRA and DoRA 
  on a single linear layer, built to understand the mechanism directly 
  (not a performance benchmark).
- `experiment/` — the actual replication: LoRA and DoRA (rank 8) fine-tuned 
  on Spider text-to-SQL using a 1.5B model, with full results, per-example 
  analysis, and an honest writeup of what did (and didn't) show up.
- `LoRA/` — the trained adapter weights from the experiment for LoRA.
- `DoRA/` - the trained adapter weights from the experiment for DoRA.

See `experiment/README.md` for the full writeup and findings.