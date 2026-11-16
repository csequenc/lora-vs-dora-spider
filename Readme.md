# LoRA vs. DoRA: From Mechanism to Experiment

Do DoRA's reported gains over LoRA show up at small scale? This repository has two parts: a from-scratch implementation of both methods to understand the mechanism, and a controlled experiment fine-tuning Qwen2.5-1.5B on Spider text-to-SQL.

## Finding

- At rank 8, with 3,000 training examples, there was **no consistent accuracy difference** between LoRA and DoRA across three seeds. The gap ranged from −0.3 to +4.0 points (LoRA minus DoRA) and changed sign between seeds.
- **DoRA trained more slowly:** about 10% longer in the seed-44 runs (3h 04m vs 3h 22m).
- Limits: one model, one task, rank 8 only, 300 evaluation examples. The study cannot resolve differences of a few points.

| | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| LoRA (execution accuracy, n=300) | 0.4233 | 0.4567 | 0.4233 |
| DoRA (execution accuracy, n=300) | 0.4133 | 0.4167 | 0.4267 |

Full methodology, per-example analysis and limitations are in [`experiment/README.md`](experiment/README.md).

## Repository layout

| Path | Contents |
|------|----------|
| `from-scratch/` | Minimal NumPy/PyTorch implementations of LoRA and DoRA on a single linear layer, written to understand the mechanism directly. Not a performance benchmark. |
| `experiment/` | The comparison on Spider text-to-SQL: notebook, evaluation scripts, per-seed results, analysis and write-up. |
| `LoRA/` | Trained LoRA adapter weights from the experiment. |
| `DoRA/` | Trained DoRA adapter weights from the experiment. |
| `tokenizer/` | Tokenizer files used with the adapters. |

> Note: state here which seed the saved adapters in `LoRA/` and `DoRA/` come from.

## Quick start

1. Open `experiment/lora_vs_dora_spider.ipynb` and set `RANK`, `USE_DORA` and `SEED` in the config cell. Run top to bottom.
2. To evaluate an already trained adapter without retraining, run `experiment/evaluate.py`.
3. To rebuild the per-seed comparison tables, run `experiment/compare_seeds.py`.

See [`experiment/README.md`](experiment/README.md) for details.

## Setup in brief

- Base model: Qwen2.5-1.5B (4-bit NF4 via bitsandbytes)
- Task and data: Spider text-to-SQL, 3,000 training examples, 300 dev examples for evaluation
- Libraries: PyTorch, Hugging Face Transformers, PEFT
- Metric: execution accuracy

## References

- Liu et al., 2024. DoRA: Weight-Decomposed Low-Rank Adaptation.
- Hu et al., 2021. LoRA: Low-Rank Adaptation of Large Language Models.
- Yu et al., 2018. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.
