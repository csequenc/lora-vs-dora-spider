# LoRA vs. DoRA at Small Scale: A Rank-8 Case Study on Text-to-SQL

## Motivation

LoRA freezes a model's pretrained weights and injects a trainable low-rank update (`W = W₀ + BA`), adapting far fewer parameters than full fine-tuning. DoRA (Liu et al., 2024) modifies this by decomposing the pretrained weight into a magnitude component and a direction component, applying the LoRA-style low-rank update only to direction, while letting magnitude adapt freely and independently. The motivation, per the original paper, is that this more closely mirrors how full fine-tuning updates weights — and DoRA is reported to outperform LoRA on several benchmarks at typical training scales.

Most published DoRA results are on larger models and larger training budgets. This project asks a narrower, more practical question: **does DoRA's advantage over LoRA hold up at small scale** — a 1.5B-parameter model, a few thousand fine-tuning examples, a single consumer GPU? That's the regime most individual practitioners and students actually operate in, and it's not the regime the original paper was testing.

## Setup

- **Base model:** Qwen2.5-1.5B, loaded in 4-bit (NF4) via `bitsandbytes`
- **Task:** Text-to-SQL on the Spider dataset — natural language question + database schema → SQL query
- **Training data:** 3,000 examples from Spider's training split
- **Adapters compared:** LoRA and DoRA, both rank 8, `lora_alpha=16`, targeting `q_proj/k_proj/v_proj/o_proj`, identical hyperparameters otherwise (learning rate, batch size, epochs) — the only difference between runs is the `use_dora` flag in PEFT's `LoraConfig`
- **Evaluation:** Execution accuracy on 300 examples from Spider's dev set — a predicted query counts as correct if running it against the actual database returns the same result set as the gold query, regardless of surface-level SQL differences (the standard metric in text-to-SQL literature)
- **Seeds:** three seeds per method (42, 43, 44) — see Seed Robustness Check below

## Results (seed 42)

| Rank | Method | Execution Accuracy (n=300) |
|---|---|---|
| 8 | LoRA | 0.4233 |
| 8 | DoRA | 0.4133 |

A roughly 1-point gap, with LoRA slightly ahead. On its own, this number doesn't say much — a single-seed, 300-example comparison can easily produce a gap this size from noise alone, which is exactly what running additional seeds below confirms.

### Per-example agreement (seed 42)

| | Count | % of 300 |
|---|---|---|
| Both correct | 110 | 36.7% |
| Both wrong | 159 | 53.0% |
| LoRA correct, DoRA wrong | 17 | 5.7% |
| DoRA correct, LoRA wrong | 14 | 4.7% |

LoRA and DoRA agree — both right or both wrong — on **89.7%** of examples. The entire aggregate gap comes down to a net difference of 3 examples out of 300.

### Breakdown by query complexity (seed 42)

Spider's HuggingFace mirror (`xlangai/spider`) doesn't carry the dataset's original human-assigned difficulty labels, so difficulty here is a structural proxy computed from each gold query: a count of joins, subqueries, aggregation functions, `GROUP BY`/`HAVING` clauses, boolean conditions, and set operations, bucketed into simple / moderate / complex.

| Complexity | N | LoRA acc | DoRA acc |
|---|---|---|---|
| Simple | 49 | 0.429 | 0.388 |
| Moderate | 139 | 0.540 | 0.554 |
| Complex | 112 | 0.277 | 0.250 |

If DoRA's magnitude/direction decoupling gave it a real edge on harder queries, we'd expect its advantage to grow with complexity. Instead, LoRA is (slightly) ahead in the complex bucket, and the gaps at every level are within the noise band suggested by the per-example agreement numbers above.

## Seed robustness check

To test whether the seed-42 gap reflected a real method difference or ordinary training noise, both adapters were retrained and re-evaluated from scratch across two additional seeds (43, 44) — same hyperparameters, same eval set, only the seed changed.

| | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| LoRA | 0.4233 | 0.4567 | 0.4233 |
| DoRA | 0.4133 | 0.4167 | 0.4267 |
| Gap (LoRA − DoRA) | +1.0 pts | +4.0 pts | **−0.3 pts** |

The gap doesn't just vary in size across seeds — it **flips sign** in seed 44, where DoRA is marginally ahead. LoRA's own accuracy alone swings by 3.3 points across seeds (0.4233 → 0.4567 → 0.4233), which is larger than every individual LoRA-vs-DoRA gap observed. Across three seeds, LoRA is ahead twice and DoRA once, with no consistent direction.

**Conclusion of the seed check:** this confirms, with considerably more confidence than a single seed allows, that at rank 8 on this model/task/scale, there is no reliable accuracy advantage for either method. The seed-42 gap reported above — and any single-seed LoRA-vs-DoRA comparison at this scale in general — is training noise, not a real effect.

Per-example agreement held a similar shape across seeds 43 and 44 as well (roughly 88-90% agreement both times), and the complexity-bucket breakdown showed no seed produced a consistent complexity-dependent pattern favoring either method.

## Training cost

Unlike accuracy, training time showed a **consistent, structural difference** across all three seeds: DoRA reliably took longer to train than LoRA.

| | LoRA | DoRA |
|---|---|---|
| Training time (seed 44) | ~3h 04m | ~3h 22m |
| Throughput | ~0.05 it/s | ~0.04 it/s |

DoRA was roughly 18-30 minutes slower per run and ran at about 20% lower iteration throughput, consistently across seeds. This matches expectations from the method itself: DoRA computes a column-wise norm of the updated weight matrix at every forward pass, and its backward pass is more involved than LoRA's plain low-rank update (see Section 4.2-4.3 of the DoRA paper, which explicitly discusses this overhead and a mitigation the authors apply to reduce it — not used here, since this project used the default PEFT `use_dora` implementation without that modification).

This is a genuinely useful, reproducible finding on its own: even where DoRA shows no measurable accuracy benefit at this scale, it carries a measurable, repeatable compute cost. Anyone choosing between the two methods at small scale should weigh that cost against an unproven benefit.

### What the disagreements actually look like

A few representative cases from seed 42's disagreements:

**DoRA correct, LoRA wrong** — LoRA hallucinated a plausible-but-wrong column name:
> Q: *Find the number of pets whose weight is heavier than 10.*
> Gold: `SELECT count(*) FROM pets WHERE weight > 10`
> LoRA: `SELECT count(*) FROM pets WHERE pet_weight > 10`
> DoRA: `SELECT count(*) FROM pets WHERE weight > 10` ✓

**LoRA correct, DoRA wrong** — DoRA substituted the wrong set operation:
> Q: *Show countries where a singer above age 40 and a singer below 30 are from.*
> Gold: `... WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30`
> LoRA: `... INTERSECT SELECT Country FROM singer WHERE Age < 30` ✓
> DoRA: `... UNION SELECT Country FROM singer WHERE Age < 30`

Both methods make similar categories of mistakes overall — unnecessary joins on single-table questions, wrong column names, wrong SQL keywords/operators — across all three seeds, with no consistent pattern in which method makes which kind of mistake.

## Conclusion

At rank 8, on a 1.5B model, with 3,000 fine-tuning examples on Spider text-to-SQL, across three independent seeds: **DoRA shows no consistent accuracy advantage over LoRA, and the direction of any observed gap is not reproducible** — LoRA leads in two of three seeds, DoRA in one, by margins smaller than LoRA's own seed-to-seed variance. This doesn't contradict DoRA's published results, which are reported at larger model/data scale; it suggests that whatever benefit DoRA's magnitude/direction decoupling provides may require more scale than tested here to reliably show up. What *is* consistent and reproducible at this scale is DoRA's training cost — roughly 20% slower throughput across all three seeds — making it a real, unrewarded overhead in this specific regime.

## Limitations

- **Only rank 8 tested.** The original question — whether DoRA's advantage depends on rank — remains open here; ranks 16 and 32 were not run, due to compute constraints on free-tier Colab/Kaggle sessions.
- **Single model, single task.** Results are specific to Qwen2.5-1.5B on Spider text-to-SQL and shouldn't be assumed to generalize to larger models, other tasks, or other domains.
- **Complexity buckets are a structural proxy**, not Spider's official difficulty labels, since those weren't available in the HuggingFace dataset version used.
- **No warmup schedule** was used during training due to a version mismatch between `trl` and `transformers` in the training environment (`SFTConfig` rejected `warmup_ratio`); all seeds and both methods were affected identically, so this shouldn't bias the comparison.
- **Seed-42 raw per-example predictions are not included separately** — only the merged comparison CSV was saved for that run, before the per-example logging format was standardized for seeds 43 and 44. The merged seed-42 comparison itself is complete and included.

## Reproducing this

- `lora_vs_dora_spider.ipynb` — training script; set `RANK`, `USE_DORA`, and `SEED` in the config cell and run top to bottom.
- `evaluate.py` — loads a trained adapter and computes execution accuracy standalone, without retraining.
- `compare_seeds.py` — merges per-example LoRA/DoRA predictions for a given seed into the full comparison table (agreement breakdown + complexity buckets).
- `results_log.csv` — aggregate accuracy for every run (all ranks/methods/seeds tested).
- `per_example_comparison_seed{42,43,44}.csv` — full per-example predictions and correctness for each seed.
- `predictions_{lora,dora}_seed{43,44}.csv` — raw per-example predictions before merging, for seeds 43 and 44.
