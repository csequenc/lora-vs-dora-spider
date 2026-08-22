# LoRA vs. DoRA at Small Scale: A Rank-8 Case Study on Text-to-SQL

## Motivation

LoRA freezes a model's pretrained weights and injects a trainable low-rank update (`W = W₀ + BA`), adapting far fewer parameters than full fine-tuning. DoRA (Liu et al., 2024) modifies this by decomposing the pretrained weight into a magnitude component and a direction component, applying the LoRA-style low-rank update only to direction, while letting magnitude adapt freely and independently. The motivation, per the original paper, is that this more closely mirrors how full fine-tuning updates weights — and DoRA is reported to outperform LoRA on several benchmarks at typical training scales.

Most published DoRA results are on larger models and larger training budgets. This project asks a narrower, more practical question: **does DoRA's advantage over LoRA hold up at small scale** — a 1.5B-parameter model, a few thousand fine-tuning examples, a single consumer GPU? That's the regime most individual practitioners and students actually operate in, and it's not the regime the original paper was testing.

## Setup

- **Base model:** Qwen2.5-1.5B, loaded in 4-bit (NF4) via `bitsandbytes`
- **Task:** Text-to-SQL on the Spider dataset — natural language question + database schema → SQL query
- **Training data:** 3,000 examples from Spider's training split
- **Adapters compared:** LoRA and DoRA, both rank 8, `lora_alpha=16`, targeting `q_proj/k_proj/v_proj/o_proj`, identical hyperparameters otherwise (learning rate, batch size, epochs) — the only difference between the two runs is the `use_dora` flag in PEFT's `LoraConfig`
- **Evaluation:** Execution accuracy on 300 examples from Spider's dev set — a predicted query counts as correct if running it against the actual database returns the same result set as the gold query, regardless of surface-level SQL differences (this is the standard metric in text-to-SQL literature, e.g. the Spider benchmark itself)

- ## Seed robustness check

To test whether the seed-42 accuracy gap reflected a real method difference
or ordinary training noise, both adapters were retrained and re-evaluated
across two additional seeds (43, 44), same hyperparameters otherwise.

| | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| LoRA | 0.4233 | 0.4567 | 0.4233 |
| DoRA | 0.4133 | 0.4167 | 0.4267 |
| Gap (LoRA − DoRA) | +1.0 pts | +4.0 pts | −0.3 pts |

The gap not only varies in size across seeds but **flips sign** in seed 44 —
DoRA is marginally ahead. LoRA's own accuracy alone swings by 3.3 points
across seeds (0.4233 to 0.4567), which exceeds every observed LoRA-vs-DoRA
gap. This confirms the original conclusion with stronger evidence than a
single seed could: at rank 8 on this model/task/scale, there is no
consistent, directional advantage for either method — the aggregate
accuracy differences reported in isolated single-seed comparisons (including
this project's own seed-42 run) are training noise, not a real effect.

## Training cost

DoRA was consistently slower to train than LoRA across all three seeds —
roughly 18-30 minutes longer per run, with throughput around 0.04 it/s for
DoRA versus 0.05 it/s for LoRA (a ~20% slower iteration rate). This matches
expectations from the method: DoRA computes a column-wise norm of the
updated weight at every forward pass and has a more involved backward pass
than LoRA's plain low-rank update (see Section 4.2-4.3 of the DoRA paper).
This cost is consistent and structural, unlike the accuracy comparison above
— even where DoRA shows no measurable accuracy benefit at this scale, it
carries a measurable, repeatable compute overhead.

## Results

| Rank | Method | Execution Accuracy (n=300) |
|---|---|---|
| 8 | LoRA | 0.4233 |
| 8 | DoRA | 0.4133 |

A roughly 1-point gap, with LoRA slightly ahead. On its own, this number doesn't say much — a single-seed, 300-example comparison can easily produce a gap this size from noise alone. The more informative view is what's underneath the aggregate number.

### Per-example agreement

| | Count | % of 300 |
|---|---|---|
| Both correct | 110 | 36.7% |
| Both wrong | 159 | 53.0% |
| LoRA correct, DoRA wrong | 17 | 5.7% |
| DoRA correct, LoRA wrong | 14 | 4.7% |

LoRA and DoRA agree — both right or both wrong — on **89.7%** of examples. The entire aggregate gap comes down to a net difference of 3 examples out of 300. That's a much weaker basis for a claim than the headline accuracy numbers suggest: this is not two methods with different strengths that happen to average out close together, it's two methods behaving almost identically on a case-by-case basis, with the small remaining disagreement split nearly evenly.

### Breakdown by query complexity

Spider's HuggingFace mirror (`xlangai/spider`) doesn't carry the dataset's original human-assigned difficulty labels, so difficulty here is a simple structural proxy computed from each gold query: a count of joins, subqueries, aggregation functions, `GROUP BY`/`HAVING` clauses, boolean conditions, and set operations, bucketed into simple / moderate / complex.

| Complexity | N | LoRA acc | DoRA acc |
|---|---|---|---|
| Simple | 49 | 0.429 | 0.388 |
| Moderate | 139 | 0.540 | 0.554 |
| Complex | 112 | 0.277 | 0.250 |

If DoRA's magnitude/direction decoupling gave it a real edge on harder queries — a plausible hypothesis, since more structurally complex queries plausibly demand a more expressive update than a plain low-rank delta — we'd expect DoRA's advantage to grow with complexity. Instead, LoRA is (slightly) ahead in the complex bucket, and the gaps at every level are within the noise band suggested by the per-example agreement numbers above. No pattern emerges.

### What the disagreements actually look like

A few representative cases from the 31 examples where the two methods disagreed:

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

**DoRA correct, LoRA wrong** — LoRA over-complicated a query that didn't need a join:
> Q: *What are all distinct countries where singers above age 20 are from?*
> Gold: `SELECT DISTINCT country FROM singer WHERE age > 20`
> LoRA: reached for a 3-table join with a nested subquery
> DoRA: `SELECT DISTINCT T1.Country FROM singer AS T1 ... WHERE T1.Age > 20` ✓ (closer, though still an unnecessary join)

Both methods make similar categories of mistakes overall — unnecessary joins on single-table questions, wrong column names, and wrong SQL keywords/operators — and neither method shows a clean, consistent edge in any one category across the disagreement set.

## Conclusion

At rank 8, on a 1.5B model, with 3,000 fine-tuning examples on Spider text-to-SQL: **DoRA shows no measurable advantage over LoRA.** Aggregate accuracy is statistically indistinguishable given the small disagreement pool (31/300 examples, split 17-14), there's no complexity-dependent pattern favoring either method, and the qualitative error types are similar across both. This doesn't contradict DoRA's published results — it suggests that whatever benefit DoRA's magnitude/direction decoupling provides may require a larger model, more training data, or a higher rank than tested here to become visible. This experiment doesn't distinguish between those explanations; it only establishes that at *this* rank and scale, the advantage doesn't show up.

## Limitations

- **Single seed per method.** No repeated runs, so the 1-point aggregate gap and the 3-example net disagreement can't be distinguished from ordinary training-run variance. A proper significance claim would need 3+ seeds per config.
- **Single rank tested (8).** The original research question — whether DoRA's advantage depends on rank — is only partially answered here. Ranks 16 and 32 were planned but not run in this phase of the project, due to Colab free-tier compute constraints; extending the rank sweep is the natural next step.
- **Single model, single task.** Results are specific to Qwen2.5-1.5B on Spider text-to-SQL and shouldn't be assumed to generalize to larger models, other tasks, or other domains.
- **Complexity buckets are a structural proxy**, not Spider's official difficulty labels, since those weren't available in the HuggingFace dataset version used. The proxy is transparent and reproducible but not directly comparable to complexity breakdowns reported in other Spider-based papers.
- **No warmup schedule** was used during training due to a version mismatch between `trl` and `transformers` in the Colab environment (`SFTConfig` rejected `warmup_ratio`); both runs were affected identically, so this shouldn't bias the LoRA-vs-DoRA comparison, but it's a deviation from typical training setups.

## Reproducing this

Code and the full per-example prediction log (`per_example_comparison.csv`, all 300 examples with both models' predictions, correctness, and complexity bucket) are available in the project repository. The training notebook and standalone evaluation script used to produce these results are included as well.
