# LoRA vs. DoRA at Small Scale: A Rank-8 Case Study on Text-to-SQL

**Finding in brief.** At rank 8, on Qwen2.5-1.5B fine-tuned with 3,000 Spider examples, there was no consistent accuracy difference between LoRA and DoRA across three seeds. The sign of the gap changed between seeds (range: −0.3 to +4.0 points, LoRA minus DoRA). DoRA took longer to train. This is a small, single-model, single-task study, and it cannot resolve differences of a few points (see Limitations).

## Motivation

LoRA freezes a model's pretrained weights and injects a trainable low-rank update (W = W₀ + BA), adapting far fewer parameters than full fine-tuning. DoRA (Liu et al., 2024) modifies this by decomposing the pretrained weight into a magnitude component and a direction component, applying the LoRA-style low-rank update only to direction, while letting magnitude adapt freely and independently. The motivation, per the original paper, is that this more closely mirrors how full fine-tuning updates weights, and DoRA is reported to outperform LoRA on several benchmarks at typical training scales.

Most published DoRA results are on larger models and larger training budgets. This project asks a narrower, more practical question: does DoRA's advantage over LoRA show up at small scale (a 1.5B-parameter model, a few thousand fine-tuning examples, a single consumer GPU)? That is the regime many individual practitioners and students work in, and it is not the regime the original paper tested.

## Setup

- **Base model:** Qwen2.5-1.5B, loaded in 4-bit (NF4) via bitsandbytes
- **Task:** Text-to-SQL on the Spider dataset (natural-language question + database schema → SQL query)
- **Training data:** 3,000 examples from Spider's training split
- **Adapters compared:** LoRA and DoRA, both rank 8, `lora_alpha=16`, targeting `q_proj`/`k_proj`/`v_proj`/`o_proj`. Learning rate, batch size and epochs were identical. The only difference between runs is the `use_dora` flag in PEFT's `LoraConfig`.
- **Evaluation:** Execution accuracy on 300 examples from Spider's dev set. A predicted query counts as correct if running it against the actual database returns the same result set as the gold query, regardless of surface-level SQL differences.
- **Seeds:** three seeds per method (42, 43, 44)

## Results (seed 42)

| Rank | Method | Execution Accuracy (n=300) |
|------|--------|----------------------------|
| 8 | LoRA | 0.4233 |
| 8 | DoRA | 0.4133 |

A gap of about one point, with LoRA slightly ahead. A single-seed, 300-example comparison can produce a gap this size from noise alone; the additional seeds below are consistent with that.

### Per-example agreement (seed 42)

| | Count | % of 300 |
|---|---|---|
| Both correct | 110 | 36.7% |
| Both wrong | 159 | 53.0% |
| LoRA correct, DoRA wrong | 17 | 5.7% |
| DoRA correct, LoRA wrong | 14 | 4.7% |

LoRA and DoRA agree (both right or both wrong) on 89.7% of examples. The aggregate gap comes down to a net difference of 3 examples out of 300. An exact McNemar test on the 17 vs 14 discordant pairs gives p = 0.72 (seed 42 only; this test was not run for seeds 43 and 44).

### Breakdown by query complexity (seed 42)

Spider's HuggingFace mirror (`xlangai/spider`) does not carry the dataset's original human-assigned difficulty labels, so difficulty here is a structural proxy computed from each gold query: a count of joins, subqueries, aggregation functions, GROUP BY/HAVING clauses, boolean conditions and set operations, bucketed into simple / moderate / complex.

| Complexity | N | LoRA acc | DoRA acc |
|------------|---|----------|----------|
| Simple | 49 | 0.429 | 0.388 |
| Moderate | 139 | 0.540 | 0.554 |
| Complex | 112 | 0.277 | 0.250 |

If DoRA's magnitude/direction decoupling gave it an edge on harder queries, its advantage should grow with complexity. Instead LoRA is slightly ahead in the complex bucket, and the gaps at every level are small relative to the per-example disagreement.

## Seed robustness check

To test whether the seed-42 gap reflected a method difference or ordinary training noise, both adapters were retrained and re-evaluated from scratch with two additional seeds (43, 44). Hyperparameters and the evaluation set were unchanged; only the seed changed.

| | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| LoRA | 0.4233 | 0.4567 | 0.4233 |
| DoRA | 0.4133 | 0.4167 | 0.4267 |
| Gap (LoRA − DoRA) | +1.0 pts | +4.0 pts | −0.3 pts |

Gaps ranged from −0.3 to +4.0 points and changed sign across seeds: LoRA was ahead in two seeds and DoRA in one. LoRA's own accuracy moved by 3.3 points between seeds (0.4233 → 0.4567 → 0.4233), comparable in size to the gaps between methods.

**Conclusion of the seed check:** across three seeds there is no evidence of a reliable accuracy advantage for either method at rank 8 on this model, task and scale. With n=300 and three seeds, differences of a few points cannot be ruled out. Any single-seed LoRA-vs-DoRA comparison at this scale should be treated as noisy.

Per-example agreement was similar in seeds 43 and 44 (roughly 88–90%), and no seed showed a consistent complexity-dependent pattern favoring either method.

## Training cost

DoRA took longer to train than LoRA in the recorded runs.

| | LoRA | DoRA |
|---|---|---|
| Training time (seed 44) | ~3h 04m | ~3h 22m |
| Throughput (rounded) | ~0.05 it/s | ~0.04 it/s |

In seed 44, DoRA took about 10% longer (3h 04m vs 3h 22m). Across runs, DoRA was roughly 18–30 minutes slower per run, which is about 10–16% longer. The iteration rates above are rounded to one digit and overstate the difference if read literally.

This is consistent with the method's design: DoRA computes a column-wise norm of the updated weight matrix at every forward pass, and its backward pass is more involved than LoRA's plain low-rank update (the DoRA paper, Sections 4.2–4.3, discusses this overhead and a mitigation the authors apply; this project used the default PEFT `use_dora` implementation without that modification). The cost was **not profiled** here. Because the base model is loaded in 4-bit NF4, dequantization overhead may also contribute; this was not tested.

Even without a measurable accuracy benefit at this scale, DoRA carried a repeatable compute cost in these runs, which is worth weighing against an unproven benefit.

## What the disagreements look like

A few representative cases from seed 42:

**DoRA correct, LoRA wrong.** LoRA used a plausible but wrong column name:

> Q: Find the number of pets whose weight is heavier than 10.
> Gold: `SELECT count(*) FROM pets WHERE weight > 10`
> LoRA: `SELECT count(*) FROM pets WHERE pet_weight > 10`
> DoRA: `SELECT count(*) FROM pets WHERE weight > 10` ✓

**LoRA correct, DoRA wrong.** DoRA substituted the wrong set operation:

> Q: Show countries where a singer above age 40 and a singer below 30 are from.
> Gold: `... WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30`
> LoRA: `... INTERSECT SELECT Country FROM singer WHERE Age < 30` ✓
> DoRA: `... UNION SELECT Country FROM singer WHERE Age < 30`

Both methods make similar categories of mistakes (unnecessary joins on single-table questions, wrong column names, wrong SQL keywords or operators), with no consistent pattern in which method makes which kind.

## Conclusion

At rank 8, on a 1.5B model, with 3,000 fine-tuning examples on Spider text-to-SQL, across three seeds, DoRA showed no consistent accuracy advantage over LoRA, and the direction of the gap was not reproducible (LoRA ahead in two seeds, DoRA in one). This does not contradict DoRA's published results, which are reported at larger model and data scale. It suggests that any benefit of DoRA's magnitude/direction decoupling may need more scale than was tested here to show up reliably. DoRA's training cost was the more consistent observation, roughly 10–16% longer training time in these runs.

## Limitations

- **Small evaluation set.** With n=300 and accuracy near 0.42, the 95% interval on a single accuracy is about ±5.6 points, so this study cannot resolve differences of a few points. Only 300 of Spider's dev examples were used.
- **Little formal testing.** A McNemar test was computed for seed 42 only. Three seeds is a small sample for estimating seed variance.
- **Only rank 8 tested.** Whether DoRA's advantage depends on rank remains open; ranks 16 and 32 were not run, due to compute constraints on free-tier Colab/Kaggle sessions.
- **Single model, single task, 4-bit base.** Results are specific to Qwen2.5-1.5B (NF4-quantized) on Spider text-to-SQL and should not be assumed to hold for larger models, other tasks, or full-precision training.
- **Complexity buckets are a proxy,** not Spider's official difficulty labels, which were not available in the HuggingFace version used.
- **No warmup schedule.** A version mismatch between `trl` and `transformers` in the training environment (`SFTConfig` rejected `warmup_ratio`) meant no warmup was used. All seeds and both methods were affected identically, so this should not bias the comparison, but it may differ from standard practice.
- **Training cost not profiled.** The explanation for DoRA's slower training is an expectation from the method, not a measurement.
- **Seed-42 raw per-example predictions** are not included separately. Only the merged comparison CSV was saved for that run, before per-example logging was standardized for seeds 43 and 44.

## Reproducing this

- `lora_vs_dora_spider.ipynb`: training script. Set `RANK`, `USE_DORA` and `SEED` in the config cell and run top to bottom.
- `evaluate.py`: loads a trained adapter and computes execution accuracy without retraining.
- `compare_seeds.py`: merges per-example LoRA/DoRA predictions for a given seed into the full comparison table (agreement breakdown + complexity buckets).
- `results_log.csv`: aggregate accuracy for every run (all ranks, methods and seeds tested).
- `per_example_comparison_seed{42,43,44}.csv`: full per-example predictions and correctness for each seed.
- `predictions_{lora,dora}_seed{43,44}.csv`: raw per-example predictions before merging, for seeds 43 and 44.

## References

- Liu et al., 2024. DoRA: Weight-Decomposed Low-Rank Adaptation.
- Hu et al., 2021. LoRA: Low-Rank Adaptation of Large Language Models.
- Yu et al., 2018. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.
