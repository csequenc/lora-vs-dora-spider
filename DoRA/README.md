# DoRA Adapter (rank 8) — Qwen2.5-1.5B, Spider Text-to-SQL

Base model: Qwen/Qwen2.5-1.5B (4-bit NF4)
PEFT method: DoRA, rank 8, alpha 16, dropout 0.05
Target modules: q_proj, k_proj, v_proj, o_proj

## Training

- Dataset: Spider (text-to-SQL), 3000 training examples
- 3 epochs, batch size 4, grad accumulation 4, lr 2e-4, cosine schedule
- Single seed (42), single T4 GPU (Colab free tier)
- Identical hyperparameters to the LoRA adapter in this study — the only
  difference is `use_dora=True` in the PEFT `LoraConfig`

## Evaluation

Execution accuracy on 300 examples from the Spider dev set: **0.4133**

Full analysis, per-example comparison against the LoRA adapter, and discussion
of limitations are in [`../../experiment/README.md`](../../experiment/README.md).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
model = PeftModel.from_pretrained(base, "path/to/this/folder")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/folder")
```

This adapter is part of a small comparative study, not a general-purpose
text-to-SQL model — accuracy at this scale/rank is modest (see experiment
writeup for the full picture). Note: this study finds **no measurable
advantage of DoRA over LoRA at rank 8** in this setup (89.7% per-example
agreement between the two adapters) — see the experiment writeup for
details.

## Method

DoRA decomposes the pretrained weight into magnitude and direction
components, applies a LoRA-style low-rank update to direction only, and
lets magnitude adapt as a separate, freely trainable vector — as
introduced in:

> Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank
> Wang, Kwang-Ting Cheng, Min-Hung Chen. **DoRA: Weight-Decomposed Low-Rank
> Adaptation.** arXiv:2402.09353, 2024. Presented at ICML 2024 (Oral).
> https://arxiv.org/abs/2402.09353

```bibtex
@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024}
}
```

Implementation used: Hugging Face [PEFT](https://github.com/huggingface/peft)
(`LoraConfig(use_dora=True)`).