# LoRA Adapter (rank 8) — Qwen2.5-1.5B, Spider Text-to-SQL

Base model: Qwen/Qwen2.5-1.5B (4-bit NF4)
PEFT method: LoRA, rank 8, alpha 16, dropout 0.05
Target modules: q_proj, k_proj, v_proj, o_proj

## Training

- Dataset: Spider (text-to-SQL), 3000 training examples
- 3 epochs, batch size 4, grad accumulation 4, lr 2e-4, cosine schedule
- Single seed (42), single T4 GPU (Colab free tier)

## Evaluation

Execution accuracy on 300 examples from the Spider dev set: **0.4233**

Full analysis, per-example comparison against the DoRA adapter, and discussion
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
writeup for the full picture).

## Method

LoRA freezes the pretrained weight matrix and adds a trainable low-rank
update alongside it, as introduced in:

> Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li,
> Shean Wang, Lu Wang, Weizhu Chen. **LoRA: Low-Rank Adaptation of Large
> Language Models.** arXiv:2106.09685, 2021.
> https://arxiv.org/abs/2106.09685

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

Implementation used: Hugging Face [PEFT](https://github.com/huggingface/peft)
(`LoraConfig`).