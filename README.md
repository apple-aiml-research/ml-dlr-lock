# Locking Pretrained Weights via Deep Low-Rank Residual Distillation (DLR-Lock)

<div>
<a href="https://arxiv.org/abs/2605.10777" target="_blank"><img alt="DLR-Lock arXiv" src="https://img.shields.io/badge/arXiv-DLR--Lock-red?logo=arxiv"/></a>
</div>

This software project accompanies the research paper:
[Locking Pretrained Weights via Deep Low-Rank Residual Distillation](https://arxiv.org/abs/2605.10777)
by *Keitaro Sakamoto, Pierre Ablin, Federico Danieli and Marco Cuturi*.

DLR-Lock makes a pretrained model's weights hard to fine-tune. Every SwiGLU MLP is
replaced by a deep low-rank residual network (DLR-Net) of matched parameter count,
distilled to reproduce the original module. Inference behavior is preserved, but
backpropagation becomes far more expensive: activation memory grows with the DLR-Net's
depth, standard fine-tuning recipes destabilize, and the backward pass resists the
optimizations that make the forward pass cheap.

Two-stage distillation turns a pretrained HuggingFace model into a locked one:

1. [dlr_lock/distill_modules.py](dlr_lock/distill_modules.py) — regress each MLP sub-block into a
   DLR-Net independently, without backpropagating through the full model.
2. [dlr_lock/distill_logits.py](dlr_lock/distill_logits.py) — correct the error that compounds across
   depth by matching the full model's output distribution.

Then measure both sides of the trade-off:

3. [dlr_lock/eval_benchmarks.py](dlr_lock/eval_benchmarks.py) — quality preservation,
   what a legitimate user sees.
4. [dlr_lock/eval_finetune.py](dlr_lock/eval_finetune.py) — fine-tuning resistance,
   what an attacker pays.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

To verify the install, run the smoke test. It builds a two-layer random model, runs a
few steps of both stages on a synthetic corpus, fine-tunes both arms, and asserts the
invariants the method relies on. No GPU, no downloads beyond a tokenizer:

```bash
python examples/smoke_test.py
```

## 1. Module-wise regression

Trains one DLR-Net per transformer layer to reproduce the full residual sub-block
`z -> z + FFN(RMSNorm(z))` under a relative mean-squared error. The layers are
independent, so this never backpropagates through the whole model.

`--rank` is the free knob, and the depth follows from it, since the parameter count is
matched to the SwiGLU MLP being replaced. Smaller rank buys a deeper network at the
same parameter count, which locks harder but can cost more forward latency. No particular
depth is required: the ranks below were chosen to hold the depth between 141 and 149
at every scale, so that results stay comparable across model sizes.

`dlr_lock.compute_depth(d, d_ff, r)` returns the depth. To leading order it is
`3 * d_ff / (2 * r)`, in which the width cancels, so the rank doubles with `d` here
only because `d_ff` does.

| Model width `d` | `d_ff` | rank `r` | depth `L` |
|---|---|---|---|
| 1024 | 3072 | 32 | 141 |
| 2048 | 6144 | 64 | 142 |
| 4096 | 12288 | 128 | 143 |
| 5120 | 25600 | 256 | 149 |

```bash
python -m dlr_lock.distill_modules \
    --model Qwen/Qwen3-0.6B \
    --rank 32 \
    --dataset wikitext \
    --output-dir /tmp/dlr-lock/phase1 \
    --steps 500000 \
    --lr 1e-3 \
    --batch-size 2 \
    --seq-len 2048
```

Outputs (in `--output-dir`):

```
config.json     # width, rank, depth, parameter count
layer00.pt      # one DLR-Net state_dict per transformer layer
layer01.pt
...
```

Every stage accepts `--revision` to pin the Hugging Face Hub revision of the
model and dataset to a branch, tag, or commit hash. Left unset they resolve to
the default branch, whose contents can change over time; pin it when a run needs
to stay reproducible.

## 2. Logit distillation

Trains all DLR-Nets jointly against the pretrained teacher, using a top-`k` KL
divergence on the output logits. Attention, embeddings, and the output head stay
frozen. Skipping stage 1 and running this alone is Pareto-dominated at every compute
budget.

```bash
python -m dlr_lock.distill_logits \
    --model Qwen/Qwen3-0.6B \
    --rank 32 \
    --phase1-dir /tmp/dlr-lock/phase1 \
    --output-dir /tmp/dlr-lock/phase2 \
    --steps 100000 \
    --lr 1e-3 \
    --topk 1000 \
    --batch-size 4 \
    --seq-len 2048
```

Outputs (in `--output-dir`): the same layout as stage 1. This directory is the locked
model. Load it with:

```python
import torch
from dlr_lock import build_locked_model, load_dlrnet_weights

model = build_locked_model(
    "Qwen/Qwen3-0.6B", rank=32, device="cuda", torch_dtype=torch.bfloat16)
load_dlrnet_weights(model, "/tmp/dlr-lock/phase2")
```

## 3. Quality preservation

Compares the locked model against the pretrained one on perplexity and LM Evaluation
Harness accuracy. Locking is only useful if legitimate users are unaffected.

```bash
python -m dlr_lock.eval_benchmarks \
    --model Qwen/Qwen3-0.6B \
    --rank 32 \
    --locked-dir /tmp/dlr-lock/phase2 \
    --output /tmp/dlr-lock/quality.json
```

## 4. Fine-tuning resistance

Fine-tunes the pretrained baseline and the locked model under identical conditions,
recording evaluation loss against tokens, FLOPs, and wall clock. Learning rate is the
attacker's main lever, so sweep it and compare each arm at its own best rate.

```bash
for lr in 3e-6 1e-5 3e-5 1e-4; do
  python -m dlr_lock.eval_finetune --model Qwen/Qwen3-0.6B \
      --dataset wikitext --lr $lr \
      --output /tmp/dlr-lock/baseline_$lr.json
  python -m dlr_lock.eval_finetune --model Qwen/Qwen3-0.6B \
      --dataset wikitext --lr $lr \
      --rank 32 --locked-dir /tmp/dlr-lock/phase2 \
      --output /tmp/dlr-lock/locked_$lr.json
done
```

Each run writes a JSON history with `step`, `tokens`, `cum_flops`, `wall_clock`,
`eval_loss`, and `peak_mem_gb`. Gradient checkpointing is on by default in both arms,
so the comparison is against an attacker who has already applied the obvious memory
mitigation; pass `--no-grad-checkpointing` to measure the unmitigated footprint.

## Inference kernels

Replacing an MLP with a depth-`L` DLR-Net leaves total inference FLOPs unchanged, but
the chain of small matrix multiplications costs wall-clock time in eager execution.
[kernels/](kernels/) contains forward-pass implementations that recover most of it by
keeping the hidden state on chip across all layers. Which one is fastest depends on
model scale and batch size; [kernels/registry.py](kernels/registry.py) maps a
`(model, batch size)` pair to an implementation, and
[kernels/bench/bench_paper.py](kernels/bench/bench_paper.py) measures them against a
`torch.compile`d SwiGLU baseline under locked clocks. The CUDA kernels target Hopper
(`sm_90a`) and build on first use.

## Datasets

Distillation and evaluation load public corpora through `datasets.load_dataset`:
`wikitext`, `allenai/c4`, and `tatsu-lab/alpaca`. The paper distills on Nemotron-CC;
any comparable web-text corpus works, and the closer the locking corpus is to the
model's own pretraining distribution, the better quality is preserved.
[dlr_lock/data.py](dlr_lock/data.py) is where to add one.

## Citation

If you find our work useful, please consider citing us as:

```bibtex
@article{sakamoto2026locking,
  title={Locking Pretrained Weights via Deep Low-Rank Residual Distillation},
  author={Sakamoto, Keitaro and Ablin, Pierre and Danieli, Federico and Cuturi, Marco},
  journal={arXiv preprint arXiv:2605.10777},
  year={2026}
}
```

## License

Released under the Apple Sample Code License. See [LICENSE](LICENSE).
