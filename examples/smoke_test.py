#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Smoke test for the whole pipeline on a tiny randomly-initialized model.

Builds a two-layer transformer, runs a handful of Phase-1 and Phase-2 steps on a
synthetic corpus, then fine-tunes both arms. It checks that the code runs and
that the invariants hold -- it is far too small and too short to say anything
about locking strength.

    python examples/smoke_test.py

Runs on CPU in a couple of minutes; no downloads, no GPU.
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dlr_lock.data as data_module
from dlr_lock.distill_logits import topk_kl_loss
from dlr_lock.model import (DLRNet, build_locked_model, compute_depth,
                            dlrnet_param_count, load_dlrnet_weights,
                            replace_layer_mlp, save_dlrnet_weights)

_WORDS = ("model locking gradient descent transformer residual low rank "
          "distillation attention perplexity kernel batch tensor norm").split()


def install_synthetic_corpus():
    """Replace the corpus loader so the test needs no network access."""
    def _fake(name, split, revision=None):
        rng = random.Random(0 if split == "train" else 1)
        n = 4000 if split == "train" else 800
        return [" ".join(rng.choice(_WORDS) for _ in range(30))
                for _ in range(n)]

    data_module.load_corpus = _fake


def build_tiny_model(path, tokenizer_source):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=64, intermediate_size=192, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=len(tokenizer), max_position_embeddings=512,
        tie_word_embeddings=True)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def check_invariants():
    """The properties the distillation relies on, checked directly."""
    d, r, depth = 64, 8, 12

    net = DLRNet(d, r, depth, alpha_init=0.0)
    n = sum(p.numel() for p in net.parameters())
    assert n == dlrnet_param_count(d, r, depth), "parameter count formula"

    # alpha = 0 makes the DLR-Net an exact identity.
    x = torch.randn(4, 7, d)
    assert torch.equal(net(x), x), "identity at alpha=0"

    # Activation checkpointing must not change values or gradients.
    net = DLRNet(d, r, depth, alpha_init=0.1)
    xg = torch.randn(3, 5, d, requires_grad=True)
    net.use_checkpoint = True
    ya = net(xg)
    ga = torch.autograd.grad(ya.sum(), xg)[0]
    net.use_checkpoint = False
    yb = net(xg)
    gb = torch.autograd.grad(yb.sum(), xg)[0]
    assert torch.allclose(ya, yb, atol=1e-6), "checkpoint changes forward"
    assert torch.allclose(ga, gb, atol=1e-6), "checkpoint changes gradients"

    # Depths from the paper, at a budget that stays under the SwiGLU count.
    for dd, d_ff, rr, expected in ((1024, 3072, 32, 141), (2048, 6144, 64, 142),
                                   (4096, 12288, 128, 143),
                                   (5120, 25600, 256, 149)):
        got = compute_depth(dd, d_ff, rr)
        assert got == expected, f"depth for d={dd}: {got} != {expected}"
        assert dlrnet_param_count(dd, rr, got) <= 3 * dd * d_ff, "over budget"

    print("invariants: parameter count, identity init, checkpoint "
          "equivalence, depth table -- all hold")


def check_surgery(model_path):
    """A locked layer must compute exactly DLRNet(h)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path).eval()
    d = model.config.hidden_size
    depth = compute_depth(d, model.config.intermediate_size, 8)
    net = replace_layer_mlp(model, 0, d, 8, depth, alpha_init=0.05)
    layer = model.model.layers[0]

    seen = {}
    layer.mlp.register_forward_pre_hook(
        lambda m, a: seen.__setitem__("h", a[0].detach().clone()))
    layer.register_forward_hook(
        lambda m, a, o: seen.__setitem__(
            "y", (o[0] if isinstance(o, tuple) else o).detach().clone()))

    with torch.no_grad():
        model(input_ids=torch.randint(0, 128, (2, 16)))
        expected = net(seen["h"])
    assert torch.allclose(seen["y"], expected, atol=1e-5), \
        "locked block does not compute DLRNet(h)"
    print("surgery: locked block computes DLRNet(h) exactly "
          "(norm and residual absorbed)")


def check_kl_objective():
    """Phase 2's loss must behave like a KL divergence over the top-k subset."""
    torch.manual_seed(0)
    teacher = torch.randn(2, 16, 128)
    student = torch.randn(2, 16, 128)

    # Zero when the two distributions coincide.
    assert abs(topk_kl_loss(teacher.clone(), teacher, 1000, 1.0).item()) < 1e-6, \
        "KL of a distribution against itself is not zero"

    # topk=0 means the full vocabulary, so it must equal a plain KL.
    manual = F.kl_div(F.log_softmax(student.float(), dim=-1),
                      F.log_softmax(teacher.float(), dim=-1),
                      reduction="batchmean", log_target=True)
    assert torch.allclose(topk_kl_loss(student, teacher, 0, 1.0), manual,
                          atol=1e-6), "full-vocabulary KL disagrees with F.kl_div"

    # A k larger than the vocabulary must clamp rather than raise, and then
    # agree with the full-vocabulary result.
    assert torch.allclose(topk_kl_loss(student, teacher, 10_000, 1.0), manual,
                          atol=1e-5), "k > vocab_size is not clamped to vocab_size"

    # A restricted k is still a positive, finite divergence.
    partial = float(topk_kl_loss(student, teacher, 5, 1.0))
    assert 0 < partial < float("inf"), "top-5 KL is not positive and finite"

    print("kl objective: zero on self, matches F.kl_div at k=0, clamps k > vocab")


def check_checkpoint_roundtrip(model_path):
    """Saving and reloading DLR-Net weights must reproduce logits exactly.

    A regression in the checkpoint format would silently corrupt every locked
    model, so compare logits rather than just tensor equality.
    """
    import tempfile

    torch.manual_seed(0)
    model = build_locked_model(model_path, rank=8, torch_dtype=torch.float32,
                              verbose=False)
    # Move off the identity initialization so the comparison is meaningful.
    for p in model.model.layers[0].mlp.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)
    model.eval()

    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()

    with tempfile.TemporaryDirectory() as tmp:
        n = save_dlrnet_weights(model, tmp)
        reloaded = build_locked_model(model_path, rank=8,
                                      torch_dtype=torch.float32, verbose=False)
        load_dlrnet_weights(reloaded, tmp)
        reloaded.eval()
        with torch.no_grad():
            after = reloaded(input_ids=ids).logits

    assert torch.allclose(before, after, atol=1e-6), \
        "save/load round-trip changes the model's output"
    print(f"checkpoint: save/load round-trip over {n} layers reproduces "
          "logits exactly")


def check_prompt_masking(tokenizer_source):
    """Instruction data must train on the response only, not the prompt.

    If the prompt stopped being masked, fine-tuning loss would shift for both
    arms and could be mistaken for a locking effect.
    """
    from dlr_lock.data import _prompt_masked_examples
    from dlr_lock.loading import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_source)
    prompt, response = "Instruction: say something.\nResponse:\n", "Something."
    seq_len = 64
    (ids, labels), = _prompt_masked_examples(tokenizer, [(prompt, response)],
                                             seq_len)

    assert ids.shape == labels.shape == (seq_len,), "example is not seq_len long"

    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert (labels[:n_prompt] == -100).all(), "prompt tokens are not masked"

    n_response = len(tokenizer(response + (tokenizer.eos_token or ""),
                               add_special_tokens=False)["input_ids"])
    supervised = labels[n_prompt:n_prompt + n_response]
    assert (supervised != -100).all(), "response tokens are masked"
    # Supervised positions must carry the true next tokens, not placeholders.
    assert torch.equal(supervised, ids[n_prompt:n_prompt + n_response]), \
        "supervised labels do not match the input ids"
    # Everything past the response is padding and must be ignored.
    assert (labels[n_prompt + n_response:] == -100).all(), \
        "padding is not masked"

    print(f"prompt masking: {n_prompt} prompt + {seq_len - n_prompt - n_response}"
          f" padding positions ignored, {n_response} response positions supervised")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                        help="Any tokenizer; only its vocabulary size is used.")
    parser.add_argument("--keep", default="",
                        help="Directory to keep artifacts in (default: temp).")
    args = parser.parse_args()

    install_synthetic_corpus()
    check_invariants()
    check_kl_objective()
    check_prompt_masking(args.tokenizer)

    workdir = args.keep or tempfile.mkdtemp(prefix="dlr_lock_smoke_")
    os.makedirs(workdir, exist_ok=True)
    model_path = build_tiny_model(os.path.join(workdir, "model"),
                                  args.tokenizer)
    print(f"built a 2-layer test model in {model_path}")
    check_surgery(model_path)
    check_checkpoint_roundtrip(model_path)

    common = ["--model", model_path, "--rank", "8",
              "--seq-len", "64", "--batch-size", "2"]

    print("\n--- Phase 1: module-wise regression ---")
    from dlr_lock.distill_modules import main as distill_modules_main
    phase1 = os.path.join(workdir, "phase1")
    distill_modules_main(common + [
        "--output-dir", phase1, "--steps", "24", "--lr", "1e-2",
        "--student-batch-tokens", "64", "--eval-tokens", "256",
        "--teacher-device", "cpu", "--student-device", "cpu",
        "--log-interval", "8", "--eval-interval", "16", "--save-interval", "0"])

    print("\n--- Phase 2: logit distillation ---")
    from dlr_lock.distill_logits import main as distill_logits_main
    phase2 = os.path.join(workdir, "phase2")
    distill_logits_main(common + [
        "--phase1-dir", phase1, "--output-dir", phase2,
        "--steps", "20", "--lr", "1e-2", "--topk", "50",
        "--teacher-device", "cpu", "--student-device", "cpu",
        "--log-interval", "5", "--eval-interval", "10", "--eval-steps", "3",
        "--save-interval", "0"])

    print("\n--- Quality preservation ---")
    # Note: eval_benchmarks takes --eval-batch-size, not --batch-size, so it
    # does not share `common`. --tasks '' skips the LM Evaluation Harness, which
    # is an optional dependency and far too slow for a smoke test; the
    # perplexity comparison still exercises the whole load-and-compare path.
    from dlr_lock.eval_benchmarks import main as eval_benchmarks_main
    quality_json = os.path.join(workdir, "quality.json")
    original, locked = eval_benchmarks_main([
        "--model", model_path, "--rank", "8",
        "--locked-dir", phase2,
        "--dataset", "wikitext", "--seq-len", "64",
        "--eval-batch-size", "2", "--eval-steps", "3",
        "--tasks", "",
        "--device", "cpu",
        "--output", quality_json,
    ])

    ppl_key = "wikitext_ppl"
    for name, results in (("pretrained", original), ("locked", locked)):
        assert ppl_key in results, f"{name} results missing {ppl_key}"
        value = results[ppl_key]
        assert math.isfinite(value) and value > 0, \
            f"{name} perplexity is not finite and positive: {value}"

    # The comparison is only meaningful if it was actually written out.
    with open(quality_json) as f:
        saved = json.load(f)
    assert saved["original"][ppl_key] == original[ppl_key], \
        "saved original perplexity does not match the returned value"
    assert saved["locked"][ppl_key] == locked[ppl_key], \
        "saved locked perplexity does not match the returned value"
    print(f"quality preservation: pretrained {original[ppl_key]:.1f} vs "
          f"locked {locked[ppl_key]:.1f} perplexity, written to quality.json")

    print("\n--- Fine-tuning both arms ---")
    from dlr_lock.eval_finetune import main as ft_main
    histories = {}
    for arm, extra in (("baseline", []), ("locked", ["--locked-dir", phase2])):
        histories[arm] = ft_main(common + extra + [
            "--dataset", "wikitext", "--lr", "1e-4", "--steps", "20",
            "--device", "cpu", "--eval-interval", "10", "--eval-batches", "3",
            "--log-interval", "10", "--early-eval-steps", "5"])

    # Both arms must be parameter- and FLOP-matched: that is the premise of the
    # whole comparison, so assert it rather than eyeballing the logs.
    b, l = histories["baseline"], histories["locked"]
    ratio = l["n_params"] / b["n_params"]
    assert 0.9 < ratio < 1.1, f"arms not parameter-matched ({ratio:.3f}x)"
    print(f"\nboth arms trained; locked/baseline parameters = {ratio:.3f}x")

    if not args.keep:
        print(f"(artifacts in {workdir}; pass --keep to choose the location)")
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
