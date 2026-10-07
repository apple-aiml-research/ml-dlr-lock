#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Text streams over public corpora.

Everything here goes through ``datasets.load_dataset`` so the code runs against
the HuggingFace Hub with no additional infrastructure. Two shapes are provided:

``build_text_stream``
    An infinite iterator of ``(batch_size, seq_len)`` token id tensors, used for
    distillation and perplexity evaluation. Text is concatenated and chopped
    into fixed-length chunks; no padding, no masking.

``build_finetune_data``
    A finite, shuffled corpus of ``(input_ids, labels)`` pairs for the
    fine-tuning attacker, where ``labels`` may mask the prompt with ``-100``.

The distillation corpus in the paper is Nemotron-CC. Any comparable web-text
corpus works; the closer it is to the model's own pretraining distribution, the
better the locked model's quality preservation.
"""

import random

import torch


def _chunk_tokens(tokenizer, texts, seq_len, max_tokens=0):
    """Concatenate ``texts`` and split into ``seq_len``-token chunks."""
    joined = "\n\n".join(t for t in texts if t and t.strip())
    ids = tokenizer(joined, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0]
    n_chunks = len(ids) // seq_len
    if max_tokens:
        n_chunks = min(n_chunks, max_tokens // seq_len)
    if n_chunks == 0:
        raise ValueError(
            f"corpus has {len(ids)} tokens, too few for seq_len={seq_len}")
    return ids[:n_chunks * seq_len].reshape(n_chunks, seq_len)


def _load_wikitext(split, revision=None):
    from datasets import load_dataset
    hf_split = {"train": "train", "validation": "validation",
                "test": "test"}[split]
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=hf_split,
                      revision=revision)
    return ds["text"]


def _load_c4(split, n_docs=200000, revision=None):
    from datasets import load_dataset
    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("allenai/c4", "en", split=hf_split, streaming=True,
                      revision=revision)
    texts = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        texts.append(row["text"])
    return texts


def load_corpus(name, split, revision=None):
    """Load a corpus by name.

    ``revision`` pins the Hugging Face Hub revision -- a branch, tag, or commit
    hash. Left unset, the dataset resolves to its default branch, whose contents
    can change over time. Pin it when a run needs to stay reproducible.
    """
    if name == "wikitext":
        return _load_wikitext(split, revision=revision)
    if name == "c4":
        return _load_c4(split, revision=revision)
    raise ValueError(f"unknown corpus: {name}")


class ChunkStream:
    """Infinite shuffled iterator over fixed-length token chunks."""

    def __init__(self, chunks, batch_size, seed, device=None):
        self.chunks = chunks
        self.batch_size = batch_size
        self.device = device
        self._rng = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(len(chunks), generator=self._rng)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos + self.batch_size > len(self._order):
            self._order = torch.randperm(len(self.chunks),
                                         generator=self._rng)
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        batch = self.chunks[idx]
        return batch.to(self.device) if self.device else batch


def build_text_stream(tokenizer, dataset, split, seq_len, batch_size, seed,
                      device=None, max_tokens=0, revision=None):
    """Infinite ``(batch_size, seq_len)`` stream of token ids.

    ``revision`` pins the dataset's Hub revision; see :func:`load_corpus`.
    """
    texts = load_corpus(dataset, split, revision=revision)
    chunks = _chunk_tokens(tokenizer, texts, seq_len, max_tokens)
    return ChunkStream(chunks, batch_size, seed, device)


# ---------------------------------------------------------------------------
# Fine-tuning (attacker) data
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

ALPACA_TEMPLATE_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def _prompt_masked_examples(tokenizer, pairs, seq_len):
    """Tokenize ``(prompt, response)`` pairs, masking prompt tokens to -100.

    Loss on the response only. This is the standard instruction-tuning
    convention and keeps the measured loss comparable across models whose
    prompt likelihoods differ.
    """
    examples = []
    eos = tokenizer.eos_token or ""
    for prompt, response in pairs:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response + eos,
                          add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[:seq_len]
        labels = ([-100] * len(p_ids) + r_ids)[:seq_len]
        if len(ids) < seq_len:
            pad = seq_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * pad
            labels = labels + [-100] * pad
        examples.append((torch.tensor(ids), torch.tensor(labels)))
    return examples


def _load_alpaca(split, seed, n_val=500, revision=None):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train", revision=revision)
    pairs = []
    for row in ds:
        if row.get("input"):
            prompt = ALPACA_TEMPLATE_INPUT.format(
                instruction=row["instruction"], input=row["input"])
        else:
            prompt = ALPACA_TEMPLATE.format(instruction=row["instruction"])
        pairs.append((prompt, row["output"]))
    # Fixed split seed so train and validation always partition consistently.
    order = list(range(len(pairs)))
    random.Random(0xCAFE).shuffle(order)
    sel = order[n_val:] if split == "train" else order[:n_val]
    return [pairs[i] for i in sel]


def build_finetune_data(tokenizer, dataset, split, seq_len, seed,
                        max_tokens=0, revision=None):
    """Return a list of ``(input_ids, labels)`` tensors.

    ``wikitext`` and ``c4`` are plain language-modeling corpora: labels equal
    inputs. ``alpaca`` is instruction data with the prompt masked out.

    ``revision`` pins the dataset's Hub revision; see :func:`load_corpus`.
    """
    if dataset == "alpaca":
        pairs = _load_alpaca(split, seed, revision=revision)
        examples = _prompt_masked_examples(tokenizer, pairs, seq_len)
    else:
        texts = load_corpus(dataset, split, revision=revision)
        chunks = _chunk_tokens(tokenizer, texts, seq_len, max_tokens)
        examples = [(c, c.clone()) for c in chunks]

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


class BatchIterator:
    """Infinite batched iterator over a finite example list."""

    def __init__(self, examples, batch_size, device, seed):
        self.examples = examples
        self.batch_size = batch_size
        self.device = device
        self._rng = random.Random(seed)
        self._order = list(range(len(examples)))
        self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos + self.batch_size > len(self._order):
            self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        ids = torch.stack([self.examples[i][0] for i in idx])
        labels = torch.stack([self.examples[i][1] for i in idx])
        return ids.to(self.device), labels.to(self.device)
