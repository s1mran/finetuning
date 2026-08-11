"""
cpt_fin_then_sft.py
===================

Two-stage domain adaptation of a small base LLM:

    base model  ->  Stage 1: CPT  ->  merge  ->  Stage 2: SFT  ->  merge  ->  final

Stage 1 (CPT / continued pre-training)
    Plain next-token prediction on raw financial prose pulled from SEC 10-K
    filings. Loss on *every* token. LoRA rank 32 targeting all linear layers
    **plus** embed_tokens and lm_head, because the model is learning new
    domain vocabulary ("EBITDA", "diluted EPS", "subordinated debentures").
    A slice of general text is mixed in to fight catastrophic forgetting.

Stage 2 (SFT / supervised fine-tuning)
    Alpaca-formatted (instruction, input, output) pairs. Loss masked to the
    response span only. LoRA rank 16 on the linear layers only -- the
    vocabulary is already fine, only behaviour needs to change.

This is the ordering that actually works: CPT teaches the model to *sound*
like the domain, SFT then teaches it to *answer*. Running SFT last means the
instruction-following behaviour is the freshest thing in the weights.

Requirements
------------
    pip install unsloth trl peft transformers datasets accelerate bitsandbytes

Needs a CUDA GPU (a free Colab T4 is plenty for a 135M model). Unsloth does
not support Apple MPS.

Usage
-----
    python cpt_fin_then_sft.py                     # full run, ~15 min on a T4
    python cpt_fin_then_sft.py --smoke             # 30 steps per stage, sanity check
    python cpt_fin_then_sft.py --cpt-steps 400 --sft-steps 400

    # publish to huggingface.co/sidhusarkar (needs `huggingface-cli login` or
    # HF_TOKEN with write scope)
    python cpt_fin_then_sft.py --push-to-hub
    python cpt_fin_then_sft.py --push-to-hub other-user/other-repo --push-adapters
"""

from __future__ import annotations

# Unsloth must be imported before transformers/trl so its patches land.
from unsloth import FastLanguageModel  # noqa: I001  isort:skip

import argparse
import inspect
import json
import math
import os
import random
import shutil
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

BASE_MODEL = "unsloth/SmolLM-135M"
MAX_SEQ_LEN = 1024
SEED = 3407

# Raw domain text for CPT. First entry that loads wins.
CPT_SOURCES = [
    # (hf_repo, config, split, text_column)
    ("eloukas/edgar-corpus", "year_2020", "train", "section_7"),   # MD&A prose
    ("virattt/financial-qa-10K", None, "train", "context"),        # always available
]
# General replay text. First entry that loads wins -- the bare `wikitext` name
# stopped resolving once the Hub required namespaced dataset ids.
GENERAL_SOURCES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text"),
    ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
]

SFT_SOURCE = ("virattt/financial-qa-10K", None, "train")

HF_USER = "sidhusarkar"                     # default Hub owner
RUN_NAME = "smollm-135m-fin-cpt-then-sft"   # default Hub repo name

_HERE = Path(__file__).resolve().parent
OUT = _HERE.parent / "reports" / "cpt_fin_then_sft"   # <repo>/CPT/reports/...
CPT_ADAPTER = OUT / "01_cpt_adapter"
CPT_MERGED = OUT / "02_cpt_merged"
SFT_ADAPTER = OUT / "03_sft_adapter"
FINAL_MERGED = OUT / "04_final_merged"

# Alpaca templates. The trailing EOS on the training side is what teaches the
# model to stop; the inference template deliberately omits it.
ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)
INSTRUCTION_PART = "### Instruction:\n"
RESPONSE_PART = "### Response:\n"

# Probes we fire at the model after every stage.
COMPLETION_PROBES = [
    "The Company recognized impairment charges of",
    "Total net revenue for the fiscal year ended",
]
INSTRUCTION_PROBES = [
    "What is EBITDA?",
    "What was the total revenue for fiscal year 2023?",
]
GENERAL_PROBE = "The cat sat on the"


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


def bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _keep_supported(fn_or_cls, kwargs: dict) -> dict:
    """Drop kwargs the installed trl/transformers version doesn't accept.

    trl keeps shuffling arguments between SFTTrainer and SFTConfig across
    releases; filtering by signature makes this script version-agnostic
    instead of pinning everyone to one trl.
    """
    target = fn_or_cls.__init__ if inspect.isclass(fn_or_cls) else fn_or_cls
    allowed = set(inspect.signature(target).parameters)
    if "kwargs" in allowed:  # **kwargs sink -- pass everything through
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _pick_trainer():
    """Prefer UnslothTrainer (native embedding_learning_rate), else plain TRL."""
    try:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        return UnslothTrainer, UnslothTrainingArguments
    except ImportError:
        from trl import SFTConfig, SFTTrainer
        return SFTTrainer, SFTConfig


# ----------------------------------------------------------------------------
# Hugging Face Hub
# ----------------------------------------------------------------------------

def resolve_repo(target: str, suffix: str = "") -> str:
    """`--push-to-hub` takes either `user/repo` or just `user`.

    Bare username -> `user/<RUN_NAME>`; the suffix distinguishes the adapter
    repos from the deployable merged one.
    """
    base = target if "/" in target else f"{target}/{RUN_NAME}"
    return base + suffix


def model_card(repo: str, report: dict, args) -> str:
    """A minimal but honest card: what was trained, in what order, and how to load it."""
    ppl = lambda key, dom: report.get(key, {}).get(dom, float("nan"))  # noqa: E731
    return f"""---
license: apache-2.0
base_model: {args.base}
library_name: transformers
tags:
- unsloth
- lora
- continued-pretraining
- sft
- finance
datasets:
- virattt/financial-qa-10K
---

# {repo.split('/')[-1]}

Two-stage domain adaptation of `{args.base}` on SEC 10-K financial text, run in
the **{report['order']}** order.

| stage | objective | LoRA | trained modules |
|---|---|---|---|
| 1 — CPT | all-token loss on raw 10-K prose (+20% general replay) | r=32 | attention + MLP + `embed_tokens` + `lm_head` |
| 2 — SFT | response-only loss on Alpaca-formatted 10-K QA | r=16 | attention + MLP |

## Results

| metric | base | after CPT | after SFT |
|---|---|---|---|
| domain perplexity | {ppl('ppl_before', 'domain'):.2f} | {ppl('ppl_after_cpt', 'domain'):.2f} | {ppl('ppl_after_sft', 'domain'):.2f} |
| general perplexity | {ppl('ppl_before', 'general'):.2f} | {ppl('ppl_after_cpt', 'general'):.2f} | {ppl('ppl_after_sft', 'general'):.2f} |

CPT first teaches the model to *sound* like the domain; SFT last leaves the
instruction-following behaviour as the freshest thing in the weights. The
companion `SFT -> CPT` run is the ablation showing what the reversed ordering
costs.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

prompt = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\\n\\n"
    "### Instruction:\\nWhat is EBITDA?\\n\\n### Response:\\n"
)
ids = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**ids, max_new_tokens=90)[0]))
```

Trained with [Unsloth](https://github.com/unslothai/unsloth) — `cpt_fin_then_sft.py`.
"""


def push_merged(model, tokenizer, repo: str, token: str | None, private: bool) -> None:
    banner(f"pushing merged model -> https://huggingface.co/{repo}")
    kwargs = _keep_supported(model.push_to_hub_merged, dict(
        repo_id=repo, tokenizer=tokenizer, save_method="merged_16bit",
        token=token, private=private,
    ))
    # Older unsloth names the first parameter `save_directory`, not `repo_id`.
    if "repo_id" not in kwargs:
        model.push_to_hub_merged(repo, **{k: v for k, v in kwargs.items()
                                          if k not in ("save_directory",)})
    else:
        model.push_to_hub_merged(**kwargs)


def push_adapter(model, tokenizer, repo: str, token: str | None, private: bool) -> None:
    banner(f"pushing LoRA adapter -> https://huggingface.co/{repo}")
    model.push_to_hub(repo, token=token, private=private)
    tokenizer.push_to_hub(repo, token=token, private=private)


def push_card(repo: str, text: str, token: str | None) -> None:
    """Card upload is best-effort: a failed README must not sink a finished run."""
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=text.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            token=token,
            commit_message="add model card",
        )
        print(f"[hub] model card written to {repo}")
    except Exception as exc:
        print(f"[hub] could not upload model card ({type(exc).__name__}: {exc})")


# ----------------------------------------------------------------------------
# Data: CPT corpus
# ----------------------------------------------------------------------------

def show_examples(label: str, texts: list[str], n: int = 2, width: int = 700) -> None:
    """Print real training strings, exactly as the trainer will see them.

    Silent data bugs -- a renamed text column, an empty context, a template
    whose response marker never renders -- are invisible in a loss curve and
    obvious the moment you look at the strings. `<eos>` should be visible at
    the end of every one; that trailing token is what teaches the model to
    stop.
    """
    if not texts:
        print(f"\n[{label}] EMPTY -- nothing will be trained from this split")
        return
    print(f"\n[{label}] {len(texts)} examples, showing {min(n, len(texts))}:")
    for i, text in enumerate(texts[:n]):
        body = text[:width] + (" ...[truncated]" if len(text) > width else "")
        print("  " + "-" * 68)
        print(f"  #{i}  ({len(text)} chars)")
        print("  | " + body.replace("\n", "\n  | "))
    print("  " + "-" * 68)


def chunk_words(text: str, size: int = 256, overlap: float = 0.2) -> list[str]:
    """Word-level chunking with overlap so no sentence is cut out of context."""
    words = text.split()
    if len(words) < 32:
        return []
    stride = max(1, int(size * (1 - overlap)))
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), stride)
        if len(words[i:i + size]) >= 32
    ]


def load_domain_chunks(n_docs: int) -> list[str]:
    for repo, config, split, column in CPT_SOURCES:
        try:
            print(f"[cpt-data] trying {repo} ({config or 'default'}) ...")
            ds = load_dataset(repo, config, split=split, streaming=True)
            docs, seen = [], 0
            for row in ds:
                text = (row.get(column) or "").strip()
                if len(text) > 400:
                    docs.append(text)
                seen += 1
                if len(docs) >= n_docs or seen >= n_docs * 20:
                    break
            if docs:
                print(f"[cpt-data] using {repo}: {len(docs)} documents")
                chunks = [c for d in docs for c in chunk_words(d)]
                print(f"[cpt-data] -> {len(chunks)} chunks of ~256 words")
                return chunks
        except Exception as exc:  # dataset moved, gated, or offline
            print(f"[cpt-data] {repo} unavailable ({type(exc).__name__}: {exc})")
    raise RuntimeError("No CPT corpus could be loaded. Check network / HF auth.")


def load_general_chunks(n: int) -> list[str]:
    for repo, config, split, column in GENERAL_SOURCES:
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
            out = []
            for row in ds:
                text = (row.get(column) or "").strip()
                if len(text.split()) > 60:
                    out.extend(chunk_words(text))
                if len(out) >= n:
                    break
            if out:
                print(f"[cpt-data] general replay: {repo}, {len(out[:n])} chunks")
                return out[:n]
        except Exception as exc:
            print(f"[cpt-data] {repo} unavailable ({type(exc).__name__}: {exc})")
    # Without replay there is no forgetting control and no general perplexity
    # number -- that is a broken experiment, not a degraded one.
    raise RuntimeError(
        "No general-replay corpus could be loaded; the catastrophic-forgetting "
        "mitigation and the general-perplexity metric both depend on it."
    )


def build_cpt_dataset(eos: str, n_docs: int, general_ratio: float):
    """80% domain / 20% general, per the forgetting-mitigation recipe."""
    domain = load_domain_chunks(n_docs)
    random.Random(SEED).shuffle(domain)

    holdout = max(50, len(domain) // 20)
    domain_eval, domain_train = domain[:holdout], domain[holdout:]

    n_general = int(len(domain_train) * general_ratio / max(1e-9, 1 - general_ratio))
    general = load_general_chunks(n_general + 100)
    general_eval, general_train = general[:100], general[100:]

    domain_texts = [t + eos for t in domain_train]
    general_texts = [t + eos for t in general_train]
    mixed = domain_texts + general_texts
    random.Random(SEED).shuffle(mixed)

    print(f"[cpt-data] train={len(mixed)} "
          f"(domain={len(domain_train)}, general={len(general_train)})")
    show_examples("cpt-data / domain", domain_texts, 2)
    show_examples("cpt-data / general replay", general_texts, 1)
    return (
        Dataset.from_dict({"text": mixed}),
        Dataset.from_dict({"text": [t + eos for t in domain_eval[:200]]}),
        domain_eval,     # raw text, for perplexity
        general_eval,    # raw text, for the forgetting check
    )


# ----------------------------------------------------------------------------
# Data: SFT corpus
# ----------------------------------------------------------------------------

def render_alpaca(instruction: str, inp: str, output: str, eos: str) -> str:
    tpl = ALPACA_WITH_INPUT if inp.strip() else ALPACA_NO_INPUT
    return tpl.format(instruction=instruction.strip(),
                      input=inp.strip(),
                      output=output.strip()) + eos


def build_sft_dataset(eos: str, n_rows: int, format_mix: float):
    repo, config, split = SFT_SOURCE
    ds = load_dataset(repo, config, split=split)
    ds = ds.shuffle(seed=SEED).select(range(min(n_rows, len(ds))))

    rng = random.Random(SEED)
    texts = []
    for row in ds:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        ctx = (row.get("context") or "").strip()
        if not q or not a:
            continue
        # Format-overfitting mitigation: show the model the question both with
        # and without the retrieved context, so it doesn't learn "answers only
        # exist when an ### Input block is present".
        use_ctx = ctx and rng.random() > format_mix
        texts.append(render_alpaca(q, ctx if use_ctx else "", a, eos))

    rng.shuffle(texts)
    split_at = max(1, int(len(texts) * 0.95))
    print(f"[sft-data] train={split_at}  eval={len(texts) - split_at}")
    # Two examples, because format_mix means the rendering differs row to row:
    # one should carry an ### Input block and one should not.
    with_input = [t for t in texts[:split_at] if "### Input:" in t]
    without_input = [t for t in texts[:split_at] if "### Input:" not in t]
    show_examples("sft-data / with ### Input", with_input, 1)
    show_examples("sft-data / no ### Input", without_input, 1)
    return (
        Dataset.from_dict({"text": texts[:split_at]}),
        Dataset.from_dict({"text": texts[split_at:]}),
    )


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(path: str, load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,                 # auto: bf16 where supported, else fp16
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


CPT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",       # CPT only: new domain vocabulary
]
SFT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def warn_if_overfit(trainer, label: str) -> dict:
    """Compare first and last eval_loss; a rise means the run memorised.

    Small fallback corpora make this easy to hit: 300 steps over a few hundred
    chunks is dozens of epochs, and train loss keeps falling while eval loss
    turns back up. The perplexity numbers downstream are then worse than the
    base model, which is a result worth seeing stated rather than inferred.
    """
    evals = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    if len(evals) < 2:
        return {}
    first, best, last = evals[0], min(evals), evals[-1]
    print(f"[fit {label}] eval_loss {first:.3f} -> {last:.3f} (best {best:.3f})")
    if last > first:
        print(f"  WARNING: eval loss ROSE by {last - first:.3f}. This stage "
              f"overfit -- train loss fell while held-out loss climbed. Lower "
              f"--{label}-steps, or raise --{label}-docs for a bigger corpus.")
    return {"eval_first": first, "eval_best": best, "eval_last": last}


def verify_merged(path, texts: list[str], expected: float, label: str) -> dict:
    """Reload the merged checkpoint and re-measure perplexity on it.

    Every number printed above this point came from the live PEFT model in
    memory. The merged directory is the thing people actually download, and
    nothing had ever loaded it back -- which is how a merge that corrupted the
    tied embed_tokens/lm_head weight stayed invisible in a summary that looked
    healthy. Compare, and refuse to call it fine if it is not.
    """
    try:
        merged, tok = load_model(str(path), False)
        got = perplexity(merged, tok, texts)
        ratio = got / max(1e-9, expected)
        print(f"[merge-check {label}] adapter ppl={expected:.2f}  "
              f"merged ppl={got:.2f}  (x{ratio:.2f})")
        if ratio > 1.5:
            print(f"  WARNING: the merged checkpoint at {path} scores {ratio:.1f}x "
                  f"worse than the adapter it came from. The merge lost "
                  f"something -- do not publish this checkpoint.")
        del merged
        torch.cuda.empty_cache()
        return {"adapter_ppl": expected, "merged_ppl": got, "ratio": ratio}
    except Exception as exc:
        print(f"[merge-check {label}] could not verify ({type(exc).__name__}: {exc})")
        return {}


def enable_training(model) -> None:
    """`FastLanguageModel.for_training` with the probe-then-train crash guarded.

    `for_inference()` stamps `_flag_for_generation` onto the modules it walks,
    and `for_training()` unconditionally `del`s it again. Probing the base
    model before stage 1 -- which is the whole point of the baseline -- means
    the LoRA wrapper added afterwards never got stamped, so the delete raises
    AttributeError. Seed the flag along the same walk so the delete is a no-op.

    `hasattr` is the wrong test here: PeftModel.__getattr__ delegates unknown
    attributes to the wrapped base model, so the flag reads as present on the
    wrapper while actually living on the inner module -- and `del` only removes
    entries from the object's *own* __dict__. Write straight into __dict__.
    """
    m, seen = model, set()
    while id(m) not in seen:
        seen.add(id(m))
        m.__dict__.setdefault("_flag_for_generation", True)
        inner = getattr(m, "model", None)
        if inner is None:
            break
        m = inner
    FastLanguageModel.for_training(model)


def resolve_cpt_targets(model, targets: list[str], freeze_embeddings: bool = False) -> list[str]:
    """Decide which of the embedding-adjacent modules CPT may adapt.

    Two separate hazards, both from `tie_word_embeddings=True` -- SmolLM makes
    `lm_head` and `embed_tokens` the *same* physical weight:

    1. Listing both puts two independent LoRA deltas on one matrix. peft warns
       that this breaks merging, and it does: the merged checkpoint scored far
       worse than the base model it came from. Always drop `lm_head`; training
       `embed_tokens` still moves the output head, because they are tied.

    2. Even one delta on a tied weight can survive training but not survive the
       merge. That shows up as an intermediate checkpoint whose perplexity is
       worse than the adapter's -- which is what `verify_merged` now measures at
       the stage-1 boundary. `--freeze-embeddings` is the escape hatch: adapt
       attention and MLP only, both of which merge cleanly.
    """
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    out = list(targets)
    if tied and "lm_head" in out:
        print("[lora] tie_word_embeddings=True -> dropping lm_head from the CPT "
              "targets (same weight as embed_tokens; adapting both corrupts the merge)")
        out = [t for t in out if t != "lm_head"]
    if freeze_embeddings and "embed_tokens" in out:
        print("[lora] --freeze-embeddings -> dropping embed_tokens; CPT will adapt "
              "attention and MLP only")
        out = [t for t in out if t != "embed_tokens"]
    return out


def attach_lora(model, *, rank: int, targets: list[str]):
    return FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank,             # alpha == r  =>  scaling factor of 1
        lora_dropout=0,              # Unsloth has a fast path for dropout=0
        target_modules=targets,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
    )


# ----------------------------------------------------------------------------
# Trainer construction
# ----------------------------------------------------------------------------

def make_trainer(model, tokenizer, train_ds, eval_ds, *, out_dir: Path,
                 lr: float, embedding_lr: float | None, max_steps: int,
                 epochs: float, packing: bool, batch: int, accum: int,
                 warmup: int):
    TrainerCls, ConfigCls = _pick_trainer()

    cfg_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_steps=warmup,
        learning_rate=lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="none",
        fp16=not bf16_ok(),
        bf16=bf16_ok(),
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=max(10, (max_steps or 200) // 5),
        per_device_eval_batch_size=batch,
        # SFTConfig-only knobs; silently dropped on plain TrainingArguments.
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=packing,
        dataset_num_proc=2,
    )
    if embedding_lr is not None:
        cfg_kwargs["embedding_learning_rate"] = embedding_lr
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    else:
        cfg_kwargs["num_train_epochs"] = epochs

    args = ConfigCls(**_keep_supported(ConfigCls, cfg_kwargs))

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # older trl wants these on the trainer; newer wants them on the config
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=packing,
        dataset_num_proc=2,
    )
    sig = set(inspect.signature(TrainerCls.__init__).parameters)
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer

    return TrainerCls(**_keep_supported(TrainerCls, trainer_kwargs))


def mask_prompt_tokens(trainer):
    """Restrict the SFT loss to response tokens (everything else -> -100)."""
    from unsloth.chat_templates import train_on_responses_only
    return train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )


def show_masking(trainer, tokenizer) -> None:
    """Print one example's supervised span so the masking is actually verified."""
    try:
        row = trainer.train_dataset[0]
        labels = row["labels"]
        kept = [t for t in labels if t != -100]
        pct = 100 * len(kept) / max(1, len(labels))
        print(f"[mask] supervising {len(kept)}/{len(labels)} tokens ({pct:.1f}%)")
        print(f"[mask] loss is computed on: {tokenizer.decode(kept)[:300]!r}")
    except Exception as exc:
        print(f"[mask] could not inspect masking: {exc}")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], limit: int = 120) -> float:
    model.eval()
    nll, ntok = 0.0, 0
    for text in texts[:limit]:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        loss = model(ids, labels=ids).loss.item()
        n = ids.shape[1] - 1
        nll += loss * n
        ntok += n
    if ntok == 0:                    # empty corpus -> no number, not "1.0"
        return float("nan")
    return math.exp(nll / ntok)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 90) -> str:
    FastLanguageModel.for_inference(model)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def probe(model, tokenizer, label: str) -> dict:
    banner(f"PROBES -- {label}")
    result = {}

    print("\n-- raw completion (does it sound like the domain?)")
    for p in COMPLETION_PROBES:
        text = generate(model, tokenizer, p, 60)
        result[f"completion::{p}"] = text
        print(f"  {p!r}\n    -> {text}\n")

    print("-- alpaca template (does it answer and stop?)")
    for q in INSTRUCTION_PROBES:
        prompt = ALPACA_NO_INPUT.format(instruction=q, output="")
        text = generate(model, tokenizer, prompt, 90)
        result[f"alpaca::{q}"] = text
        print(f"  {q!r}\n    -> {text}\n")

    print("-- bare question, no template (format-overfitting check)")
    q = INSTRUCTION_PROBES[0]
    text = generate(model, tokenizer, q, 60)
    result[f"bare::{q}"] = text
    print(f"  {q!r}\n    -> {text}\n")

    print("-- general english (catastrophic-forgetting check)")
    text = generate(model, tokenizer, GENERAL_PROBE, 30)
    result[f"general::{GENERAL_PROBE}"] = text
    print(f"  {GENERAL_PROBE!r}\n    -> {text}")

    return result


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="CPT then SFT")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--cpt-steps", type=int, default=300)
    ap.add_argument("--sft-steps", type=int, default=300)
    ap.add_argument("--cpt-docs", type=int, default=1500)
    ap.add_argument("--sft-rows", type=int, default=5000)
    ap.add_argument("--general-ratio", type=float, default=0.20,
                    help="fraction of the CPT mix that is general text")
    ap.add_argument("--format-mix", type=float, default=0.25,
                    help="fraction of SFT rows rendered without an ### Input block")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="exclude embed_tokens from the CPT LoRA targets; use "
                         "when the stage-1 merge check reports a degraded "
                         "intermediate checkpoint")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check wiring")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--push-to-hub", nargs="?", const=HF_USER, metavar="USER[/REPO]",
                    help="publish the final merged model to a HF profile; bare "
                         f"flag uses {HF_USER}/{RUN_NAME}, a bare username "
                         "becomes USER/<run name>")
    ap.add_argument("--push-adapters", action="store_true",
                    help="also publish the stage-1 and stage-2 LoRA adapters "
                         "as separate repos")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="defaults to $HF_TOKEN, else your cached CLI login")
    ap.add_argument("--hf-private", action="store_true",
                    help="create the Hub repos private instead of public")
    args = ap.parse_args()

    if args.smoke:
        args.cpt_steps = args.sft_steps = 30
        args.cpt_docs, args.sft_rows = 200, 400

    random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"order": "CPT -> SFT", "base": args.base}

    # ---------------------------------------------------------------- stage 0
    banner("STAGE 0 -- baseline")
    model, tokenizer = load_model(args.base, args.load_in_4bit)
    eos = tokenizer.eos_token

    cpt_train, cpt_eval, domain_eval_raw, general_eval_raw = build_cpt_dataset(
        eos, args.cpt_docs, args.general_ratio)

    report["ppl_before"] = {
        "domain": perplexity(model, tokenizer, domain_eval_raw),
        "general": perplexity(model, tokenizer, general_eval_raw),
    }
    print(f"[ppl] base -> {report['ppl_before']}")
    report["probes_base"] = probe(model, tokenizer, "base model")

    # ---------------------------------------------------------------- stage 1
    banner("STAGE 1 -- CPT (all-token loss, rank 32, embeddings unfrozen)")
    model = attach_lora(model, rank=32,
                    targets=resolve_cpt_targets(model, CPT_TARGETS,
                                                args.freeze_embeddings))
    model.print_trainable_parameters()
    enable_training(model)

    trainer = make_trainer(
        model, tokenizer, cpt_train, cpt_eval,
        out_dir=OUT / "cpt_ckpt",
        lr=2e-4,
        embedding_lr=2e-5,        # 10x lower: these layers touch every token
        max_steps=args.cpt_steps,
        epochs=1,
        packing=True,             # no wasted padding on raw text
        batch=args.batch,
        accum=args.accum,
        warmup=min(100, max(5, args.cpt_steps // 10)),
    )
    cpt_stats = trainer.train()
    report["fit_cpt"] = warn_if_overfit(trainer, "cpt")
    report["cpt_train_loss"] = cpt_stats.training_loss

    report["ppl_after_cpt"] = {
        "domain": perplexity(model, tokenizer, domain_eval_raw),
        "general": perplexity(model, tokenizer, general_eval_raw),
    }
    print(f"[ppl] after CPT -> {report['ppl_after_cpt']}")
    report["probes_after_cpt"] = probe(model, tokenizer, "after CPT")

    model.save_pretrained(str(CPT_ADAPTER))
    tokenizer.save_pretrained(str(CPT_ADAPTER))
    banner("merging CPT adapter into the base weights")
    model.save_pretrained_merged(str(CPT_MERGED), tokenizer,
                                 save_method="merged_16bit")
    # Stage 2 trains on THIS file, not on the model still in memory.
    report["merge_check_stage1"] = verify_merged(
        CPT_MERGED, domain_eval_raw,
        report["ppl_after_cpt"]["domain"], "cpt-intermediate")

    if args.push_to_hub and args.push_adapters:
        push_adapter(model, tokenizer,
                     resolve_repo(args.push_to_hub, "-stage1-cpt-lora"),
                     args.hf_token, args.hf_private)

    del model, trainer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- stage 2
    banner("STAGE 2 -- SFT (response-only loss, rank 16, embeddings frozen)")
    model, tokenizer = load_model(str(CPT_MERGED), args.load_in_4bit)
    eos = tokenizer.eos_token

    sft_train, sft_eval = build_sft_dataset(eos, args.sft_rows, args.format_mix)

    model = attach_lora(model, rank=16, targets=SFT_TARGETS)
    model.print_trainable_parameters()
    enable_training(model)

    trainer = make_trainer(
        model, tokenizer, sft_train, sft_eval,
        out_dir=OUT / "sft_ckpt",
        lr=2e-4,
        embedding_lr=None,        # vocabulary is fine; only behaviour changes
        max_steps=args.sft_steps,
        epochs=2,
        packing=False,            # required: response-only masking needs
                                  # one example per sequence
        batch=args.batch,
        accum=args.accum,
        warmup=min(50, max(5, args.sft_steps // 10)),
    )
    trainer = mask_prompt_tokens(trainer)
    show_masking(trainer, tokenizer)

    sft_stats = trainer.train()
    report["fit_sft"] = warn_if_overfit(trainer, "sft")
    report["sft_train_loss"] = sft_stats.training_loss

    report["ppl_after_sft"] = {
        "domain": perplexity(model, tokenizer, domain_eval_raw),
        "general": perplexity(model, tokenizer, general_eval_raw),
    }
    print(f"[ppl] after SFT -> {report['ppl_after_sft']}")
    report["probes_final"] = probe(model, tokenizer, "after CPT + SFT (final)")

    # ---------------------------------------------------------------- export
    model.save_pretrained(str(SFT_ADAPTER))
    tokenizer.save_pretrained(str(SFT_ADAPTER))
    banner("merging SFT adapter -> deployable single-checkpoint model")
    model.save_pretrained_merged(str(FINAL_MERGED), tokenizer,
                                 save_method="merged_16bit")
    report["merge_check"] = verify_merged(
        FINAL_MERGED, domain_eval_raw,
        report["ppl_after_sft"]["domain"], "final")

    if args.push_to_hub:
        repo = resolve_repo(args.push_to_hub)
        push_merged(model, tokenizer, repo, args.hf_token, args.hf_private)
        push_card(repo, model_card(repo, report, args), args.hf_token)
        report["hub_model"] = f"https://huggingface.co/{repo}"
        if args.push_adapters:
            adapter_repo = resolve_repo(args.push_to_hub, "-stage2-sft-lora")
            push_adapter(model, tokenizer, adapter_repo,
                         args.hf_token, args.hf_private)
            report["hub_adapters"] = [
                f"https://huggingface.co/{resolve_repo(args.push_to_hub, '-stage1-cpt-lora')}",
                f"https://huggingface.co/{adapter_repo}",
            ]

    if not args.keep_intermediate:
        shutil.rmtree(CPT_MERGED, ignore_errors=True)
        print(f"[cleanup] removed {CPT_MERGED} (pass --keep-intermediate to retain)")

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    banner("SUMMARY -- CPT -> SFT")
    print(f"  domain ppl : {report['ppl_before']['domain']:.1f}"
          f" -> {report['ppl_after_cpt']['domain']:.1f} (CPT)"
          f" -> {report['ppl_after_sft']['domain']:.1f} (SFT)")
    print(f"  general ppl: {report['ppl_before']['general']:.1f}"
          f" -> {report['ppl_after_cpt']['general']:.1f} (CPT)"
          f" -> {report['ppl_after_sft']['general']:.1f} (SFT)")
    print(f"\n  final model : {FINAL_MERGED}")
    print(f"  cpt adapter : {CPT_ADAPTER}")
    print(f"  sft adapter : {SFT_ADAPTER}")
    print(f"  report      : {OUT / 'report.json'}")
    if report.get("hub_model"):
        print(f"  on the hub  : {report['hub_model']}")
        for url in report.get("hub_adapters", []):
            print(f"                {url}")
    print("\nExpectation: domain perplexity drops sharply after CPT and stays "
          "low; the alpaca probes only start producing short, terminated "
          "answers after SFT.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
