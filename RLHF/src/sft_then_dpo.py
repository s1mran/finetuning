"""
sft_then_dpo.py
===============

Preference alignment of a small base LLM, in the standard RLHF ordering:

    base model  ->  Stage 1: SFT  ->  merge  ->  Stage 2: DPO  ->  merge  ->  final

Stage 1 (SFT / supervised fine-tuning)
    Empathetic dialogue turns from `Estwld/empathetic_dialogues_llm`, rendered
    Alpaca-style with the speaker's situation as context. Loss masked to the
    response span only. This is the stage that teaches *warmth*: how a caring
    reply is shaped, what it acknowledges before it advises.

Stage 2 (DPO / direct preference optimisation)
    Human preference pairs (chosen vs rejected). No reward model is trained --
    DPO folds the reward model into the policy's own log-ratio against a frozen
    reference. See the loss commentary above `make_dpo_trainer`.

A note on the data, stated plainly: there is no public preference dataset
*about empathy*. The empathy signal comes from stage 1; stage 2 uses general
human preference data for response quality. Do not read the DPO stage as
"learning to be kind" -- it is learning what annotators preferred, which is a
different and much leakier target. That gap is the whole reason the
reward-hacking guards below exist.

Reward hacking
--------------
There is no learned reward model here to game, but DPO is still optimising a
*proxy*: annotator preference. Known ways a model wins on the proxy while
getting worse in reality, and what this script does about each:

  1. Length exploitation -- annotators prefer longer answers, so the model
     learns "longer = better" and pads. Guarded by `--max-length-ratio`, which
     drops pairs whose chosen response is disproportionately longer than the
     rejected one, and by reporting mean response length at every stage.
  2. Drift from the reference -- the policy wanders far from the SFT model to
     chase preference signal. Guarded by beta (the KL leash) and by tracking
     general-text perplexity across stages.
  3. Degeneration -- repetitive, high-confidence filler that scores well on
     fluency. Guarded by the repetition metric in `response_stats`.
  4. Overfitting the preference set -- train accuracy saturates while held-out
     accuracy stalls. Guarded by `preference_accuracy` on a held-out split,
     measured before and after.

None of these are solved, only measured. A guard you cannot see the effect of
is a guard you should not trust.

Requirements
------------
    pip install unsloth trl peft transformers datasets accelerate bitsandbytes

Needs a CUDA GPU (a free Colab T4 is plenty for a 135M model). Unsloth does
not support Apple MPS.

Usage
-----
    python sft_then_dpo.py                     # full run, ~20 min on a T4
    python sft_then_dpo.py --smoke             # 30 steps per stage, sanity check
    python sft_then_dpo.py --beta 0.3          # tighter KL leash
    python sft_then_dpo.py --max-length-ratio 1.2   # stricter length filter

    # publish to huggingface.co/sidhusarkar (needs `huggingface-cli login` or
    # HF_TOKEN with write scope)
    python sft_then_dpo.py --push-to-hub
"""

from __future__ import annotations

# Unsloth must be imported before transformers/trl so its patches land.
from unsloth import FastLanguageModel  # noqa: I001  isort:skip

# Older unsloth needs an explicit DPO patch before trl is imported; newer
# releases patch on import and drop the symbol entirely.
try:  # noqa: SIM105
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
except ImportError:
    pass

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

BASE_MODEL = "HuggingFaceTB/SmolLM-135M"
MAX_SEQ_LEN = 512
MAX_PROMPT_LEN = 256
SEED = 3407

# Empathetic dialogue for stage 1. First entry that loads wins.
SFT_SOURCES = [
    # (hf_repo, config, split)
    ("Estwld/empathetic_dialogues_llm", None, "train"),
    ("lavita/ChatDoctor-HealthCareMagic-100k", None, "train"),
]
# Preference pairs for stage 2. First entry that loads wins.
#
# rm-static leads because it matches what stage 1 teaches. Stage 1 SFTs short
# empathetic dialogue turns (~10 tokens); UltraFeedback's chosen responses
# average ~4,800 characters of essay and code. Preference accuracy is measured
# by summed log-probability over the response, so a model tuned to be terse
# scores those long answers down for reasons that have nothing to do with
# preference -- which is what dropped pref_acc from 0.610 to 0.559 after SFT.
# rm-static is hh-rlhf-derived conversation, ~210 characters, already split
# into prompt/chosen/rejected.
PREF_SOURCES = [
    ("Dahoas/rm-static", None, "train"),
    ("argilla/ultrafeedback-binarized-preferences-cleaned", None, "train"),
    ("Anthropic/hh-rlhf", None, "train"),
]
# General text, for the drift check only -- never trained on here.
GENERAL_SOURCES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text"),
    ("wikimedia/wikipedia", "20231101.en", "train", "text"),
]

HF_USER = "sidhusarkar"                  # default Hub owner
RUN_NAME = "smollm-135m-empathy-sft-then-dpo"

_HERE = Path(__file__).resolve().parent
OUT = _HERE.parent / "reports" / "sft_then_dpo"    # <repo>/RLHF/reports/...
SFT_ADAPTER = OUT / "01_sft_adapter"
SFT_MERGED = OUT / "02_sft_merged"
DPO_ADAPTER = OUT / "03_dpo_adapter"
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

# Probes we fire at the model after every stage. Emotional disclosures, because
# that is what stage 1 is meant to change: does the model acknowledge the
# feeling before it starts problem-solving?
EMPATHY_PROBES = [
    "I just found out I didn't get the job I really wanted.",
    "My dog died last week and the house feels so empty.",
    "I'm really nervous about my exam tomorrow.",
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

    trl keeps shuffling arguments between trainer and config across releases;
    filtering by signature makes this script version-agnostic instead of
    pinning everyone to one trl.
    """
    target = fn_or_cls.__init__ if inspect.isclass(fn_or_cls) else fn_or_cls
    allowed = set(inspect.signature(target).parameters)
    if "kwargs" in allowed:  # **kwargs sink -- pass everything through
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def show_examples(label: str, texts: list[str], n: int = 2, width: int = 700) -> None:
    """Print real training strings, exactly as the trainer will see them.

    Silent data bugs -- a renamed column, an empty context, a template whose
    response marker never renders -- are invisible in a loss curve and obvious
    the moment you look at the strings.
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


def warn_if_overfit(trainer, label: str) -> dict:
    """Compare first and last eval_loss; a rise means the stage memorised."""
    evals = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    if len(evals) < 2:
        return {}
    first, best, last = evals[0], min(evals), evals[-1]
    print(f"[fit {label}] eval_loss {first:.3f} -> {last:.3f} (best {best:.3f})")
    if last > first:
        print(f"  WARNING: eval loss ROSE by {last - first:.3f}. This stage "
              f"overfit -- train loss fell while held-out loss climbed.")
    return {"eval_first": first, "eval_best": best, "eval_last": last}


def verify_merged(path, texts: list[str], expected: float, label: str,
                  fatal: bool = False) -> dict:
    """Reload the merged checkpoint and re-measure perplexity on it.

    Everything above this point is measured on the live PEFT model. The merged
    directory is what stage 2 loads and what people download, and until it is
    read back nothing has checked it. In the CPT scripts that blind spot hid a
    102x regression behind a summary that looked healthy.

    This script's LoRA never touches embed_tokens or lm_head, so it is not
    exposed to the tied-weight merge failure -- but "not exposed to the one we
    found" is not the same as verified.
    """
    try:
        merged, tok = load_model(str(path), False)
        got = perplexity(merged, tok, texts)
        ratio = got / max(1e-9, expected)
        print(f"[merge-check {label}] adapter ppl={expected:.2f}  "
              f"merged ppl={got:.2f}  (x{ratio:.2f})")
        if ratio > 1.5:
            print(f"  WARNING: the merged checkpoint at {path} scores {ratio:.1f}x "
                  f"worse than the adapter it came from -- do not publish it.")
            if fatal:
                raise SystemExit(
                    "\n[abort] Stage 2 trains on this file, so continuing would "
                    "spend the whole stage on a broken base.\n"
                    "         Pass --allow-bad-merge to continue anyway.")
        del merged
        torch.cuda.empty_cache()
        return {"adapter_ppl": expected, "merged_ppl": got, "ratio": ratio}
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[merge-check {label}] could not verify ({type(exc).__name__}: {exc})")
        return {}


def save_merged(model, tokenizer, path) -> None:
    """Collapse the adapter into the base weights and write one checkpoint.

    Unsloth's fast merge maps LoRA stats onto tensors by NAME, and on some
    architectures it pairs the wrong ones: on Qwen2.5-1.5B it tried to add a
    [1536, 1536] product into a [1536, 8960] MLP tensor and raised

        RuntimeError: Bad in-place call: input tensor size [1536, 1536] and
        output tensor size [1536, 8960] should match

    peft's own merge walks the module tree instead, so it cannot mismatch a
    projection with an MLP. It is slower and wants the weights unquantised,
    which is the trade -- if it also fails, re-run with --no-4bit.
    """
    try:
        model.save_pretrained_merged(str(path), tokenizer,
                                     save_method="merged_16bit")
        return
    except Exception as exc:
        print(f"[merge] unsloth fast merge failed "
              f"({type(exc).__name__}: {str(exc)[:160]})")
        print("[merge] falling back to peft merge_and_unload")
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(path), safe_serialization=True)
        tokenizer.save_pretrained(str(path))
        print(f"[merge] wrote {path} via peft")
    except Exception as exc:
        quantised = getattr(getattr(model, "config", None), "quantization_config", None)
        detail = str(exc).strip() or type(exc).__name__
        hint = (
            "        The model is loaded in 4-bit. peft can fold the adapter into\n"
            "        quantised weights, but transformers cannot serialise a 4-bit\n"
            "        model, so the save is what fails -- not the merge.\n"
            "        Re-run with --no-4bit; this model in fp16 fits a T4.\n"
            if quantised is not None else
            "        Neither merge path could write this checkpoint. --no-4bit is\n"
            "        worth trying; it is the path with the fewest special cases.\n")
        raise SystemExit(
            f"\n[abort] could not write a merged checkpoint at {path}.\n"
            f"        peft raised {type(exc).__name__}: {detail[:200]}\n"
            + hint) from exc


def enable_training(model) -> None:
    """`FastLanguageModel.for_training` with the probe-then-train crash guarded.

    `for_inference()` stamps `_flag_for_generation` onto the modules it walks,
    and `for_training()` unconditionally `del`s it again. Probing the base
    model before stage 1 means the LoRA wrapper added afterwards never got
    stamped, so the delete raises AttributeError.

    `hasattr` is the wrong test: PeftModel.__getattr__ delegates unknown
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


# ----------------------------------------------------------------------------
# Hugging Face Hub
# ----------------------------------------------------------------------------

def resolve_repo(target: str, suffix: str = "") -> str:
    """`--push-to-hub` takes either `user/repo` or just `user`."""
    base = target if "/" in target else f"{target}/{RUN_NAME}"
    return base + suffix


def model_card(repo: str, report: dict, args) -> str:
    """A minimal but honest card, including the metrics that would expose hacking."""
    def g(*keys, default=float("nan")):
        """Walk nested report keys, yielding nan for anything not a number.

        A run interrupted after stage 1 leaves later keys missing, and the
        stage dicts are nested; both must degrade to `nan` rather than throwing
        a format error and taking the whole card down with them.
        """
        cur = report
        for k in keys:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            if cur is None:
                return default
        return cur if isinstance(cur, (int, float)) else default

    return f"""---
license: apache-2.0
base_model: {args.base}
library_name: transformers
tags:
- unsloth
- lora
- sft
- dpo
- rlhf
datasets:
- Estwld/empathetic_dialogues_llm
---

# {repo.split('/')[-1]}

Preference alignment of `{args.base}`: SFT on empathetic dialogue, then DPO on
human preference pairs.

| stage | objective | LoRA | loss |
|---|---|---|---|
| 1 — SFT | empathetic dialogue, response-only loss | r=16 | token cross-entropy on the response span |
| 2 — DPO | human preference pairs | r=16 | `-log σ(β · (Δ log-ratio chosen − Δ log-ratio rejected))` |

## Results

| metric | base | after SFT | after DPO |
|---|---|---|---|
| held-out preference accuracy | {g('pref_acc_before'):.3f} | {g('pref_acc_after_sft'):.3f} | {g('pref_acc_after_dpo'):.3f} |
| mean response length (tokens) | {g('stats_base', 'mean_tokens'):.1f} | {g('stats_after_sft', 'mean_tokens'):.1f} | {g('stats_after_dpo', 'mean_tokens'):.1f} |
| repetition rate | {g('stats_base', 'repetition'):.3f} | {g('stats_after_sft', 'repetition'):.3f} | {g('stats_after_dpo', 'repetition'):.3f} |
| general perplexity (drift) | {g('ppl_before'):.2f} | {g('ppl_after_sft'):.2f} | {g('ppl_after_dpo'):.2f} |

`beta = {args.beta}` (KL leash), preference pairs filtered at a chosen/rejected
length ratio of {args.max_length_ratio}.

**Read the length and repetition rows before the accuracy row.** Preference
accuracy that climbs while responses get markedly longer or more repetitive is
the signature of reward hacking, not alignment.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

prompt = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\\n\\n"
    "### Instruction:\\nI'm really nervous about my exam tomorrow.\\n\\n"
    "### Response:\\n"
)
ids = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**ids, max_new_tokens=90)[0]))
```

Trained with [Unsloth](https://github.com/unslothai/unsloth) — `sft_then_dpo.py`.
"""


def push_merged(model, tokenizer, repo: str, token: str | None, private: bool) -> None:
    banner(f"pushing merged model -> https://huggingface.co/{repo}")
    kwargs = _keep_supported(model.push_to_hub_merged, dict(
        repo_id=repo, tokenizer=tokenizer, save_method="merged_16bit",
        token=token, private=private,
    ))
    if "repo_id" not in kwargs:
        model.push_to_hub_merged(repo, **{k: v for k, v in kwargs.items()
                                          if k not in ("save_directory",)})
    else:
        model.push_to_hub_merged(**kwargs)


def push_card(repo: str, text: str, token: str | None) -> None:
    """Card upload is best-effort: a failed README must not sink a finished run."""
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=text.encode(), path_in_repo="README.md",
            repo_id=repo, repo_type="model", token=token,
            commit_message="add model card",
        )
        print(f"[hub] model card written to {repo}")
    except Exception as exc:
        print(f"[hub] could not upload model card ({type(exc).__name__}: {exc})")


# ----------------------------------------------------------------------------
# Data: empathetic SFT corpus
# ----------------------------------------------------------------------------

def render_alpaca(instruction: str, inp: str, output: str, eos: str) -> str:
    tpl = ALPACA_WITH_INPUT if inp.strip() else ALPACA_NO_INPUT
    return tpl.format(instruction=instruction.strip(), input=inp.strip(),
                      output=output.strip()) + eos


def _rows_from_empathetic(row, rng, format_mix, eos) -> list[str]:
    """One dialogue -> one training string per assistant turn.

    `conversations` alternates user/assistant. Every assistant turn is a
    supervised target whose instruction is the user turn before it; `situation`
    becomes optional context so the model does not learn that empathy requires
    a briefing.
    """
    convo = row.get("conversations") or []
    situation = (row.get("situation") or "").strip()
    out = []
    for i, turn in enumerate(convo):
        if turn.get("role") != "assistant" or i == 0:
            continue
        user = (convo[i - 1].get("content") or "").strip()
        reply = (turn.get("content") or "").strip()
        if len(user) < 8 or len(reply) < 8:
            continue
        use_ctx = situation and rng.random() > format_mix
        out.append(render_alpaca(user, situation if use_ctx else "", reply, eos))
    return out


def _rows_from_alpaca_like(row, rng, format_mix, eos) -> list[str]:
    """Fallback shape: an already Alpaca-columned dataset."""
    q = (row.get("instruction") or "").strip()
    a = (row.get("output") or "").strip()
    ctx = (row.get("input") or "").strip()
    if not q or not a:
        return []
    use_ctx = ctx and rng.random() > format_mix
    if use_ctx:
        return [render_alpaca(q, ctx, a, eos)]
    return [render_alpaca(f"{q}\n\n{ctx}".strip(), "", a, eos)]


def build_sft_dataset(eos: str, n_rows: int, format_mix: float):
    rng = random.Random(SEED)
    for repo, config, split in SFT_SOURCES:
        try:
            print(f"[sft-data] trying {repo} ...")
            ds = load_dataset(repo, config, split=split)
            ds = ds.shuffle(seed=SEED).select(range(min(n_rows, len(ds))))
            cols = set(ds.column_names)
            render = (_rows_from_empathetic if "conversations" in cols
                      else _rows_from_alpaca_like)
            texts = [t for row in ds for t in render(row, rng, format_mix, eos)]
            if texts:
                print(f"[sft-data] using {repo}: {len(texts)} training strings")
                break
        except Exception as exc:
            print(f"[sft-data] {repo} unavailable ({type(exc).__name__}: {exc})")
    else:
        raise RuntimeError("No SFT corpus could be loaded. Check network / HF auth.")

    rng.shuffle(texts)
    split_at = max(1, int(len(texts) * 0.95))
    print(f"[sft-data] train={split_at}  eval={len(texts) - split_at}")
    with_input = [t for t in texts[:split_at] if "### Input:" in t]
    without_input = [t for t in texts[:split_at] if "### Input:" not in t]
    show_examples("sft-data / with ### Input", with_input, 1)
    show_examples("sft-data / no ### Input", without_input, 1)
    return (Dataset.from_dict({"text": texts[:split_at]}),
            Dataset.from_dict({"text": texts[split_at:]}))


# ----------------------------------------------------------------------------
# Data: preference pairs
# ----------------------------------------------------------------------------

def _last_assistant(messages) -> str:
    """Extract the final assistant message from a chat-formatted column."""
    if isinstance(messages, str):
        return messages.strip()
    for turn in reversed(messages or []):
        if turn.get("role") == "assistant":
            return (turn.get("content") or "").strip()
    return ""


def _split_hh(text: str) -> tuple[str, str]:
    """`Anthropic/hh-rlhf` stores whole transcripts; split off the last reply."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()
    return text[:idx + len(marker)].strip(), text[idx + len(marker):].strip()


def _normalise_pairs(ds) -> list[dict]:
    """Reduce whatever shape the source uses to {prompt, chosen, rejected}."""
    cols = set(ds.column_names)
    pairs = []
    for row in ds:
        if "prompt" in cols:                       # ultrafeedback-style
            prompt = (row.get("prompt") or "").strip()
            chosen = _last_assistant(row.get("chosen"))
            rejected = _last_assistant(row.get("rejected"))
        else:                                      # hh-rlhf transcripts
            p_c, chosen = _split_hh(row.get("chosen") or "")
            p_r, rejected = _split_hh(row.get("rejected") or "")
            # Only usable when both sides share a prompt; otherwise the pair is
            # comparing answers to two different questions.
            if p_c != p_r:
                continue
            prompt = p_c
        if prompt and chosen and rejected and chosen != rejected:
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def _length_stats(pairs: list[dict]) -> tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    c = sum(len(p["chosen"]) for p in pairs) / len(pairs)
    r = sum(len(p["rejected"]) for p in pairs) / len(pairs)
    return c, r


def build_pref_dataset(n_rows: int, max_length_ratio: float, eval_frac: float = 0.05):
    """Load preference pairs and strip the length bias out of them.

    GUARD 1 (length exploitation). Human annotators systematically prefer longer
    answers, so in most preference corpora the chosen response is meaningfully
    longer than the rejected one. Train on that unfiltered and DPO will happily
    learn "more tokens = higher reward" -- the model gets verbose, scores better
    on the proxy, and is worse to talk to. Dropping pairs whose chosen response
    is more than `max_length_ratio` times the rejected one removes most of the
    signal that rewards padding, at the cost of some training data.
    """
    for repo, config, split in PREF_SOURCES:
        try:
            print(f"[pref-data] trying {repo} ...")
            ds = load_dataset(repo, config, split=split)
            ds = ds.shuffle(seed=SEED).select(range(min(n_rows * 2, len(ds))))
            pairs = _normalise_pairs(ds)
            if pairs:
                print(f"[pref-data] using {repo}: {len(pairs)} usable pairs")
                break
        except Exception as exc:
            print(f"[pref-data] {repo} unavailable ({type(exc).__name__}: {exc})")
    else:
        raise RuntimeError("No preference corpus could be loaded. Check network / HF auth.")

    before_c, before_r = _length_stats(pairs)
    kept = [p for p in pairs
            if len(p["chosen"]) <= max_length_ratio * max(1, len(p["rejected"]))]
    after_c, after_r = _length_stats(kept)

    print(f"[pref-data] length bias  chosen/rejected mean chars: "
          f"{before_c:.0f}/{before_r:.0f} (ratio {before_c / max(1e-9, before_r):.2f})")
    print(f"[pref-data] after ratio filter <= {max_length_ratio}: "
          f"{after_c:.0f}/{after_r:.0f} (ratio {after_c / max(1e-9, after_r):.2f})")
    print(f"[pref-data] dropped {len(pairs) - len(kept)}/{len(pairs)} pairs "
          f"({100 * (len(pairs) - len(kept)) / max(1, len(pairs)):.1f}%) as length-biased")
    if not kept:
        raise RuntimeError(
            f"--max-length-ratio {max_length_ratio} removed every pair; raise it.")

    kept = kept[:n_rows]
    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_eval = max(1, int(len(kept) * eval_frac))
    eval_pairs, train_pairs = kept[:n_eval], kept[n_eval:]

    show_examples("pref-data / prompt", [p["prompt"] for p in train_pairs], 1)
    show_examples("pref-data / chosen", [p["chosen"] for p in train_pairs], 1)
    show_examples("pref-data / rejected", [p["rejected"] for p in train_pairs], 1)
    print(f"[pref-data] train={len(train_pairs)}  eval={len(eval_pairs)}")

    def to_ds(rows):
        return Dataset.from_dict({
            "prompt": [ALPACA_NO_INPUT.format(instruction=r["prompt"], output="")
                       for r in rows],
            "chosen": [r["chosen"] for r in rows],
            "rejected": [r["rejected"] for r in rows],
        })

    return to_ds(train_pairs), to_ds(eval_pairs), eval_pairs


def load_general_chunks(n: int) -> list[str]:
    """General text for the drift check. Never trained on in this script."""
    for repo, config, split, column in GENERAL_SOURCES:
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
            out = []
            for row in ds:
                text = (row.get(column) or "").strip()
                if len(text.split()) > 60:
                    out.append(text)
                if len(out) >= n:
                    break
            if out:
                print(f"[drift] general eval: {repo}, {len(out)} passages")
                return out
        except Exception as exc:
            print(f"[drift] {repo} unavailable ({type(exc).__name__}: {exc})")
    raise RuntimeError("No general corpus for the drift check could be loaded.")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(path: str, load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def attach_lora(model, *, rank: int):
    return FastLanguageModel.get_peft_model(
        model, r=rank,
        lora_alpha=rank,             # alpha == r  =>  scaling factor of 1
        lora_dropout=0,              # Unsloth has a fast path for dropout=0
        target_modules=LORA_TARGETS,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
    )


# ----------------------------------------------------------------------------
# Stage 1 trainer -- supervised cross-entropy
# ----------------------------------------------------------------------------

def make_sft_trainer(model, tokenizer, train_ds, eval_ds, *, out_dir: Path,
                     lr: float, max_steps: int, epochs: float,
                     batch: int, accum: int, warmup: int):
    """Ordinary next-token cross-entropy, masked to the response span.

    LOSS
        L_SFT = -(1/|R|) * sum_{t in R} log p_theta(y_t | y_<t, x)

    where R is the set of response-token positions. Prompt tokens are set to
    -100 so they contribute nothing: the model is not being taught to predict
    the user's own words, only how to answer them. Every response token is
    weighted equally, which is why SFT teaches *shape* -- phrasing, structure,
    when to stop -- and cannot express "this answer is better than that one".
    Expressing that is exactly what stage 2 is for.
    """
    from trl import SFTConfig, SFTTrainer

    cfg_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_steps=warmup,
        learning_rate=lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=SEED,
        report_to="none",
        fp16=not bf16_ok(),
        bf16=bf16_ok(),
        eval_strategy="steps",
        eval_steps=max(10, (max_steps or 200) // 5),
        # Ship the checkpoint with the lowest held-out loss rather than
        # whichever one the last step happened to produce.
        save_strategy="steps",
        save_steps=max(10, (max_steps or 200) // 5),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size=batch,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,          # required: response-only masking needs one
                                # example per sequence
        dataset_num_proc=2,
    )
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    else:
        cfg_kwargs["num_train_epochs"] = epochs

    args = SFTConfig(**_keep_supported(SFTConfig, cfg_kwargs))
    trainer_kwargs = dict(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        packing=False, dataset_num_proc=2,
    )
    sig = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer
    return SFTTrainer(**_keep_supported(SFTTrainer, trainer_kwargs))


def mask_prompt_tokens(trainer):
    """Restrict the SFT loss to response tokens (everything else -> -100)."""
    from unsloth.chat_templates import train_on_responses_only
    return train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)


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
# Stage 2 trainer -- direct preference optimisation
# ----------------------------------------------------------------------------

def make_dpo_trainer(model, tokenizer, train_ds, eval_ds, *, out_dir: Path,
                     lr: float, beta: float, max_steps: int, epochs: float,
                     batch: int, accum: int, warmup: int):
    """Direct Preference Optimisation.

    LOSS
        L_DPO = -log sigmoid( beta * ( r_theta(x, y_w) - r_theta(x, y_l) ) )

        where  r_theta(x, y) = log pi_theta(y|x) - log pi_ref(y|x)

    Read it in three moves:

    1. `r_theta` is an *implicit reward*. Classic RLHF trains a separate reward
       model on preference pairs and then optimises the policy against it with
       PPO. DPO's derivation shows the optimal policy for that objective can be
       written in closed form, which lets you skip the reward model entirely:
       the policy's own log-ratio against a frozen reference *is* the reward.
       Fewer moving parts, no reward-model overfitting, no PPO instability.

    2. The loss is a logistic loss on the *difference* of those rewards. It only
       ever asks that chosen outscore rejected. It says nothing about absolute
       quality, which is why DPO cannot fix a behaviour the SFT model never
       exhibits -- it can only reweight what is already in the distribution.

    3. `beta` is the KL leash. It appears in the derivation as the coefficient
       of a KL(pi_theta || pi_ref) penalty. Low beta lets the policy drift far
       from the reference to chase preference signal, which is where
       degeneration and length blowup come from; high beta keeps it close and
       learns less. 0.1 is the common default; raise it if the guards below
       start firing.

    REFERENCE MODEL
        `ref_model=None` is deliberate. With a PEFT model, TRL computes the
        reference log-probs by *disabling the adapter*, which recovers exactly
        the merged stage-1 SFT weights. That is the correct reference -- the KL
        leash is anchored to the model we just supervised, not to the raw base
        model -- and it costs no extra GPU memory.

    WHAT TO WATCH IN THE LOGS
        `rewards/accuracies` -- fraction of pairs where chosen outscores
        rejected. Climbing to ~1.0 within a few dozen steps means the model is
        memorising the preference set, not learning a preference.
        `rewards/margins`   -- mean reward gap. Large and growing alongside
        rising response length is the length-exploitation signature.
    """
    from trl import DPOConfig, DPOTrainer

    cfg_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_steps=warmup,
        # DPO wants a markedly lower LR than SFT: it is nudging an already
        # competent policy, and a large step here is what produces collapse.
        learning_rate=lr,
        beta=beta,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=SEED,
        report_to="none",
        fp16=not bf16_ok(),
        bf16=bf16_ok(),
        eval_strategy="steps",
        eval_steps=max(10, (max_steps or 200) // 5),
        # Ship the checkpoint with the lowest held-out loss rather than
        # whichever one the last step happened to produce.
        save_strategy="steps",
        save_steps=max(10, (max_steps or 200) // 5),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size=batch,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        dataset_num_proc=2,
    )
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    else:
        cfg_kwargs["num_train_epochs"] = epochs

    args = DPOConfig(**_keep_supported(DPOConfig, cfg_kwargs))
    trainer_kwargs = dict(
        model=model,
        ref_model=None,          # see REFERENCE MODEL above
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        beta=beta,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
    )
    sig = set(inspect.signature(DPOTrainer.__init__).parameters)
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer
    return DPOTrainer(**_keep_supported(DPOTrainer, trainer_kwargs))


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], limit: int = 80) -> float:
    """GUARD 2 (drift). General-text perplexity should stay roughly flat."""
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
def sequence_logprob(model, tokenizer, prompt: str, response: str) -> float:
    """Sum log p(response | prompt), used for the held-out preference check."""
    p_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                      max_length=MAX_PROMPT_LEN).input_ids
    full = tokenizer(prompt + response, return_tensors="pt", truncation=True,
                     max_length=MAX_SEQ_LEN).input_ids.to(model.device)
    n_prompt = p_ids.shape[1]
    if full.shape[1] <= n_prompt + 1:
        return float("-inf")
    logits = model(full).logits[0, :-1].float().log_softmax(-1)
    targets = full[0, 1:]
    tok_lp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp[n_prompt - 1:].sum().item()


@torch.no_grad()
def preference_accuracy(model, tokenizer, pairs: list[dict], limit: int = 60) -> float:
    """GUARD 4 (overfitting). Held-out fraction where chosen beats rejected.

    This is measured on pairs the model never trained on. Train-set accuracy
    saturating while this stays flat is the definition of memorising the
    annotators rather than learning what they valued.
    """
    model.eval()
    wins = total = 0
    for p in pairs[:limit]:
        prompt = ALPACA_NO_INPUT.format(instruction=p["prompt"], output="")
        lp_c = sequence_logprob(model, tokenizer, prompt, p["chosen"])
        lp_r = sequence_logprob(model, tokenizer, prompt, p["rejected"])
        if math.isinf(lp_c) or math.isinf(lp_r):
            continue
        wins += int(lp_c > lp_r)
        total += 1
    return wins / max(1, total)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 90) -> str:
    FastLanguageModel.for_inference(model)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens,
        # Greedy decoding on a 135M model manufactures repeat loops that read
        # as degeneration but are a decoding artefact. Sample instead, and
        # penalise repeats, so the length/repetition guards measure the model
        # rather than the decoder.
        do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeats. 0 = no repetition, ->1 = looping."""
    toks = text.split()
    if len(toks) <= n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def response_stats(model, tokenizer, label: str) -> dict:
    """GUARDS 1 & 3, measured. Length and repetition across the empathy probes.

    Printed at every stage boundary so length blowup and degeneration are
    visible as numbers, not just as a vibe from reading the samples.
    """
    lengths, reps, eos_hits, samples = [], [], 0, {}
    for probe in EMPATHY_PROBES:
        prompt = ALPACA_NO_INPUT.format(instruction=probe, output="")
        text = generate(model, tokenizer, prompt, 120)
        samples[probe] = text
        lengths.append(len(tokenizer(text).input_ids))
        reps.append(repetition_rate(text))
        # A response shorter than the cap means generation stopped on its own.
        eos_hits += int(len(tokenizer(text).input_ids) < 118)
    stats = {
        "mean_tokens": sum(lengths) / max(1, len(lengths)),
        "max_tokens": max(lengths) if lengths else 0,
        "repetition": sum(reps) / max(1, len(reps)),
        "eos_rate": eos_hits / max(1, len(EMPATHY_PROBES)),
        "samples": samples,
    }
    print(f"[stats {label}] mean_tokens={stats['mean_tokens']:.1f} "
          f"max={stats['max_tokens']} repetition={stats['repetition']:.3f} "
          f"eos_rate={stats['eos_rate']:.0%}")
    return stats


def probe(model, tokenizer, label: str) -> dict:
    banner(f"PROBES -- {label}")
    result = {}
    print("\n-- empathy prompts (does it acknowledge the feeling?)")
    for p in EMPATHY_PROBES:
        prompt = ALPACA_NO_INPUT.format(instruction=p, output="")
        text = generate(model, tokenizer, prompt, 90)
        result[f"empathy::{p}"] = text
        print(f"  {p!r}\n    -> {text}\n")

    print("-- general english (drift check)")
    text = generate(model, tokenizer, GENERAL_PROBE, 30)
    result[f"general::{GENERAL_PROBE}"] = text
    print(f"  {GENERAL_PROBE!r}\n    -> {text}")
    return result


def compare_stages(report: dict) -> None:
    """The before/after table. This is the point of the whole script."""
    banner("BEFORE / AFTER")
    rows = [("base", "stats_base", "pref_acc_before", "ppl_before"),
            ("after SFT", "stats_after_sft", "pref_acc_after_sft", "ppl_after_sft"),
            ("after DPO", "stats_after_dpo", "pref_acc_after_dpo", "ppl_after_dpo")]
    print(f"  {'stage':<10} {'pref acc':>9} {'mean tok':>9} {'repetition':>11} "
          f"{'eos':>6} {'gen ppl':>9}")
    for label, skey, akey, pkey in rows:
        st = report.get(skey) or {}
        print(f"  {label:<10} {report.get(akey, float('nan')):>9.3f} "
              f"{st.get('mean_tokens', float('nan')):>9.1f} "
              f"{st.get('repetition', float('nan')):>11.3f} "
              f"{st.get('eos_rate', float('nan')):>6.0%} "
              f"{report.get(pkey, float('nan')):>9.2f}")

    print("\n  How to read this:")
    print("    pref acc rising, length and repetition flat  -> alignment worked")
    print("    pref acc rising, mean tok climbing sharply   -> length exploitation")
    print("    pref acc rising, repetition climbing         -> degeneration")
    print("    gen ppl climbing sharply                     -> drifted off the reference")

    base = (report.get("stats_base") or {}).get("mean_tokens")
    final = (report.get("stats_after_dpo") or {}).get("mean_tokens")
    if base and final and final > 1.5 * base:
        print(f"\n  WARNING: mean response length grew {final / base:.1f}x "
              f"({base:.0f} -> {final:.0f} tokens). That is the length-exploitation "
              f"signature; raise --beta or lower --max-length-ratio and re-run.")


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SFT then DPO")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--sft-steps", type=int, default=300)
    ap.add_argument("--dpo-steps", type=int, default=200)
    ap.add_argument("--sft-rows", type=int, default=5000)
    ap.add_argument("--pref-rows", type=int, default=4000)
    ap.add_argument("--format-mix", type=float, default=0.25,
                    help="fraction of SFT rows rendered without an ### Input block")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO KL leash; higher keeps the policy nearer the SFT model")
    ap.add_argument("--max-length-ratio", type=float, default=1.5,
                    help="drop preference pairs whose chosen response is more "
                         "than this many times longer than the rejected one")
    ap.add_argument("--sft-lr", type=float, default=2e-4)
    ap.add_argument("--dpo-lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    # 4-bit is the default, as in the class notebooks: it is what fits a T4
    # comfortably and what these hyperparameters were tuned against.
    ap.add_argument("--allow-bad-merge", action="store_true",
                    help="continue past a failed stage-1 merge check instead "
                         "of aborting; the DPO result will be meaningless")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                    default=True, help="load in fp16/bf16 instead of 4-bit")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check wiring")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--push-to-hub", nargs="?", const=HF_USER, metavar="USER[/REPO]",
                    help=f"publish the final merged model; bare flag uses "
                         f"{HF_USER}/{RUN_NAME}")
    ap.add_argument("--push-adapters", action="store_true",
                    help="also publish the stage-1 and stage-2 LoRA adapters")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--hf-private", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.sft_steps = args.dpo_steps = 30
        args.sft_rows, args.pref_rows = 400, 400

    random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"order": "SFT -> DPO", "base": args.base,
                    "beta": args.beta, "max_length_ratio": args.max_length_ratio}

    # ---------------------------------------------------------------- stage 0
    banner("STAGE 0 -- baseline")
    model, tokenizer = load_model(args.base, args.load_in_4bit)
    eos = tokenizer.eos_token

    sft_train, sft_eval = build_sft_dataset(eos, args.sft_rows, args.format_mix)
    pref_train, pref_eval, pref_eval_raw = build_pref_dataset(
        args.pref_rows, args.max_length_ratio)
    general_eval = load_general_chunks(80)

    report["ppl_before"] = perplexity(model, tokenizer, general_eval)
    report["pref_acc_before"] = preference_accuracy(model, tokenizer, pref_eval_raw)
    print(f"[base] general ppl={report['ppl_before']:.2f} "
          f"pref_acc={report['pref_acc_before']:.3f}")
    report["probes_base"] = probe(model, tokenizer, "base model")
    report["stats_base"] = response_stats(model, tokenizer, "base")

    # ---------------------------------------------------------------- stage 1
    banner("STAGE 1 -- SFT (empathetic dialogue, response-only loss)")
    model = attach_lora(model, rank=16)
    model.print_trainable_parameters()
    enable_training(model)

    trainer = make_sft_trainer(
        model, tokenizer, sft_train, sft_eval,
        out_dir=OUT / "sft_ckpt", lr=args.sft_lr,
        max_steps=args.sft_steps, epochs=2,
        batch=args.batch, accum=args.accum,
        warmup=min(50, max(5, args.sft_steps // 10)),
    )
    trainer = mask_prompt_tokens(trainer)
    show_masking(trainer, tokenizer)
    report["sft_train_loss"] = trainer.train().training_loss
    report["fit_sft"] = warn_if_overfit(trainer, "sft")

    report["ppl_after_sft"] = perplexity(model, tokenizer, general_eval)
    report["pref_acc_after_sft"] = preference_accuracy(model, tokenizer, pref_eval_raw)
    report["probes_after_sft"] = probe(model, tokenizer, "after SFT")
    report["stats_after_sft"] = response_stats(model, tokenizer, "after SFT")

    model.save_pretrained(str(SFT_ADAPTER))
    tokenizer.save_pretrained(str(SFT_ADAPTER))
    banner("merging SFT adapter -- this becomes the DPO reference model")
    save_merged(model, tokenizer, SFT_MERGED)
    # DPO trains on THIS file and anchors its KL leash to it.
    report["merge_check_stage1"] = verify_merged(
        SFT_MERGED, general_eval, report["ppl_after_sft"], "sft-intermediate",
        fatal=not args.allow_bad_merge)

    if args.push_to_hub and args.push_adapters:
        repo = resolve_repo(args.push_to_hub, "-stage1-sft-lora")
        banner(f"pushing LoRA adapter -> https://huggingface.co/{repo}")
        model.push_to_hub(repo, token=args.hf_token, private=args.hf_private)
        tokenizer.push_to_hub(repo, token=args.hf_token, private=args.hf_private)

    del model, trainer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- stage 2
    banner("STAGE 2 -- DPO (preference pairs, KL-anchored to the SFT model)")
    model, tokenizer = load_model(str(SFT_MERGED), args.load_in_4bit)
    model = attach_lora(model, rank=16)
    model.print_trainable_parameters()
    enable_training(model)

    trainer = make_dpo_trainer(
        model, tokenizer, pref_train, pref_eval,
        out_dir=OUT / "dpo_ckpt", lr=args.dpo_lr, beta=args.beta,
        max_steps=args.dpo_steps, epochs=1,
        batch=args.batch, accum=args.accum,
        warmup=min(30, max(5, args.dpo_steps // 10)),
    )
    report["dpo_train_loss"] = trainer.train().training_loss
    report["fit_dpo"] = warn_if_overfit(trainer, "dpo")

    # Final DPO log line carries rewards/accuracies and rewards/margins; keep
    # them in the report so the run can be judged without the console scrollback.
    for entry in reversed(trainer.state.log_history):
        keep = {k: v for k, v in entry.items() if k.startswith("rewards/")}
        if keep:
            report["dpo_rewards"] = keep
            print(f"[dpo] final reward metrics: {keep}")
            break

    report["ppl_after_dpo"] = perplexity(model, tokenizer, general_eval)
    report["pref_acc_after_dpo"] = preference_accuracy(model, tokenizer, pref_eval_raw)
    report["probes_after_dpo"] = probe(model, tokenizer, "after SFT + DPO (final)")
    report["stats_after_dpo"] = response_stats(model, tokenizer, "after DPO")

    # ---------------------------------------------------------------- export
    model.save_pretrained(str(DPO_ADAPTER))
    tokenizer.save_pretrained(str(DPO_ADAPTER))
    banner("merging DPO adapter -> deployable single-checkpoint model")
    save_merged(model, tokenizer, FINAL_MERGED)
    report["merge_check"] = verify_merged(
        FINAL_MERGED, general_eval, report["ppl_after_dpo"], "final")

    if args.push_to_hub:
        repo = resolve_repo(args.push_to_hub)
        push_merged(model, tokenizer, repo, args.hf_token, args.hf_private)
        push_card(repo, model_card(repo, report, args), args.hf_token)
        report["hub_model"] = f"https://huggingface.co/{repo}"
        if args.push_adapters:
            arepo = resolve_repo(args.push_to_hub, "-stage2-dpo-lora")
            banner(f"pushing LoRA adapter -> https://huggingface.co/{arepo}")
            model.push_to_hub(arepo, token=args.hf_token, private=args.hf_private)
            tokenizer.push_to_hub(arepo, token=args.hf_token, private=args.hf_private)

    if not args.keep_intermediate:
        shutil.rmtree(SFT_MERGED, ignore_errors=True)
        print(f"[cleanup] removed {SFT_MERGED} (pass --keep-intermediate to retain)")

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    compare_stages(report)
    print(f"\n  final model : {FINAL_MERGED}")
    print(f"  sft adapter : {SFT_ADAPTER}")
    print(f"  dpo adapter : {DPO_ADAPTER}")
    print(f"  report      : {OUT / 'report.json'}")
    if report.get("hub_model"):
        print(f"  on the hub  : {report['hub_model']}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
