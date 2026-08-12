"""
sft_then_grpo.py
================

RLVR (reinforcement learning from verifiable rewards) with GRPO, arranged so
the algorithm has something to work with:

    base model  ->  Stage 1: SFT cold-start  ->  merge  ->  Stage 2: GRPO  ->  final

Why the cold start is not optional
----------------------------------
GRPO computes each sample's advantage *within a group* of `num_generations`
completions of the same prompt:

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

Read what that does when every completion in the group scores the same. The
numerator is zero for all of them. The advantage is zero, the policy-gradient
term is zero, and the update is zero -- no matter how long you train. The loss
sitting at ~0 with a flat reward curve is not a bug, it is the algorithm
correctly reporting that it has been handed no signal.

That is exactly what happens when you point GRPO at a base model that never
emits the target format. Every completion earns 0 on correctness, 0 on strict
format, 0 on soft format, 0 on the graduated XML count -- identical rewards,
zero variance, zero gradient. The graduated rewards exist to break ties, but
they cannot break a tie at zero: a model that never writes "<reasoning>" scores
0.0 on tag-counting every single time.

So this script does the thing the standard recipe does and the demo notebooks
skip: **SFT the format in first**, from GSM8K's own worked solutions, then let
GRPO amplify it. DeepSeek call this the cold start; R1-Zero (pure RL, no SFT)
worked only because a 671B base model already produced the seed behaviour by
chance.

RL polishes. RL amplifies. RL does not teach.

The precondition gate
---------------------
Before spending a single GRPO step, `reward_audit` samples completions and
measures the one quantity that decides whether GRPO can learn at all: the mean
*within-group* reward standard deviation. If it is ~0, training is a no-op and
the script says so and stops, rather than burning an hour to produce a flat
line. `--force-grpo` overrides, which is the right flag if reproducing the
failure is the point.

Model size
----------
SmolLM-135M is below the practical floor for this, and the audit will usually
say so even after the cold start. It is the default because it is what fits a
free T4 and it makes the failure legible. For GRPO that actually moves:

    python sft_then_grpo.py --base unsloth/Qwen2.5-1.5B-Instruct

Requirements
------------
    pip install unsloth trl peft transformers datasets accelerate bitsandbytes

Needs a CUDA GPU. Unsloth does not support Apple MPS.

Usage
-----
    python sft_then_grpo.py                    # cold start + GRPO, ~35 min on a T4
    python sft_then_grpo.py --smoke            # tiny run, checks the wiring
    python sft_then_grpo.py --skip-sft         # the failure mode, deliberately
    python sft_then_grpo.py --force-grpo       # train even with zero reward variance

Reward functions are Will Brown's GSM8K stack (github.com/willccbb/verifiers),
as used in Unsloth's GRPO recipe.
"""

from __future__ import annotations

# Unsloth must be imported before trl so its patches land.
from unsloth import FastLanguageModel  # noqa: I001  isort:skip

try:  # GRPO needs an explicit patch on some unsloth versions
    from unsloth import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    pass

import argparse
import inspect
import json
import os
import random
import re
import shutil
import statistics
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

BASE_MODEL = "HuggingFaceTB/SmolLM-135M"
MAX_SEQ_LEN = 1024
MAX_PROMPT_LEN = 256
MAX_COMPLETION_LEN = 200
SEED = 3407

# The task decides whether GRPO can learn at all, more than any hyperparameter.
#
# GRPO needs the completions in a group to DISAGREE. With a free-form numeric
# answer a small model is right essentially never, so all G completions score
# zero, the group has no variance and the gradient is exactly zero. With 4-way
# multiple choice, chance alone is 25%: groups routinely come back split, which
# is the variance the advantage is normalised by.
#
# That is why arc is the default here and gsm8k is the honest hard mode.
TASKS = {
    "arc": {
        "repo": ("allenai/ai2_arc", "ARC-Easy"),
        "answer_kind": "letter",
        "blurb": "ARC-Easy, 4-way multiple choice (~25% by chance)",
    },
    "gsm8k": {
        "repo": ("openai/gsm8k", "main"),
        "answer_kind": "digit",
        "blurb": "GSM8K grade-school word problems (~0% by chance below ~1.5B)",
    },
}

# The trl reward-function signature is fixed, so the shape check reads this
# module-level value rather than taking it as an argument. main() sets it once.
ANSWER_KIND = "letter"

HF_USER = "sidhusarkar"
RUN_NAME = "smollm-135m-sft-then-grpo"

_HERE = Path(__file__).resolve().parent
OUT = _HERE.parent / "reports" / "sft_then_grpo"   # <repo>/RLVR/reports/...
SFT_ADAPTER = OUT / "01_sft_adapter"
SFT_MERGED = OUT / "02_sft_merged"
GRPO_ADAPTER = OUT / "03_grpo_adapter"
FINAL_MERGED = OUT / "04_final_merged"

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

XML_COT_FORMAT = """<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>"""


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


def bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _keep_supported(cls, kwargs: dict) -> dict:
    """Drop kwargs this trl version doesn't accept.

    GRPOConfig gains and loses arguments almost every release; filtering by
    signature keeps this runnable instead of pinning one trl.
    """
    allowed = set(inspect.signature(cls.__init__).parameters)
    if "kwargs" in allowed:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def show_examples(label: str, texts: list[str], n: int = 2, width: int = 700) -> None:
    """Print real training strings, exactly as the trainer will see them."""
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
    """`for_training` with the probe-then-train crash guarded.

    `for_inference()` stamps `_flag_for_generation` on the modules it walks and
    `for_training()` deletes it. PeftModel delegates attribute lookup to the
    wrapped model, so `hasattr` reports the flag as present on the wrapper while
    it actually lives on the inner module -- and `del` only removes entries from
    the object's own __dict__. Write into __dict__ directly.
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
# Reward functions -- Will Brown's GSM8K stack
# ----------------------------------------------------------------------------

def extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def extract_hash_answer(text: str) -> str | None:
    return text.split("####")[1].strip() if "####" in text else None


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """+2.0 when the extracted <answer> matches ground truth."""
    got = [extract_xml_answer(c[0]["content"]) for c in completions]
    return [2.0 if g == a else 0.0 for g, a in zip(got, answer)]


def shape_reward(completions, **kwargs) -> list[float]:
    """+0.5 when the answer has the right SHAPE, regardless of correctness.

    A digit for arithmetic, a single choice letter for multiple choice. This is
    a cheap partial credit that separates "answered the question badly" from
    "did not answer the question", which is another tie the group can be broken
    on before correctness starts firing.
    """
    got = [extract_xml_answer(c[0]["content"]) for c in completions]
    if ANSWER_KIND == "letter":
        return [0.5 if g.strip().upper() in {"A", "B", "C", "D", "E"} else 0.0
                for g in got]
    return [0.5 if g.isdigit() else 0.0 for g in got]


def strict_format_reward(completions, **kwargs) -> list[float]:
    """+0.5 for the exact XML layout, newlines included."""
    pat = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    return [0.5 if re.match(pat, c[0]["content"], re.DOTALL) else 0.0
            for c in completions]


def soft_format_reward(completions, **kwargs) -> list[float]:
    """+0.5 for the XML layout, lenient about whitespace."""
    pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pat, c[0]["content"], re.DOTALL) else 0.0
            for c in completions]


def _count_xml(text: str) -> float:
    """Graduated credit per tag -- the tie-breaker that keeps gradients alive.

    Correctness is all-or-nothing, so early in training every completion scores
    the same 0.0 and the group has no variance. Per-tag credit gives partial
    scores that differ between completions, which is what produces a non-zero
    advantage. It only helps once the model emits *some* tags: at zero tags this
    scores 0.0 for everyone and ties right back up.
    """
    c = 0.0
    if text.count("<reasoning>\n") == 1:
        c += 0.125
    if text.count("\n</reasoning>\n") == 1:
        c += 0.125
    if text.count("\n<answer>\n") == 1:
        c += 0.125
    if text.count("\n</answer>") == 1:
        c += 0.125
        c -= len(text.split("\n</answer>")[-1]) * 0.001  # penalise trailing text
    return c


def xmlcount_reward(completions, **kwargs) -> list[float]:
    return [_count_xml(c[0]["content"]) for c in completions]


REWARD_FUNCS = [xmlcount_reward, soft_format_reward, strict_format_reward,
                shape_reward, correctness_reward]
REWARD_NAMES = [f.__name__ for f in REWARD_FUNCS]


def score_all(text: str, gold: str) -> dict[str, float]:
    """Every reward for one completion, as the trainer would compute them."""
    c = [[{"role": "assistant", "content": text}]]
    return {
        "xmlcount_reward": xmlcount_reward(c)[0],
        "soft_format_reward": soft_format_reward(c)[0],
        "strict_format_reward": strict_format_reward(c)[0],
        "shape_reward": shape_reward(c)[0],
        "correctness_reward": correctness_reward(None, c, [gold])[0],
    }


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def clean_solution(answer: str) -> str:
    """GSM8K reasoning with the calculator annotations and #### line removed."""
    body = answer.split("####")[0]
    body = re.sub(r"<<[^>]*>>", "", body)         # strip <<48/2=24>>
    return re.sub(r"\n{2,}", "\n", body).strip()


LETTERS = "ABCDE"


def load_task_rows(task: str, n_rows: int) -> list[dict]:
    """Normalise whichever task to {question, gold, reasoning}.

    `reasoning` is what the cold start imitates. GSM8K ships human worked
    solutions; ARC does not, so the reasoning is a single sentence naming the
    chosen option. That is enough -- the cold start is teaching the *format*,
    and the reasoning content is what GRPO is then supposed to improve.
    """
    repo, config = TASKS[task]["repo"]
    ds = load_dataset(repo, config, split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_rows, len(ds))))

    rows = []
    for r in ds:
        if task == "gsm8k":
            gold = extract_hash_answer(r["answer"])
            if gold is None:
                continue
            rows.append({"question": r["question"], "gold": gold,
                         "reasoning": clean_solution(r["answer"])})
        else:
            texts = r["choices"]["text"]
            labels = [str(l) for l in r["choices"]["label"]]
            gold = str(r["answerKey"]).strip()
            # A few ARC rows label choices 1-4 rather than A-D; normalise both
            # the options and the gold key so the reward can compare letters.
            if gold not in LETTERS:
                if gold not in labels:
                    continue
                gold = LETTERS[labels.index(gold)]
            elif gold not in labels:
                continue
            elif labels[0] not in LETTERS:
                gold = LETTERS[labels.index(gold)]
            opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))
            if gold not in LETTERS[:len(texts)]:
                continue
            rows.append({
                "question": f"{r['question']}\n\n{opts}",
                "gold": gold,
                "reasoning": f"The correct option is {gold}: "
                             f"{texts[LETTERS.index(gold)]}",
            })
    return rows


def build_grpo_dataset(task: str, n_rows: int) -> Dataset:
    """Prompts in chat form plus the gold answer GRPO's reward will check."""
    rows = [{"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": r["question"]}],
             "answer": r["gold"]}
            for r in load_task_rows(task, n_rows)]
    print(f"[grpo-data] {task}: {len(rows)} prompts")
    return Dataset.from_list(rows)


def build_sft_dataset(tokenizer, task: str, n_rows: int):
    """GSM8K worked solutions, rewritten into the exact target format.

    This is the cold start. The model is shown the shape it will later be
    rewarded for producing, using reasoning humans already wrote, so GRPO has a
    behaviour to amplify rather than one to invent.
    """
    texts = []
    for r in load_task_rows(task, n_rows):
        target = XML_COT_FORMAT.format(reasoning=r["reasoning"], answer=r["gold"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True)
        texts.append(prompt + target + tokenizer.eos_token)

    random.Random(SEED).shuffle(texts)
    split_at = max(1, int(len(texts) * 0.95))
    print(f"[sft-data] train={split_at}  eval={len(texts) - split_at}")
    show_examples("sft-data / cold start", texts[:split_at], 1, width=900)
    return (Dataset.from_dict({"text": texts[:split_at]}),
            Dataset.from_dict({"text": texts[split_at:]}))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(path: str, load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=load_in_4bit,
        fast_inference=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # Base models ship without one; GRPO and the cold start both need to
        # render the same prompt, so define it once here.
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] + ':\n' + m['content'] + '\n\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ 'assistant:\n' }}{% endif %}"
        )
    return model, tokenizer


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def attach_lora(model, *, rank: int):
    return FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=rank, lora_dropout=0,
        target_modules=LORA_TARGETS, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED, use_rslora=False,
    )


# ----------------------------------------------------------------------------
# The precondition: can GRPO learn anything from this model at all?
# ----------------------------------------------------------------------------

@torch.no_grad()
def sample_group(model, tokenizer, prompt_msgs, n: int, temperature: float) -> list[str]:
    """n independent completions of one prompt -- exactly what GRPO scores."""
    FastLanguageModel.for_inference(model)
    text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False,
                                         add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_PROMPT_LEN).to(model.device)
    out = model.generate(
        **ids, max_new_tokens=MAX_COMPLETION_LEN,
        do_sample=True, temperature=temperature, top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    cut = ids["input_ids"].shape[1]
    return [tokenizer.decode(o[cut:], skip_special_tokens=True) for o in out]


def reward_audit(model, tokenizer, dataset, *, n_prompts: int, n_gens: int,
                 temperature: float, label: str) -> dict:
    """Measure whether GRPO has any gradient available, before spending steps.

    The number that matters is `mean_group_std`: the standard deviation of total
    reward *within* each group of completions, averaged over prompts. GRPO
    normalises advantages by exactly that quantity, so when it is zero every
    advantage is zero and every update is zero. A flat loss curve and a flat
    reward curve are the symptom; this is the cause, and it is measurable in a
    couple of minutes instead of an hour.
    """
    banner(f"REWARD AUDIT -- {label}")
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(dataset)), min(n_prompts, len(dataset)))

    per_func = {k: [] for k in REWARD_NAMES}
    totals, group_stds, samples = [], [], []
    for j, i in enumerate(idxs):
        row = dataset[i]
        gens = sample_group(model, tokenizer, row["prompt"], n_gens, temperature)
        group_totals = []
        for g in gens:
            s = score_all(g, row["answer"])
            for k, v in s.items():
                per_func[k].append(v)
            group_totals.append(sum(s.values()))
        totals.extend(group_totals)
        group_stds.append(statistics.pstdev(group_totals) if len(group_totals) > 1 else 0.0)
        if j == 0:
            samples = gens[:2]

    stats = {
        "mean_total_reward": sum(totals) / max(1, len(totals)),
        "max_total_reward": max(totals) if totals else 0.0,
        "mean_group_std": sum(group_stds) / max(1, len(group_stds)),
        "nonzero_rate": sum(1 for t in totals if t > 0) / max(1, len(totals)),
        "fired": {k: sum(1 for v in vals if v > 0) / max(1, len(vals))
                  for k, vals in per_func.items()},
        "mean_by_func": {k: sum(vals) / max(1, len(vals))
                         for k, vals in per_func.items()},
    }

    print(f"  prompts={len(idxs)}  generations each={n_gens}  temperature={temperature}")
    print(f"  mean total reward : {stats['mean_total_reward']:.4f}  "
          f"(max seen {stats['max_total_reward']:.3f})")
    print(f"  completions scoring > 0 : {stats['nonzero_rate']:.0%}")
    print(f"  mean WITHIN-GROUP std   : {stats['mean_group_std']:.4f}   "
          f"<-- GRPO divides advantages by this")
    print("  fired in:")
    for k in REWARD_NAMES:
        print(f"    {k:<24} {stats['fired'][k]:6.1%}   mean {stats['mean_by_func'][k]:+.4f}")
    for i, s in enumerate(samples):
        print(f"\n  sample #{i}:\n  | " + s[:400].replace("\n", "\n  | "))
    return stats


def gate_on_variance(stats: dict, force: bool) -> None:
    """Refuse to run GRPO with no reward variance, unless told otherwise."""
    std = stats["mean_group_std"]
    if std > 0.01:
        print(f"\n[gate] within-group reward std {std:.4f} -- GRPO has signal to work with")
        return
    msg = (f"\n[gate] within-group reward std is {std:.4f}.\n"
           f"       Every completion in a group scores the same, so every advantage\n"
           f"       is (r - mean)/std = 0 and every GRPO update is exactly zero.\n"
           f"       Training would produce a flat loss and a flat reward curve for\n"
           f"       as long as you let it run.\n\n"
           f"       Fixes, in order of effect:\n"
           f"         1. A bigger base model. --base unsloth/Qwen2.5-1.5B-Instruct\n"
           f"            is the usual floor; 135M rarely clears it even after SFT.\n"
           f"         2. More SFT cold start: --sft-steps 600 --sft-rows 8000.\n"
           f"         3. More exploration: --temperature 1.2 --num-generations 8.\n\n"
           f"       --force-grpo runs anyway, which is the right call if you want\n"
           f"       the flat curve as evidence.")
    if force:
        print(msg + "\n\n[gate] --force-grpo set; continuing into a no-op.")
        return
    raise SystemExit(msg)


# ----------------------------------------------------------------------------
# Trainers
# ----------------------------------------------------------------------------

def make_sft_trainer(model, tokenizer, train_ds, eval_ds, *, out_dir: Path,
                     lr: float, max_steps: int, batch: int, accum: int):
    """Cold start: ordinary next-token cross-entropy on the target format.

    LOSS
        L = -(1/|R|) * sum_{t in R} log p(y_t | y_<t, x)

    Loss on the whole rendered example rather than a masked response span --
    the point here is to make the *format* fluent, and the prompt half is a
    fixed template the model may as well learn to expect.
    """
    from trl import SFTConfig, SFTTrainer
    every = max(10, (max_steps or 200) // 5)
    cfg = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        max_steps=max_steps,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        seed=SEED,
        report_to="none",
        fp16=not bf16_ok(),
        bf16=bf16_ok(),
        eval_strategy="steps",
        eval_steps=every,
        per_device_eval_batch_size=batch,
        save_strategy="steps",
        save_steps=every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_num_proc=2,
    )
    args = SFTConfig(**_keep_supported(SFTConfig, cfg))
    kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
              dataset_text_field="text", max_seq_length=MAX_SEQ_LEN, packing=False)
    sig = set(inspect.signature(SFTTrainer.__init__).parameters)
    kw["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer
    return SFTTrainer(**_keep_supported(SFTTrainer, kw))


def make_grpo_trainer(model, tokenizer, train_ds, *, out_dir: Path, lr: float,
                      max_steps: int, num_generations: int, temperature: float,
                      batch: int, accum: int, beta: float):
    """Group Relative Policy Optimisation.

    LOSS
        For each prompt x, sample G completions y_1..y_G from the current policy
        and score them with the verifiable reward functions. Then

            A_i = (r_i - mean(r)) / std(r)

            L = -(1/G) * sum_i [ min( w_i * A_i,
                                      clip(w_i, 1-eps, 1+eps) * A_i ) ]
                + beta * KL(pi_theta || pi_ref)

        where w_i = pi_theta(y_i|x) / pi_old(y_i|x) is the importance ratio.

    Three things follow from that formula, and all three matter in practice:

    1. **No value network.** PPO learns a critic to estimate the baseline; GRPO
       uses the group mean instead. That is the whole saving, and it is why GRPO
       fits on a T4 where PPO would not.

    2. **The baseline is the group, so the group must disagree.** If all G
       rewards are equal then A_i = 0 for every i and the gradient vanishes
       identically. This is not a soft failure -- there is no small signal to
       accumulate over many steps. It is zero. `reward_audit` measures this
       before training rather than after.

    3. **`beta` is the leash to the reference policy.** Verifiable rewards are
       cheap to game (emit the tags, emit a digit, say nothing useful), and the
       KL term is what stops the policy walking off to a degenerate optimum that
       scores well and reads badly.
    """
    from trl import GRPOConfig, GRPOTrainer
    cfg = dict(
        output_dir=str(out_dir),
        learning_rate=lr,
        beta=beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        # G. Bigger groups give a better baseline estimate and more chance that
        # two completions differ, at linear cost in generation time.
        num_generations=num_generations,
        temperature=temperature,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        max_grad_norm=0.1,
        report_to="none",
        seed=SEED,
        fp16=not bf16_ok(),
        bf16=bf16_ok(),
        use_vllm=False,
    )
    args = GRPOConfig(**_keep_supported(GRPOConfig, cfg))
    kw = dict(model=model, args=args, train_dataset=train_ds,
              reward_funcs=REWARD_FUNCS, processing_class=tokenizer)
    return GRPOTrainer(**_keep_supported(GRPOTrainer, kw))


def reward_trajectory(trainer) -> dict:
    """Pull the reward curve out of the trainer log.

    `reward_std` is the diagnostic: if it never rises above ~0, the run was a
    no-op regardless of what the loss did.
    """
    rewards = [e["reward"] for e in trainer.state.log_history if "reward" in e]
    stds = [e["reward_std"] for e in trainer.state.log_history if "reward_std" in e]
    if not rewards:
        return {}
    half = max(1, len(rewards) // 2)
    out = {
        "reward_first_half": sum(rewards[:half]) / half,
        "reward_last_half": sum(rewards[half:]) / max(1, len(rewards) - half),
        "reward_std_mean": (sum(stds) / len(stds)) if stds else 0.0,
        "reward_std_max": max(stds) if stds else 0.0,
    }
    print(f"[grpo] reward {out['reward_first_half']:.4f} -> {out['reward_last_half']:.4f}"
          f"   reward_std mean={out['reward_std_mean']:.4f} max={out['reward_std_max']:.4f}")
    if out["reward_std_max"] < 0.01:
        print("  WARNING: reward_std never left zero. Every group agreed with "
              "itself, so every update was zero -- this run taught the model "
              "nothing. See the audit gate for what to change.")
    return out


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SFT cold start, then GRPO on GSM8K")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--task", default="arc", choices=sorted(TASKS),
                    help="; ".join(f"{k}: {v['blurb']}" for k, v in TASKS.items()))
    ap.add_argument("--sft-steps", type=int, default=300)
    ap.add_argument("--grpo-steps", type=int, default=150)
    ap.add_argument("--sft-rows", type=int, default=4000)
    ap.add_argument("--grpo-rows", type=int, default=2000)
    ap.add_argument("--num-generations", type=int, default=4,
                    help="G: completions per prompt; the group the baseline comes from")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="higher explores more, which is what creates reward variance")
    ap.add_argument("--beta", type=float, default=0.04, help="KL leash to the reference")
    ap.add_argument("--sft-lr", type=float, default=2e-4)
    ap.add_argument("--grpo-lr", type=float, default=5e-6)
    ap.add_argument("--audit-prompts", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--skip-sft", action="store_true",
                    help="GRPO straight off the base model -- the documented failure")
    ap.add_argument("--resume-from-sft", action="store_true",
                    help="skip cold-start training and reuse the adapter already "
                         "in reports/, e.g. after a merge failure")
    ap.add_argument("--force-grpo", action="store_true",
                    help="train even when the audit finds zero reward variance")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--push-to-hub", nargs="?", const=HF_USER, metavar="USER[/REPO]")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if args.smoke:
        args.sft_steps, args.grpo_steps = 30, 10
        args.sft_rows, args.grpo_rows, args.audit_prompts = 400, 200, 3

    random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    global ANSWER_KIND
    ANSWER_KIND = TASKS[args.task]["answer_kind"]
    print(f"[task] {args.task} -- {TASKS[args.task]['blurb']}")

    report: dict = {"order": "SFT -> GRPO", "base": args.base, "task": args.task,
                    "num_generations": args.num_generations,
                    "temperature": args.temperature, "beta": args.beta,
                    "skip_sft": args.skip_sft}

    # ---------------------------------------------------------------- stage 0
    banner("STAGE 0 -- baseline")
    model, tokenizer = load_model(args.base, args.load_in_4bit)
    grpo_ds = build_grpo_dataset(args.task, args.grpo_rows)

    report["audit_base"] = reward_audit(
        model, tokenizer, grpo_ds, n_prompts=args.audit_prompts,
        n_gens=args.num_generations, temperature=args.temperature,
        label="base model, before any training")

    # ---------------------------------------------------------------- stage 1
    if args.skip_sft:
        banner("STAGE 1 -- SKIPPED (--skip-sft)")
        print("Going straight to GRPO from the base model. This is the "
              "configuration the demo notebooks use and it is expected to "
              "produce a flat reward curve.")
        grpo_base = args.base
    elif args.resume_from_sft:
        # The cold start is the expensive half and it is saved before the merge
        # is attempted, so a merge failure should not cost it. Unsloth reads
        # adapter_config.json and pulls the base itself, which also means the
        # adapter can be re-merged at a precision it did not train at.
        banner("STAGE 1 -- REUSING the saved cold-start adapter")
        if not (SFT_ADAPTER / "adapter_config.json").exists():
            raise SystemExit(
                f"\n[abort] --resume-from-sft needs an adapter at {SFT_ADAPTER}\n"
                f"        and there is none. Run without the flag to train one.")
        print(f"loading {SFT_ADAPTER}")
        if not args.load_in_4bit:
            print("re-merging in fp16: the adapter trained against 4-bit weights, "
                  "so expect small numerical differences.")
        del model
        torch.cuda.empty_cache()
        model, tokenizer = load_model(str(SFT_ADAPTER), args.load_in_4bit)

        report["audit_after_sft"] = reward_audit(
            model, tokenizer, grpo_ds, n_prompts=args.audit_prompts,
            n_gens=args.num_generations, temperature=args.temperature,
            label="the reused cold-start adapter")

        banner("merging the cold-start adapter -- GRPO's starting policy and reference")
        save_merged(model, tokenizer, SFT_MERGED)
        grpo_base = str(SFT_MERGED)

        del model
        torch.cuda.empty_cache()
    else:
        banner("STAGE 1 -- SFT cold start (teach the format GRPO will reward)")
        sft_train, sft_eval = build_sft_dataset(tokenizer, args.task, args.sft_rows)
        model = attach_lora(model, rank=16)
        model.print_trainable_parameters()
        enable_training(model)

        trainer = make_sft_trainer(
            model, tokenizer, sft_train, sft_eval, out_dir=OUT / "sft_ckpt",
            lr=args.sft_lr, max_steps=args.sft_steps,
            batch=max(2, args.batch * 4), accum=args.accum)
        report["sft_train_loss"] = trainer.train().training_loss

        report["audit_after_sft"] = reward_audit(
            model, tokenizer, grpo_ds, n_prompts=args.audit_prompts,
            n_gens=args.num_generations, temperature=args.temperature,
            label="after the SFT cold start")

        model.save_pretrained(str(SFT_ADAPTER))
        tokenizer.save_pretrained(str(SFT_ADAPTER))
        banner("merging the cold-start adapter -- GRPO's starting policy and reference")
        save_merged(model, tokenizer, SFT_MERGED)
        grpo_base = str(SFT_MERGED)

        del model, trainer
        torch.cuda.empty_cache()

    # ------------------------------------------------- the precondition gate
    audit = report.get("audit_after_sft") or report["audit_base"]
    gate_on_variance(audit, args.force_grpo)

    # ---------------------------------------------------------------- stage 2
    banner("STAGE 2 -- GRPO (verifiable rewards, group-relative advantages)")
    model, tokenizer = load_model(grpo_base, args.load_in_4bit)
    model = attach_lora(model, rank=16)
    model.print_trainable_parameters()
    enable_training(model)

    trainer = make_grpo_trainer(
        model, tokenizer, grpo_ds, out_dir=OUT / "grpo_ckpt",
        lr=args.grpo_lr, max_steps=args.grpo_steps,
        num_generations=args.num_generations, temperature=args.temperature,
        batch=args.batch, accum=args.accum, beta=args.beta)
    report["grpo_train_loss"] = trainer.train().training_loss
    report["grpo_trajectory"] = reward_trajectory(trainer)

    report["audit_after_grpo"] = reward_audit(
        model, tokenizer, grpo_ds, n_prompts=args.audit_prompts,
        n_gens=args.num_generations, temperature=args.temperature,
        label="after GRPO")

    # ---------------------------------------------------------------- export
    model.save_pretrained(str(GRPO_ADAPTER))
    tokenizer.save_pretrained(str(GRPO_ADAPTER))
    banner("merging GRPO adapter -> deployable single checkpoint")
    save_merged(model, tokenizer, FINAL_MERGED)

    if args.push_to_hub:
        target = args.push_to_hub
        repo = target if "/" in target else f"{target}/{RUN_NAME}"
        banner(f"pushing -> https://huggingface.co/{repo}")
        model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit",
                                 token=args.hf_token)
        report["hub_model"] = f"https://huggingface.co/{repo}"

    if not args.keep_intermediate:
        shutil.rmtree(SFT_MERGED, ignore_errors=True)

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    # ---------------------------------------------------------------- summary
    banner("SUMMARY -- SFT cold start then GRPO")
    rows = [("base", report.get("audit_base")),
            ("after SFT", report.get("audit_after_sft")),
            ("after GRPO", report.get("audit_after_grpo"))]
    print(f"  {'stage':<12}{'mean reward':>12}{'>0 rate':>10}"
          f"{'group std':>12}{'correct':>10}{'strict fmt':>12}")
    for label, a in rows:
        if not a:
            continue
        print(f"  {label:<12}{a['mean_total_reward']:>12.4f}{a['nonzero_rate']:>10.0%}"
              f"{a['mean_group_std']:>12.4f}"
              f"{a['fired']['correctness_reward']:>10.1%}"
              f"{a['fired']['strict_format_reward']:>12.1%}")

    print("\n  How to read this:")
    print("    group std ~0 anywhere  -> GRPO had no gradient at that point")
    print("    strict fmt rising      -> the cold start took; GRPO has a behaviour to amplify")
    print("    correct rising         -> reasoning is actually improving, not just formatting")
    print(f"\n  report : {OUT / 'report.json'}")
    print(f"  model  : {FINAL_MERGED}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
