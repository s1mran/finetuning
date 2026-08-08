# finetuning

Does the order of **continued pre-training** and **supervised fine-tuning** matter?

Two scripts, identical in every respect except the order of the two stages, so
the ordering effect is isolated from everything else:

| script | order | output |
|---|---|---|
| [`cpt_then_sft.py`](cpt_then_sft.py) | CPT → SFT | `runs/cpt_then_sft/` |
| [`sft_then_cpt.py`](sft_then_cpt.py) | SFT → CPT | `runs/sft_then_cpt/` |

Same base model, same corpora, same seed (`3407`), same step counts, same
held-out eval slices. Run both, diff the two `report.json` files.

## The experiment

`unsloth/SmolLM-135M` is adapted to SEC 10-K financial text in two stages.

**CPT (continued pre-training)** — plain next-token prediction on raw 10-K
prose. Loss on *every* token. LoRA rank 32 targeting all linear layers **plus**
`embed_tokens` and `lm_head`, because the model is learning new domain
vocabulary ("EBITDA", "diluted EPS", "subordinated debentures"). 20% general
text (WikiText) is mixed in to fight catastrophic forgetting, and the embedding
layers get a 10× lower learning rate since they touch every token.

**SFT (supervised fine-tuning)** — Alpaca-formatted `(instruction, input,
output)` pairs. Loss masked to the response span only. LoRA rank 16 on the
linear layers alone; the vocabulary is already fine, only behaviour needs to
change. Packing is off, because response-only masking needs one example per
sequence. A quarter of the rows are rendered without the retrieved context so
the model doesn't learn "answers only exist when an `### Input` block is
present".

Between stages the adapter is merged into the base weights, so stage 2 starts
from a single clean checkpoint rather than stacking adapters.

### What each ordering predicts

**CPT → SFT is the recipe that holds up.** CPT teaches the model to *sound*
like the domain; SFT then teaches it to *answer*. Running SFT last leaves the
instruction-following behaviour as the freshest thing in the weights.

**SFT → CPT is the ablation.** It's the ordering people reach for intuitively
("teach it to answer, then teach it the domain") and it degrades. All-token
loss on raw prose has no reason to preserve response-only behaviour — the CPT
gradient pushes *every* position toward continuing a 10-K, including the
positions right after `### Response:`. Expect domain perplexity to land in
roughly the same place, paired with alpaca probes that ramble past the answer
and never emit EOS. `sft_then_cpt.py` records an explicit `eos_rate` per stage
to measure exactly that.

## Evaluation

Both scripts probe the model at every stage boundary — base, after stage 1,
after stage 2 — and write everything to `report.json`:

- **domain perplexity** on held-out 10-K text — did the domain adaptation work?
- **general perplexity** on held-out WikiText — how much did it forget?
- **raw completion probes** — does it sound like the domain?
- **alpaca probes** — does it answer, and does it *stop*?
- **bare-question probe** — did it overfit to the template?
- **general-English probe** — a readable catastrophic-forgetting check
- **`eos_rate`** (SFT→CPT only) — fraction of alpaca probes that terminate

## Setup

```bash
pip install unsloth trl peft transformers datasets accelerate bitsandbytes
```

Needs a **CUDA GPU** — a free Colab T4 is plenty for a 135M model. Unsloth does
not support Apple MPS, so these won't run on a Mac.

## Usage

```bash
python cpt_then_sft.py                      # full run, ~15 min on a T4
python cpt_then_sft.py --smoke              # 30 steps per stage, checks the wiring
python cpt_then_sft.py --cpt-steps 400 --sft-steps 400
```

`sft_then_cpt.py` takes the same flags. Useful ones:

| flag | default | what it does |
|---|---|---|
| `--base` | `unsloth/SmolLM-135M` | base model to adapt |
| `--smoke` | off | 30 steps/stage, tiny corpora — sanity check |
| `--general-ratio` | `0.20` | fraction of the CPT mix that is general text |
| `--format-mix` | `0.25` | fraction of SFT rows rendered without `### Input` |
| `--load-in-4bit` | off | QLoRA, for tighter VRAM |
| `--keep-intermediate` | off | retain the stage-1 merged checkpoint |

Run both with the same step counts before comparing, or the diff means nothing.

## Publishing to Hugging Face

```bash
huggingface-cli login          # token needs WRITE scope
python cpt_then_sft.py --push-to-hub
```

Pushes the merged 16-bit checkpoint plus a generated model card (base model,
tags, and the run's own before/after perplexity table) to
[huggingface.co/sidhusarkar](https://huggingface.co/sidhusarkar).

| flag | effect |
|---|---|
| `--push-to-hub` | bare flag → `sidhusarkar/<run-name>` |
| `--push-to-hub user/repo` | publish somewhere else |
| `--push-adapters` | also publish each stage's LoRA as its own repo |
| `--hf-private` | create the repos private |
| `--hf-token` | defaults to `$HF_TOKEN`, else the cached CLI login |

Card upload is best-effort — a failed README never discards a finished run.

## Artifacts

Each run writes to `runs/<name>/`:

```
01_<stage1>_adapter/   LoRA adapter from stage 1
02_<stage1>_merged/    merged checkpoint stage 2 starts from (deleted unless --keep-intermediate)
03_<stage2>_adapter/   LoRA adapter from stage 2
04_final_merged/       deployable single-checkpoint model
report.json            perplexities, losses, and every probe output
```

## Notes

Both scripts filter trainer kwargs by signature (`_keep_supported`) rather than
pinning a `trl` version — TRL keeps moving arguments between `SFTTrainer` and
`SFTConfig` across releases, and this way the scripts survive it. They also
prefer `UnslothTrainer` when available, for its native
`embedding_learning_rate`, and fall back to plain TRL otherwise.
