# finetuning

Does the order of **CPT** (continued pre-training) and **SFT** (supervised
fine-tuning) matter? Two scripts, identical except for stage order.

| script | order | verdict |
|---|---|---|
| [`cpt_then_sft.py`](cpt_then_sft.py) | CPT → SFT | the recipe that holds up |
| [`sft_then_cpt.py`](sft_then_cpt.py) | SFT → CPT | the ablation |

Same base model (`unsloth/SmolLM-135M`), same corpora (SEC 10-K prose +
`virattt/financial-qa-10K`), same seed, same step counts. Run both, diff the
two `report.json` files.

## The two stages

|  | CPT | SFT |
|---|---|---|
| objective | next-token, loss on **every** token | Alpaca pairs, loss on **response span only** |
| LoRA rank | 32 | 16 |
| targets | linear + `embed_tokens` + `lm_head` | linear only |
| packing | on | off (masking needs 1 example/sequence) |
| why | new domain vocabulary — *EBITDA*, *diluted EPS* | vocabulary is fine; only behaviour changes |

CPT mixes in 20% general text (WikiText) against catastrophic forgetting and
gives embeddings a 10× lower LR. SFT renders 25% of rows without the retrieved
context, against format overfitting. The stage-1 adapter is merged before
stage 2 starts.

## Why the order matters

**CPT → SFT**: CPT teaches the model to *sound* like the domain, SFT teaches it
to *answer*. SFT last leaves instruction-following freshest in the weights.

**SFT → CPT**: all-token loss on raw prose has no reason to preserve
response-only behaviour — it pushes *every* position toward continuing a 10-K,
including the ones right after `### Response:`. Expect comparable domain
perplexity but probes that ramble and never emit EOS. `sft_then_cpt.py` tracks
`eos_rate` per stage to measure it.

## Evaluated at every stage boundary

Domain perplexity (did adaptation work?) · general perplexity (how much did it
forget?) · completion probes · alpaca probes (does it answer *and stop*?) ·
bare-question probe (template overfitting?) · `eos_rate`. All to `report.json`.

## Usage

```bash
pip install unsloth trl peft transformers datasets accelerate bitsandbytes

python cpt_then_sft.py            # ~15 min on a T4
python cpt_then_sft.py --smoke    # 30 steps/stage, checks the wiring
```

Needs a CUDA GPU — Unsloth has no Apple MPS support. Both scripts take the same
flags: `--base`, `--cpt-steps` / `--sft-steps`, `--general-ratio`,
`--format-mix`, `--load-in-4bit`, `--keep-intermediate`. Use matching step
counts on both sides or the comparison means nothing.

## Publishing

```bash
huggingface-cli login             # WRITE scope
python cpt_then_sft.py --push-to-hub
```

Pushes the merged checkpoint plus a generated model card to
[huggingface.co/sidhusarkar](https://huggingface.co/sidhusarkar). Also:
`--push-to-hub user/repo`, `--push-adapters`, `--hf-private`, `--hf-token`.

## Artifacts

`runs/<name>/` holds both stage adapters, `04_final_merged/` (deployable), and
`report.json`. The intermediate merge is deleted unless `--keep-intermediate`.
