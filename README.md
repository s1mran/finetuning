# finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s1mran/finetuning/blob/main/run_in_colab.ipynb)

Post-training experiments on `unsloth/SmolLM-135M`, each one built to answer a
single question and to show its working. Every run prints real training strings
before it trains, probes the model at every stage boundary, and writes a
`report.json` you can diff against a sibling run.

```
CPT/   src/ …  reports/     CPT-first ordering
SFT/   src/ …  reports/     SFT-first ordering (the ablation)
RLHF/  src/ …  reports/     preference alignment
```

Artifacts land in `<TYPE>/reports/<run>/`, resolved from the script's own
location — so it does not matter what directory you launch from. `report.json`
is tracked; the adapters and merged checkpoints beside it are gitignored.

## The experiments

### Does CPT → SFT beat SFT → CPT?

| script | order | domain |
|---|---|---|
| [`CPT/src/cpt_fin_then_sft.py`](CPT/src/cpt_fin_then_sft.py) | CPT → SFT | SEC 10-K financial text |
| [`SFT/src/sft_fin_then_cpt.py`](SFT/src/sft_fin_then_cpt.py) | SFT → CPT | SEC 10-K financial text |
| [`CPT/src/cpt_med_then_sft.py`](CPT/src/cpt_med_then_sft.py) | CPT → SFT | medical guidelines |
| [`SFT/src/sft_med_then_cpt.py`](SFT/src/sft_med_then_cpt.py) | SFT → CPT | medical guidelines |

Same base model, same seed, same step counts — only the stage order changes.
Run a matched pair and diff the two `report.json` files.

|  | CPT | SFT |
|---|---|---|
| objective | next-token, loss on **every** token | Alpaca pairs, loss on **response span only** |
| LoRA rank | 32 | 16 |
| targets | linear + `embed_tokens` + `lm_head` | linear only |
| packing | on | off (masking needs 1 example/sequence) |
| why | new domain vocabulary — *EBITDA*, *myocardial infarction* | vocabulary is fine; only behaviour changes |

CPT mixes in 20% general text (WikiText) against catastrophic forgetting and
gives embeddings a 10× lower LR. SFT renders 25% of rows without the retrieved
context, against format overfitting. The stage-1 adapter is merged before
stage 2 starts.

**CPT → SFT**: CPT teaches the model to *sound* like the domain, SFT teaches it
to *answer*. SFT last leaves instruction-following freshest in the weights.

**SFT → CPT**: all-token loss on raw prose has no reason to preserve
response-only behaviour — it pushes *every* position toward continuing a
filing, including the ones right after `### Response:`. Expect comparable domain
perplexity but probes that ramble and never emit EOS; `eos_rate` measures it.

### Preference alignment: SFT → DPO

[`RLHF/src/sft_then_dpo.py`](RLHF/src/sft_then_dpo.py) — SFT on empathetic
dialogue, then DPO on human preference pairs.

The DPO loss, and why each part is there, is documented inline above
`make_dpo_trainer`. Short version:

```
L_DPO = -log σ( β · ( r(x, yw) − r(x, yl) ) )     r(x,y) = log π_θ(y|x) − log π_ref(y|x)
```

No reward model is trained; the policy's own log-ratio against a frozen
reference *is* the reward. `β` is the KL leash. The reference is the merged
stage-1 SFT model, recovered for free by disabling the adapter.

**Reward hacking.** There is no learned reward model to game, but DPO still
optimises a proxy — annotator preference — and the proxy has known holes. Each
is guarded *and measured*, because a guard you cannot see the effect of is a
guard you should not trust:

| failure | guard | metric |
|---|---|---|
| length exploitation | `--max-length-ratio` drops pairs whose chosen response is disproportionately longer | mean response tokens per stage |
| drift from reference | `--beta` (KL leash) | general perplexity per stage |
| degeneration | — | 4-gram repetition rate per stage |
| overfitting the preference set | held-out split | preference accuracy per stage |

The run ends with a before/after table and an explicit warning if mean response
length grew more than 1.5×.

One thing stated plainly: there is no public preference dataset *about empathy*.
The empathy signal comes from stage 1 (`Estwld/empathetic_dialogues_llm`);
stage 2 uses general preference data for response quality. Do not read the DPO
stage as "learning to be kind".

## Evaluated at every stage boundary

Domain perplexity (did adaptation work?) · general perplexity (how much did it
forget?) · completion probes · alpaca probes (does it answer *and stop*?) ·
bare-question probe (template overfitting?) · `eos_rate`. All to `report.json`.

Before any training, each script prints one or two **real rendered training
strings** for every split it built — with and without an `### Input` block, and
domain vs general replay. A renamed column or an empty context is invisible in
a loss curve and obvious in the strings.

## Usage

Easiest path is [`run_in_colab.ipynb`](run_in_colab.ipynb) — setup, runs, and a
comparison cell, one step at a time. Otherwise:

```bash
pip install unsloth trl peft transformers datasets accelerate bitsandbytes

python CPT/src/cpt_fin_then_sft.py --smoke    # 30 steps/stage, checks the wiring
python CPT/src/cpt_fin_then_sft.py            # ~15 min on a T4
python RLHF/src/sft_then_dpo.py               # ~20 min on a T4
```

Needs a CUDA GPU — Unsloth has no Apple MPS support. The ordering scripts share
`--base`, `--cpt-steps` / `--sft-steps`, `--general-ratio`, `--format-mix`,
`--load-in-4bit`, `--keep-intermediate`. Use matching step counts across a pair
or the comparison means nothing. The DPO script adds `--beta`,
`--max-length-ratio`, `--sft-lr`, `--dpo-lr`.

## Publishing

```bash
huggingface-cli login             # WRITE scope
python CPT/src/cpt_fin_then_sft.py --push-to-hub
```

Pushes the merged checkpoint plus a generated model card to
[huggingface.co/sidhusarkar](https://huggingface.co/sidhusarkar). Also:
`--push-to-hub user/repo`, `--push-adapters`, `--hf-private`, `--hf-token`.

To publish a model you already trained, without retraining:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("user/repo", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="CPT/reports/cpt_fin_then_sft/04_final_merged",
                  repo_id="user/repo", repo_type="model")
```

## Known limitation

`eloukas/edgar-corpus` — the raw 10-K MD&A prose the finance CPT stage wants —
is script-based, and `datasets` dropped support for script loaders. Every EDGAR
mirror checked has the same problem, so the finance runs fall back to the
`context` column of `virattt/financial-qa-10K`, **the same corpus their SFT
stage trains on**. That is consistent with the weak result observed in practice
(domain perplexity 22.0 → 19.9 → 20.3, rather than the sharp drop the design
predicts). The medical runs do not have this problem — `epfl-llm/guidelines` is
genuinely distinct from the ChatDoctor SFT data — so treat those as the more
informative ordering comparison until a loadable EDGAR mirror turns up.
