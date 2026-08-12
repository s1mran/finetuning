# finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s1mran/finetuning/blob/main/run_in_colab.ipynb)

Post-training experiments on `unsloth/SmolLM-135M`, each one built to answer a
single question and to show its working. Every run prints real training strings
before it trains, probes the model at every stage boundary, and writes a
`report.json` you can diff against a sibling run.

```
CPT/   src/ …  reports/     CPT-first ordering
SFT/   src/ …  reports/     SFT-first ordering (the ablation)
RLHF/  src/ …  reports/     preference alignment (SFT -> DPO)
RLVR/  src/ …  reports/     verifiable rewards (SFT cold start -> GRPO)
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

### Verifiable rewards: SFT cold start → GRPO

[`RLVR/src/sft_then_grpo.py`](RLVR/src/sft_then_grpo.py) — GSM8K with Will
Brown's reward stack.

GRPO computes each completion's advantage against the mean of its own group:

```
A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)
```

When every completion in a group scores the same, `A_i` is zero for all of
them and the update is exactly zero — not small, zero. A base model that never
emits `<reasoning>` scores 0.0 on every reward including the graduated tag
count, so the group always agrees with itself and training is a no-op with a
flat loss curve. RL amplifies behaviour; it cannot invent it.

So stage 1 SFTs the target format in from GSM8K's own worked solutions before
GRPO runs — the cold start in the R1 recipe. Then `reward_audit` measures mean
within-group reward std and **refuses to start GRPO if it is ~0**, naming what
to change, rather than spending an hour to draw a flat line. `--force-grpo`
overrides; `--skip-sft` reproduces the failure on purpose.

135M is below the practical floor here and the gate will usually say so. For a
run that moves: `--base unsloth/Qwen2.5-1.5B-Instruct`.

## Evaluated at every stage boundary

Domain perplexity (did adaptation work?) · general perplexity (how much did it
forget?) · completion probes · alpaca probes (does it answer *and stop*?) ·
bare-question probe (template overfitting?) · `eos_rate`. All to `report.json`.

Before any training, each script prints one or two **real rendered training
strings** for every split it built — with and without an `### Input` block, and
domain vs general replay. A renamed column or an empty context is invisible in
a loss curve and obvious in the strings.

## Usage

Two notebooks:

- [`run_rlvr.ipynb`](run_rlvr.ipynb) — GRPO with the cold start, the
  reward-variance gate, and the failure reproduced on purpose.
- [`medical_cpt_sft.ipynb`](medical_cpt_sft.ipynb) — the medical ordering pair,
  ~40 min, with an auto-fallback runner and publish cells for HF, GitHub and
  Drive.
- [`run_medical.ipynb`](run_medical.ipynb) — the same pair plus the RLHF run,
  one script per cell.
- [`run_in_colab.ipynb`](run_in_colab.ipynb) — all five experiments.

Otherwise:

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

## Design notes

Configuration follows the class notebooks (`class_9a_f`, `class_9b`,
`class_10a`): 4-bit loading, 512-token sequences, `HuggingFaceTB/SmolLM-135M`,
cosine schedule, and `load_best_model_at_end` on `eval_loss` so a stage that
starts overfitting ships its best checkpoint rather than its last.

Two things the notebooks got right that earlier versions of these scripts did
not:

- **Split documents before chunking.** Chunking first and splitting after puts
  overlapping windows of the same filing on both sides of the split, so ~21% of
  eval vocabulary is verbatim in training and the perplexity is partly
  memorisation.
- **Sample when probing.** Greedy decoding on a 135M model collapses into
  repeat loops that read as degeneration but are a decoding artefact.

The finance CPT corpus is `PleIAs/SEC` — full 10-K filings, plain parquet.
It replaces `eloukas/edgar-corpus`, which is script-based and no longer
loadable; its fallback was `virattt/financial-qa-10K`, the same corpus stage 2
trains on, which quietly made the ordering comparison meaningless.
